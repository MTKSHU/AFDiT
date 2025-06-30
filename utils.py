import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2

from PIL import Image


def compute_edge_map(img_tensor: torch.Tensor):
    # Ensure the input is an RGB image
    assert img_tensor.shape[0] == 3, "Input tensor must have 3 channels (RGB)"
    
    # Convert the RGB image to grayscale using the luminance formula.
    grayscale = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
    
    # Prepare the grayscale image for convolution.
    gray_unsq = grayscale.unsqueeze(0).unsqueeze(0)

    # Define Sobel kernels for edge detection.
    sobel_kernel_x = torch.tensor(
        [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]],
        dtype=img_tensor.dtype, device=img_tensor.device
    ).unsqueeze(0).unsqueeze(0)
    
    sobel_kernel_y = torch.tensor(
        [[-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]],
        dtype=img_tensor.dtype, device=img_tensor.device
    ).unsqueeze(0).unsqueeze(0)
    
    # Apply the Sobel filters to compute image gradients.
    grad_x = F.conv2d(gray_unsq, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(gray_unsq, sobel_kernel_y, padding=1)
    
    # Compute the gradient magnitude.
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0).squeeze(0)
    
    # Normalize the gradient magnitude to [0, 1] range.
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    normalized = (grad_magnitude - grad_min) / (grad_max - grad_min + 1e-8)
    
    # For white cloth on a white background, subtle edges may be too weak.
    # You can enhance them via gamma correction. Using gamma < 1 amplifies tiny differences.
    gamma = 0.5
    enhanced = torch.clamp(normalized ** gamma, 0, 1)
    
    return enhanced.unsqueeze(0)
    

def focal_frequency_loss_fn(pred, target, alpha=1.0, gamma=1.0, eps=1e-8):
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    pred_fft = torch.fft.fft2(pred, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")
    diff_mag = torch.abs(pred_fft - target_fft)

    # 使用目标频谱的模长作为注意力权重
    # 引入 log1p 降低频谱极值对权重的影响
    weight = torch.log1p(torch.abs(target_fft)) ** gamma
    # 使用均值归一化 + 最大值裁剪，防止 weight 异常放大导致 loss 爆炸
    weight = weight / (torch.mean(weight, dim=(-2, -1), keepdim=True) + eps)
    
    # 加权 MSE
    loss = alpha * torch.mean(weight * (diff_mag ** 2))
    return loss


def gaussian(window_size: int, sigma: float):
    gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)  # [window_size, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # [1,1,window_size,window_size]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    # img1, img2 的 shape 为 [N, C, H, W]
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_loss_fn(pred, target, window_size=11, size_average=True):
    pred = pred.to(dtype=torch.float32) 
    target = target.to(dtype=torch.float32)
    # 返回 1 - SSIM 作为 loss 值，与其他 loss 联合使用
    return 1 - ssim(pred, target, window_size, size_average)


def dice_loss_fn(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - (num / denom).mean()


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, dtype=torch.float32):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(dtype=dtype)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids = None, device = 'cuda', dtype = torch.float32):
        super(VGGLoss, self).__init__()
        # self.register_buffer('vgg', Vgg19())
        self.vgg = Vgg19(dtype=dtype)
        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class CannyLoss(nn.Module):
    """
    基于 Sobel 算子近似 Canny 算子的边缘检测损失。
    如果输入为彩色图像，会先转换为灰度图，再计算边缘梯度。
    """
    def __init__(self, grayscale: bool = True):
        super(CannyLoss, self).__init__()
        self.grayscale = grayscale
        # 定义 Sobel 核，用于梯度计算
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: 预测图像，形状 (N, C, H, W)
        target: 真实图像，形状 (N, C, H, W)
        如果图像为彩色且 grayscale=True，则将其转换为灰度图:
        Y = 0.299R + 0.587G + 0.114B
        """
        if self.grayscale and pred.size(1) >= 3:
            pred = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # 使用反射填充处理边缘
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='reflect')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='reflect')
        
        # 计算 x 和 y 方向的梯度
        pred_gx = F.conv2d(pred_pad, self.sobel_x.to(pred_pad.device), padding=0)
        pred_gy = F.conv2d(pred_pad, self.sobel_y.to(pred_pad.device), padding=0)
        target_gx = F.conv2d(target_pad, self.sobel_x.to(target_pad.device), padding=0)
        target_gy = F.conv2d(target_pad, self.sobel_y.to(target_pad.device), padding=0)
        
        # 计算梯度幅值（加上一个较小的数以避免开方为0）
        pred_edge = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        target_edge = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)
        
        loss = F.l1_loss(pred_edge, target_edge)
        return loss


def fusion_pil_mask(
        ori_mask: Image.Image, 
        warped_mask: Image.Image, 
        iters: int=3):
    """
    膨胀融合两个二值掩码
    ori_mask: PIL Image, 二值掩码
    warped_mask: PIL Image, 二值掩码
    iters: int, 膨胀迭代次数
    return: PIL Image, 膨胀融合后的掩码
    """
    ori_mask = ori_mask.convert("L")
    warped_mask = warped_mask.convert("L")
    ori_mask_np = np.array(ori_mask).astype(np.float32)
    warped_mask_np = np.array(warped_mask).astype(np.float32)
    warped_mask_np = ((warped_mask_np > 200) * 255).astype(np.uint8)
    # 定义膨胀核（你可以调整大小，例如 (5, 5)）
    kernel = np.ones((3, 3), np.uint8)
    # 应用膨胀操作
    warped_mask_np = cv2.dilate(warped_mask_np, kernel, iterations=iters)
    
    fusion_np = np.clip(ori_mask_np + warped_mask_np, 0, 255)
    fusion_np = cv2.dilate(fusion_np, kernel, iterations=1)
    fusion_mask = Image.fromarray(fusion_np.astype(np.uint8))
    warped_mask = Image.fromarray(warped_mask_np.astype(np.uint8))
    return fusion_mask, warped_mask


def fusion_tensor_mask(mask1: torch.Tensor, warped_mask: torch.Tensor, iters=3, threshold=127) -> torch.Tensor:
    """
    融合 mask1（二值）和 warped_mask（0~1），返回 0.0 或 1.0 的 float mask，支持膨胀。
    
    参数:
        mask1: Tensor[..., H, W]，值为 0 或 1
        warped_mask: Tensor[..., H, W]，值为 0~1 的 float
        iters: 膨胀次数
        threshold: 二值化阈值（默认127）
    返回:
        Tensor[..., H, W]，float32，值为 0.0 或 1.0
    """
    original_shape = mask1.shape
    device = mask1.device

    # 确保为 float
    mask1 = mask1.float()
    warped_mask = warped_mask.float()

    # 融合（取 max 或相加 clip）
    fusion = torch.clamp(mask1 + warped_mask, 0.0, 1.0)

    # 转 numpy，缩放到 0~255
    fusion_np = (fusion.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    fusion_np = cv2.dilate(fusion_np, kernel, iterations=iters)

    # 二值化并转 float32（0.0 或 1.0）
    binary_np = (fusion_np >= threshold).astype(np.float32)

    # 转回 tensor 并还原 shape 和 device
    fusion_tensor = torch.from_numpy(binary_np).to(device)
    fusion_tensor = fusion_tensor.view(*original_shape)

    return fusion_tensor
