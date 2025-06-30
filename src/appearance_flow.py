import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.autoencoders.autoencoder_kl import Decoder
from diffusers.models.embeddings import PatchEmbed
from diffusers.models import ModelMixin


def active_function_selector(act_fn: str = "relu"):
    if act_fn == "relu":
        return nn.ReLU(inplace=True)
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act_fn == "silu":
        return nn.SiLU(inplace=True)
    elif act_fn == "tanh":
        return nn.Tanh()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class AttentionBlock(nn.Module):
    def __init__(self, skip_channels, gating_channels=None):
        super(AttentionBlock, self).__init__()
        if gating_channels is None:
            gating_channels = skip_channels
        self.skip_channels = skip_channels
        self.gating_channels = gating_channels
        reduced_dim = skip_channels // 8 if skip_channels >= 8 else 1
        self.query_conv = nn.Conv2d(skip_channels, reduced_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(gating_channels, reduced_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, skip, gating=None):
        B, C, H, W = skip.size()
        if gating is None:
            gating = skip
        proj_query = self.query_conv(skip).view(B, -1, H * W)  # (B, reduced_dim, N)
        # Use gating tensor's own spatial dims.
        proj_key = self.key_conv(gating).view(B, -1, gating.size(2) * gating.size(3))  # (B, reduced_dim, N')
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # (B, N, N')
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(skip).view(B, C, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + skip

class ConvBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int=3, 
            out_channels: int=64,
            act_fn: str="relu",
            use_attn: bool=False):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            active_function_selector(act_fn),
        )
        if use_attn:
            self.attn = AttentionBlock(skip_channels=out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        return self.attn(self.block(x))


class UpConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            use_attn: bool=False
        ):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels=out_channels*2, out_channels=out_channels)
        if use_attn:
            self.attn = AttentionBlock(skip_channels=out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x, skip):
        x = self.up(x)
        # Concatenate with skip connection (assumed to have matching spatial dims)
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.conv(x))



class DecodeConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            use_attn: bool=False
        ):
        super(DecodeConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels=out_channels, out_channels=out_channels)
        if use_attn:
            self.attn = AttentionBlock(skip_channels=out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x):
        return self.attn(self.conv(self.up(x)))


def warp_cloth(cloth, flow):
    """
    Warps the clothing image using the flow field.
    
    cloth: Tensor of shape (N, C, H, W), the clothing image.
    flow: Tensor of shape (N, 2, H, W), predicted flow field in pixel units.
    
    Returns:
    warped cloth tensor of shape (N, C, H, W)
    """
    weight_dtype = cloth.dtype
    N, C, H, W = cloth.size()
    # create a normalized mesh grid in range [-1, 1]
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=cloth.device), 
                                    torch.linspace(-1, 1, W, device=cloth.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1)  # shape (H, W, 2)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # shape (N, H, W, 2)
    
    # Convert flow from pixel units to normalized coordinates
    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0, :, :] = flow[:, 0, :, :] / ((W - 1) / 2)
    norm_flow[:, 1, :, :] = flow[:, 1, :, :] / ((H - 1) / 2)
    norm_flow = norm_flow.permute(0, 2, 3, 1)  # shape (N, H, W, 2)
    
    grid_warp = grid + norm_flow
    # Use grid_sample to warp the clothing image
    if cloth.device.type == 'mps':
        padding_mode = "zeros"
    else:
        padding_mode = "border"
    cloth = cloth.to(dtype=weight_dtype)
    grid_warp = grid_warp.to(dtype=weight_dtype)
    warped_cloth = F.grid_sample(cloth, grid_warp, padding_mode=padding_mode, align_corners=True)
    return warped_cloth, grid_warp


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveFeatureFusion, self).__init__()
        self.weight_x = nn.Parameter(torch.ones(num_features, 1, 1))
        self.weight_y = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x, y):
        return x * self.weight_x + y * self.weight_y + self.bias


class ChannelsIdentity(nn.Module):
    def __init__(self):
        super(ChannelsIdentity, self).__init__()
    
    def forward(self, x, y):
        return x, y


class AppearanceFlowEncDec(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
            self, 
            pose_channels: int=3, 
            garm_channels: int=3,
            feature_channels: int=64,
            num_blocks: int=4,
            act_fn: str="relu",
            attn_layers: List[bool]=[False, False, True, True],
            mid_block_attn: bool=True,
            out_channels: int=2,
            out_kernel_size: int=7,
            out_stride: int=1,
            out_padding: int=3
        ):
        """
        A deeper UNet-style architecture with cross attention skip connections.
        """
        super(AppearanceFlowEncDec, self).__init__()
        assert len(attn_layers) == num_blocks
        self.num_blocks = num_blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.pose_enc = nn.ModuleList([
            ConvBlock(
                pose_channels if idx == 0 else feature_channels * 2 ** (idx - 1), 
                feature_channels * 2 ** idx, 
                act_fn, 
                attn_layers[idx]
            ) for idx in range(num_blocks)
        ])
        self.garm_enc = nn.ModuleList([
            ConvBlock(
                garm_channels if idx == 0 else feature_channels * 2 ** (idx - 1), 
                feature_channels * 2 ** idx, 
                act_fn, 
                attn_layers[idx]
            ) for idx in range(num_blocks)
        ])
        self.enc_attn = nn.ModuleList([
            AdaptiveFeatureFusion(num_features=feature_channels * 2 ** (idx))
            if idx in (0, 1) else
            AttentionBlock(
                skip_channels=feature_channels * 2 ** (idx),
                gating_channels=feature_channels * 2 ** (idx)
            ) for idx in range(num_blocks)
        ])
        # Bottleneck
        self.bottleneck = ConvBlock(
            feature_channels * 2 ** (num_blocks - 1), 
            feature_channels * 2 ** num_blocks, 
            act_fn, 
            mid_block_attn
        )

        # Decoder
        self.dec = nn.ModuleList([
            DecodeConvBlock(
                feature_channels * 2 ** (num_blocks - idx), 
                feature_channels * 2 ** (num_blocks - idx - 1), 
                attn_layers[num_blocks - idx - 1]
            ) for idx in range(num_blocks)
        ])
        
        # Final prediction: 2-channel flow field
        self.out_flow = nn.Conv2d(
            feature_channels, 
            out_channels, 
            kernel_size=out_kernel_size, 
            stride=out_stride, 
            padding=out_padding)
        self.out_mask = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
    
    def encode(self, pose, garm, edge):
        # Encoder
        garm_inp = torch.cat([garm, edge], dim=1)
        enc_x = []
        for idx in range(self.num_blocks):
            pose = self.pose_enc[idx](pose if idx == 0 else self.pool(pose))
            garm_inp = self.garm_enc[idx](garm_inp if idx == 0 else self.pool(garm_inp))
            x = self.enc_attn[idx](pose, garm_inp)
            pose = pose + x
            garm_inp = garm_inp + x
            enc_x.append(x)
        
        # Bottleneck
        d = self.bottleneck(self.pool(x))
        return enc_x[-1], d

    def decode(self, d):
        # Decoder
        for idx in range(self.num_blocks):
            d = self.dec[idx](d)
        
        flow = self.out_flow(d)
        return flow

    def forward(self, pose, garm, edge):
        # Encoder
        enc_out, d = self.encode(pose, garm, edge)

        # Decoder
        flow = self.decode(d)
        warped_img, grid_warp = warp_cloth(garm, flow)
        
        mask_features = torch.cat([warped_img, flow], dim=1)
        mask = self.out_mask(mask_features)
        mask = torch.sigmoid(mask)
        return flow, warped_img, mask


if __name__ == "__main__":
    from torchinfo import summary

    # # Create an instance of the AppearanceFlow model
    appearance_flow_model = AppearanceFlowEncDec(
        pose_channels=3,
        garm_channels=3, 
        feature_channels=64,
        num_blocks=4,
        act_fn="silu",
        attn_layers=[False, False, True, True],
        mid_block_attn=True
    )

    print(appearance_flow_model)
    summary(appearance_flow_model, input_size=[(1, 3, 512, 512), (1, 3, 512, 512)], device="cpu")
    appearance_flow_model.save_pretrained("appearance_flow")
