import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Literal, Tuple
from transformers import CLIPImageProcessor
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomCrop
from utils import compute_edge_map


class VitonHDDataset(Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (384, 512)
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.width = size[0]
        self.height = size[1]
        self.size = size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.5], [0.5])

        self.order = order
        im_names = []
        c_names = []
        dataroot_names = []

        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.clip_processor = CLIPImageProcessor()

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).convert("RGB").copy()
        image = Image.open(os.path.join(self.dataroot, self.phase, "image", im_name)).convert("RGB").copy()
        if random.random() > 0.2:
            rect_mask = Image.open(os.path.join(self.dataroot, self.phase, "rect_mask", im_name)).convert("L").copy()
            rect_mask_np = np.array(rect_mask).astype(np.float32)
            rect_mask_np = ((rect_mask_np > 200) * 255).astype(np.uint8)
            # 定义膨胀核（你可以调整大小，例如 (5, 5)）
            kernel = np.ones((3, 3), np.uint8)
            # 应用膨胀操作
            rect_mask_np = cv2.dilate(rect_mask_np, kernel, iterations=random.randint(1, 10))
            rect_mask = Image.fromarray(rect_mask_np.astype(np.uint8), mode='L')
        else:
            rect_mask = Image.open(os.path.join(self.dataroot, self.phase, "rectangle_mask", im_name)).convert("L").copy()
        densepose_map = Image.open(os.path.join(self.dataroot, self.phase, "image-densepose", im_name)).convert("RGB").copy()
        upper_mask = Image.open(os.path.join(self.dataroot, self.phase, "gt_cloth_warped_mask", im_name)).convert("L").copy()
        # upper_mask = Image.open(os.path.join(self.dataroot, self.phase, "upper_mask", im_name.replace('.jpg','.png'))).convert("L").copy()
        
        cloth = cloth.resize((self.width, self.height))
        image = image.resize((self.width, self.height))
        rect_mask = rect_mask.resize((self.width, self.height))
        densepose_map = densepose_map.resize((self.width, self.height))
        upper_mask = upper_mask.resize((self.width, self.height))

        if self.phase == "train":
            # Apply transforms consistently
            if random.random() > 0.5:
                cloth = TF.hflip(cloth)
                image = TF.hflip(image)
                rect_mask = TF.hflip(rect_mask)
                densepose_map = TF.hflip(densepose_map)
                upper_mask = TF.hflip(upper_mask)
            if random.random() > 0.5:

                i, j, h, w = RandomCrop.get_params(
                                image, output_size=(int(self.height * random.randint(75, 99) / 100),
                                                    int(self.width * random.randint(75, 99) / 100)))
                
                image = TF.crop(image, i, j, h, w)
                cloth = TF.crop(cloth, i, j, h, w)
                rect_mask = TF.crop(rect_mask, i, j, h, w)
                upper_mask = TF.crop(upper_mask, i, j, h, w)
                densepose_map = TF.crop(densepose_map, i, j, h, w)
                
                cloth = cloth.resize((self.width, self.height))
                image = image.resize((self.width, self.height))
                rect_mask = rect_mask.resize((self.width, self.height))
                densepose_map = densepose_map.resize((self.width, self.height))
                upper_mask = upper_mask.resize((self.width, self.height))
                
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)  # 小角度扰动，单位为度
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                cloth = TF.rotate(cloth, angle, interpolation=TF.InterpolationMode.BILINEAR)
                rect_mask = TF.rotate(rect_mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                upper_mask = TF.rotate(upper_mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                densepose_map = TF.rotate(densepose_map, angle, interpolation=TF.InterpolationMode.BILINEAR)
            if random.random() > 0.5:
                # Apply color jitter only to RGB images
                brightness = random.uniform(0.1, 0.5)
                contrast = random.uniform(0.1, 0.5)
                saturation = random.uniform(0.1, 0.5)
                hue = random.uniform(0.0, 0.2)
                color_jitter = transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
                cloth = color_jitter(cloth)
                image = color_jitter(image)

        cloth_tensor = self.transform(cloth)
        image_tensor = self.transform(image)

        rect_mask_tensor = self.toTensor(rect_mask)
        rect_mask_tensor = rect_mask_tensor[:1]
        rect_mask_tensor[rect_mask_tensor < 0.5] = 0
        rect_mask_tensor[rect_mask_tensor >= 0.5] = 1
        im_rect = image_tensor * ( 1 - rect_mask_tensor)
        upper_mask_tensor = self.toTensor(upper_mask)
        upper_mask_tensor = upper_mask_tensor[:1]
        upper_mask_tensor[upper_mask_tensor < 0.5] = 0
        upper_mask_tensor[upper_mask_tensor >= 0.5] = 1
        white = torch.ones_like(image_tensor)
        im_upper = image_tensor * upper_mask_tensor + (1 - upper_mask_tensor) * white
        pose_img_tensor = self.transform(densepose_map)

        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["image"] = image_tensor
        result["cloth"] = cloth_tensor
        result["cloth-edge"] = compute_edge_map(cloth_tensor)
        result["cloth-clip"] = self.clip_processor(images=cloth, return_tensors="pt").data['pixel_values']
        result["rect_mask"] = rect_mask_tensor
        result["image_rect"] = im_rect
        result["image-densepose"] = pose_img_tensor
        result["image-upper-mask"] = upper_mask_tensor
        result["image-upper"] = im_upper

        return result

    def __len__(self):
        return len(self.im_names)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = VitonHDDataset(
        dataroot_path="./zalando-hd-resized",
        phase="train",
        order="pairs",
        size=(384, 512),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data["c_name"][0])
        print(data["im_name"][0])
        print(data["image"].shape)
        print(data["cloth"].shape)
        print(data["rect_mask"].shape)
        print(data["image-densepose"].shape)
        print(data["image-upper-mask"].shape)
        print(data["image-upper"].shape)
        if i == 10:
            break
        # Uncomment the following lines to save the images
    # print()
    # transforms.ToPILImage()(dataset[0]["image"]).save("output/image.png")
    # transforms.ToPILImage()(dataset[0]["garment"]).save("output/garment.png")
    # transforms.ToPILImage()(dataset[0]["pose_img"]).save("output/pose_img.png")
