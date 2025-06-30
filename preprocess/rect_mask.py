import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from dwpose import DWposeDetector
from humanparsing.run_parsing import Parsing
from utils_mask import get_mask_location

from multiprocessing import Pool, cpu_count

def process_one_image(file_path):
    im_dict, pose_image = get_mask(
        file_path,
        category="Upper-body",
        offset_top=0,
        offset_bottom=0,
        offset_left=0,
        offset_right=0
    )
    mask_array = im_dict["layers"][0][:, :, 3]
    mask_img = Image.fromarray(mask_array)
    mask_img, _, _ = pad_and_resize(mask_img, new_width=768, new_height=1024, pad_color=(0, 0, 0))
    mask_img = mask_img.convert("L")
    mask_img.save(output_dir / file_path.name)


def pad_and_resize(im: Image.Image,
                   new_width: int = 768,
                   new_height: int = 1024,
                   pad_color: tuple = (255, 255, 255),
                   mode=Image.LANCZOS) -> tuple[Image.Image, int, int]:
    """
    Resize and pad an image with the specified pad color.

    Returns:
        new_im: The padded image.
        pad_w: The horizontal padding value.
        pad_h: The vertical padding value.
    """
    old_width, old_height = im.size

    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)

    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h


def unpad_and_resize(padded_im: Image.Image,
                     pad_w: int,
                     pad_h: int,
                     original_width: int,
                     original_height: int) -> Image.Image:
    """
    Remove padding from the image and resize to the original dimensions.
    """
    width, height = padded_im.size
    cropped_im = padded_im.crop((pad_w, pad_h, width - pad_w, height - pad_h))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)
    return resized_im


def resize_image(img: Image.Image, target_size: int = 768) -> Image.Image:
    """
    Resize image proportionally so that the minimum edge is equal to target_size.
    """
    width, height = img.size
    scale = target_size / min(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return img.resize((new_width, new_height), Image.LANCZOS)


def get_mask(vton_img_path: Path,
             category: str = "Upper-body",
             offset_top: int = 0,
             offset_bottom: int = 0,
             offset_left: int = 0,
             offset_right: int = 0) -> tuple[dict, Image.Image]:
    """
    Generate mask for the specified category and offsets using DWpose and Parsing models.

    Returns:
        A tuple (im, pose_image) where im is a dict containing 'background',
        'layers', and 'composite', and pose_image is the pose image.
    """
    with torch.inference_mode():
        vton_img = Image.open(vton_img_path)
        vton_img_det = resize_image(vton_img)
        # Process the image using DWposeDetector and Parsing
        pose_image_np, keypoints, _, candidate = dwprocessor(np.array(vton_img_det)[:, :, ::-1])
        candidate[candidate < 0] = 0
        candidate = candidate[0]
        candidate[:, 0] *= vton_img_det.width
        candidate[:, 1] *= vton_img_det.height

        # Convert pose image to PIL Image (RGB)
        pose_image = Image.fromarray(pose_image_np[:, :, ::-1])
        model_parse, _ = parsing_model(vton_img_det)

        mask, mask_gray = get_mask_location(
            category, model_parse, candidate,
            model_parse.width, model_parse.height,
            offset_top, offset_bottom, offset_left, offset_right
        )
        mask = mask.resize(vton_img.size)
        mask_gray = mask_gray.resize(vton_img.size)
        mask = mask.convert("L")
        mask_gray = mask_gray.convert("L")
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        im = {
            'background': np.array(vton_img.convert("RGBA")),
            'layers': [np.concatenate((np.array(mask_gray.convert("RGB")),
                                        np.array(mask)[:, :, np.newaxis]), axis=2)],
            'composite': np.array(masked_vton_img.convert("RGBA"))
        }
        return im, pose_image


def parse_image(file_path):
    vton_img = Image.open(file_path)
    vton_img_det = resize_image(vton_img)
    model_parse, _ = parsing_model(vton_img_det)
    return model_parse


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    # Define paths using pathlib for better path handling
    model_root = Path("../pretrained_models/BoyuanJiang/FitDiT")
    data_root = Path("../zalando-hd-resized/train")
    output_dir = data_root / "rectangle_mask"
    output_dir.mkdir(parents=True, exist_ok=True)

    parsing_model = Parsing(model_root=str(model_root), device="cpu")
    dwprocessor = DWposeDetector(model_root=str(model_root), device="cpu")
    # file_path = "../zalando-hd-resized/test/image/05524_00.jpg"
    # parse_img = parse_image(file_path)
    # parse_img.save("05524_parse_img.png")
    
    image_dir = data_root / "image"
    image_files = list(image_dir.iterdir())

    # for file_path in tqdm(image_files):
    #     im_dict, pose_image = get_mask(file_path,
    #                                    category="Upper-body",
    #                                    offset_top=0,
    #                                    offset_bottom=0,
    #                                    offset_left=0,
    #                                    offset_right=0)
    #     # Extract the mask (assumed to be in the 4th channel)
    #     mask_array = im_dict["layers"][0][:, :, 3]
    #     mask_img = Image.fromarray(mask_array)
    #     mask_img, _, _ = pad_and_resize(mask_img, new_width=768, new_height=1024, pad_color=(0, 0, 0))
    #     mask_img = mask_img.convert("L")
    #     mask_img.save(output_dir / file_path.name)
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_one_image, image_files), total=len(image_files)))