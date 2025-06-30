import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import (
    is_torch_xla_available,
    logging
)
from src.appearance_flow import AppearanceFlowEncDec
from utils import compute_edge_map

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


class AflowEncDecPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin, FromSingleFileMixin):
    
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]
    def __init__(
            self,
            aflow: AppearanceFlowEncDec,
    ):
        super().__init__()
        self.register_modules(
            aflow=aflow,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def check_inputs(
            self,
            height,
            width,
            # callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(
            self,
            height: Optional[int] = None,
            width: Optional[int] = None,
            cloth_image=None,
            pose_image=None
    ):
        r"""
        Args:
            height (int, optional): The height of the image to generate. Defaults to None.
            width (int, optional): The width of the image to generate. Defaults to None.
            cloth_image (torch.Tensor, optional): The cloth image to use. Defaults to None.
            pose_image (torch.Tensor, optional): The pose image to use. Defaults to None.
        """
        # 1. Check inputs
        self.check_inputs(height, width)  # , callback_on_step_end_tensor_inputs)

        if cloth_image is None or pose_image is None:
            raise ValueError("必须传入 cloth_image 和 pose_image 才能执行生成流程。")
        
        # 2. prepare cloth and pose latents
        cloth = self.image_processor.preprocess(
            cloth_image, 
            height=height, 
            width=width).to(self.device)
        
        pose_image = self.image_processor.preprocess(
            pose_image,
            height=height,
            width=width).to(self.device)
        
        # 3. warp cloth image to coordinate with pose image
        cloth_edge = compute_edge_map(cloth.squeeze(0))
        cloth_edge = cloth_edge.unsqueeze(0)
        flow, warped_img, mask = self.aflow(pose_image, cloth, cloth_edge)
        warp_out = self.image_processor.postprocess(warped_img)
        mask_out = self.image_processor.postprocess(mask)
        cloth_edge = self.image_processor.postprocess(cloth_edge)
        return {"images": warp_out, "flows": flow, "masks": mask_out, "clothes_edge": cloth_edge}
