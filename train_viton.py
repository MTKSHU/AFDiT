#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import random
import shutil
import warnings
import random
import torch
import diffusers
import numpy as np

from contextlib import nullcontext
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from tqdm.auto import tqdm
from transformers import PretrainedConfig, CLIPVisionModelWithProjection, CLIPImageProcessor

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data.distributed import DistributedSampler

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from src.appearance_flow import AppearanceFlowEncDec, warp_cloth
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline

from utils import focal_frequency_loss_fn
from vitonhd_dataset import VitonHDDataset

if is_wandb_available():
    import wandb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
check_min_version("0.31.0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--repo", 
        type=str, 
        default="pretrained_models/BoyuanJiang/FitDiT",
        required=True, 
        help="Path or identifier of the pre-trained diffusers model repository.")
    parser.add_argument(
        "--aflow_repo",
        type=str,
        default="output/aflow_vton/enc_dec_1e-4_256x192/aflow",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--image_encoder_bigG_path",
        type=str,
        default="./pretrained_models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--image_encoder_large_path",
        type=str,
        default="./pretrained_models/openai/clip-vit-large-patch14",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None, required=False, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")

    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--max_train_steps", type=int, default=None, required=False)
    
    parser.add_argument("--dataroot_path", type=str, default="./zalando-hd-resized", help="Root path of the dataset.")
    parser.add_argument("--image_height", type=int, default=1024, required=False, help="Height of the input image.")
    parser.add_argument("--image_width", type=int, default=768, required=False, help="Width of the input image.")
    
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
    parser.add_argument("--scale_lr", type=bool, default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", type=bool, default=False)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--validation_epochs", type=int, default=20)
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        )
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use: adamw or prodigy.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for the learning rate scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor for the polynomial learning rate scheduler.")
    parser.add_argument("--weighting_scheme", type=str, default="mse", help="Weighting scheme for loss computation.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit-based timestep sampling.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Std deviation for logit-based timestep sampling.")
    parser.add_argument("--mode_scale", type=float, default=1.0, help="Mode scale factor for timestep sampling.")
    parser.add_argument("--precondition_outputs", action="store_true", help="Precondition model outputs if set.")
    parser.add_argument("--with_prior_preservation", action="store_true", help="Use prior preservation loss if set.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Loss weight for prior preservation.")
    parser.add_argument("--prodigy_beta3", type=float, default=0.999, help="Beta3 parameter for the Prodigy optimizer.")
    parser.add_argument("--prodigy_decouple", action="store_true", help="Decouple gradient updates in Prodigy optimizer.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Use bias correction in Prodigy optimizer.")
    parser.add_argument("--prodigy_safeguard_warmup", type=int, default=0, help="Warmup steps safeguard for Prodigy optimizer.")
    
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/aflow",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--visualize_outputs", action="store_true", help="Visualize model outputs if set.")
    parser.add_argument("--freq_loss", action="store_true", help="Use frequency loss if set.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    # 优化设置
    if args.allow_tf32 and torch.__version__ >= "2.0":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    seed_base = 2025
    set_seed(accelerator.process_index + seed_base)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Create dataset
    train_dataset = VitonHDDataset(
        dataroot_path=args.dataroot_path,
        phase="train",
        size=(args.image_width, args.image_height),
    )

    # Create sampler (for DDP)
    # train_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=accelerator.num_processes,
    #     rank=accelerator.process_index,
    #     shuffle=True,  # shuffle 内部控制，每个 epoch 打乱
    #     drop_last=False
    # )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        # sampler=train_sampler,
        # shuffle=False,  # 必须是 False，避免重复打乱
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        # pin_memory=True,
        # worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
    )

    # # Create validation dataset and dataloader (phase set to "validation")
    # val_dataset = VitonHDDataset(
    #     dataroot_path=args.dataroot_path,
    #     phase="test",
    #     size=(args.image_width, args.image_height),
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=False,
    #     num_workers=args.dataloader_num_workers,
    # )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.repo,
        subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
        args.repo,
        subfolder="transformer_garm",
        revision=args.revision,
        torch_dtype=weight_dtype)
    transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
        args.repo,
        subfolder="transformer_vton",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        device_map=None)
    aflow =  AppearanceFlowEncDec.from_pretrained(
        args.aflow_repo, 
        revision=args.revision, 
        torch_dtype=weight_dtype)
    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained( 
        "./pretrained_models/openai/clip-vit-large-patch14",
        revision=args.revision,
        torch_dtype=weight_dtype)
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
        "./pretrained_models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
        revision=args.revision,
        torch_dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.repo, 
        subfolder="vae",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    vit_processing = CLIPImageProcessor()

    transformer_vton.requires_grad_(True)

    transformer_garm.requires_grad_(False)
    aflow.requires_grad_(False)
    image_encoder_large.requires_grad_(False)
    image_encoder_bigG.requires_grad_(False)
    vae.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer_vton.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * \
            args.train_batch_size * accelerator.num_processes
        )

    transformer_vton_parameters_with_lr = {
        "params": transformer_vton.parameters(),
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_vton_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    # print(optimizer_class.__name__)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    transformer_garm, transformer_vton, aflow, image_encoder_large, image_encoder_bigG, vae, \
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer_garm, transformer_vton,  aflow, image_encoder_large, image_encoder_bigG, vae, \
                optimizer, train_dataloader, lr_scheduler
            )

    # Recalculate total training steps since dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers on main process.
    if accelerator.is_main_process:
        tracker_name = "aflow-vton-sd3"
        accelerator.init_trackers(tracker_name)   # , config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    # logger.info(f"  Sampler length (per process) = {len(train_sampler)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Load checkpoint if available.
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            checkpoint_path = os.path.join(args.output_dir, path)
            accelerator.load_state(
                input_dir=checkpoint_path,
                # state_objects={
                #     "transformer_vton": transformer_vton,
                #     "optimizer": optimizer,
                #     "lr_scheduler": lr_scheduler,
                # }
            )
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Helper functions used in loss computation.
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _get_clip_image_embeds(cloth):
        image_embeds_large = image_encoder_large(cloth).image_embeds
        image_embeds_bigG = image_encoder_bigG(cloth).image_embeds
        return torch.cat([image_embeds_large, image_embeds_bigG], dim=1)

    def prepare_image_latents(image):
        image_latents = vae.encode(image).latent_dist.sample()
        image_latents = (image_latents-vae.config.shift_factor) * vae.config.scaling_factor
        return image_latents

    # Define the loss computation function used for both training and validation.
    def compute_loss(batch):
        image = batch["image"].to(dtype=weight_dtype)
        garment = batch["cloth"].to(dtype=weight_dtype)
        agnostic_mask = batch["rect_mask"].to(dtype=weight_dtype)
        agnostic_image = batch["image_rect"].to(dtype=weight_dtype)
        pose_image = batch["image-densepose"].to(dtype=weight_dtype)
        garment_pixel = torch.stack([x.squeeze() for x in batch["cloth-clip"].to(dtype=weight_dtype)], dim=0)
        
        with torch.no_grad():
            # 1. Appearance flow encoder decoder pipeline.
            cloth_aflow_inp = image_processor.preprocess(
                garment, 
                height=256, 
                width=192)
            pose_aflow_inp = image_processor.preprocess(
                pose_image,
                height=256,
                width=192)
            cloth_aflow_edge = image_processor.preprocess(
                batch["cloth-edge"], 
                height=256, 
                width=192)
            cloth_aflow_inp = cloth_aflow_inp.to(image)
            cloth_aflow_edge = cloth_aflow_edge.to(image)
            pose_aflow_inp = pose_aflow_inp.to(image)
            warped_flow, warped_img, warped_mask = aflow(pose_aflow_inp, cloth_aflow_inp, cloth_aflow_edge)
            
            warped_img = torch.nn.functional.interpolate(
                warped_img, 
                size=(args.image_height, args.image_width), 
                mode='bilinear', 
                align_corners=False)
            warped_img = warped_img.to(image)
            # warped_mask = torch.nn.functional.interpolate(
            #     warped_mask, 
            #     size=(args.image_height, args.image_width), 
            #     mode='bilinear', 
            #     align_corners=False)
            
            
            # 2. model mask latents prepare.
            model_input = prepare_image_latents(image)
            masked_latents = prepare_image_latents(agnostic_image)
            
            mask = torch.stack(
                [
                    torch.nn.functional.interpolate(agnostic_mask, size=(args.image_height // 8, args.image_width // 8))
                ]
            )
            mask = mask.reshape(-1, 1, args.image_height // 8, args.image_width // 8)

            # 3. noise prepare.
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            latent_model_input = torch.cat([noisy_model_input, masked_latents, mask], dim=1)

            # 4. garment and aflow prepare.
            garm_model_latents = prepare_image_latents(garment)
            cloth_image_embeds = _get_clip_image_embeds(garment_pixel)
            
            warped_img = prepare_image_latents(warped_img)
            pose_img = prepare_image_latents(pose_image)
            
            _, ref_key, ref_value = transformer_garm(
                hidden_states=garm_model_latents,
                timestep=timesteps * 0,
                pooled_projections=cloth_image_embeds,
                encoder_hidden_states=None,
                return_dict=False)
            
        model_pred = transformer_vton( 
            hidden_states=latent_model_input,
            timestep=timesteps,
            pooled_projections=cloth_image_embeds,
            encoder_hidden_states=None,
            ref_key=ref_key,
            ref_value=ref_value,
            return_dict=False,
            aflow_cond=warped_img,
            pose_cond=pose_img)[0]
        
        if args.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        if args.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            prior_loss = torch.mean(
                (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                    target_prior.shape[0], -1
                ),
                1,
            )
            prior_loss = prior_loss.mean()
        gen_loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        gen_loss = gen_loss.mean()
        
        if args.with_prior_preservation:
            gen_loss = gen_loss + args.prior_loss_weight * prior_loss
        if args.freq_loss:
            with torch.no_grad():
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input
                model_pred = model_pred.to(image)
                pred_image = vae.decode(model_pred / vae.config.scaling_factor + vae.config.shift_factor).sample
                target = target.to(image)
                target_image = vae.decode(target / vae.config.scaling_factor + vae.config.shift_factor).sample
                # pred_image = torch.nn.functional.interpolate(
                #     pred_image, 
                #     size=(args.image_height // 8, args.image_width // 8), 
                #     mode='bilinear', 
                #     align_corners=False)
                # target_image = torch.nn.functional.interpolate(
                #     target_image, 
                #     size=(args.image_height // 8, args.image_width // 8), 
                #     mode='bilinear', 
                #     align_corners=False)
            freq_loss = focal_frequency_loss_fn(pred_image, target_image)
            total_loss = gen_loss + 10 * freq_loss.mean()
        else:
            freq_loss = torch.Tensor([0])
            total_loss = gen_loss #  + 0.5 * freq_loss.mean()
        return total_loss, gen_loss, freq_loss

    file_list = os.listdir('zalando-hd-resized/test/cloth')
    file_names = random.sample(file_list, 24)

    def visualize_and_save(
        phase: str = "validation",
        epoch: int = None,
        visualize: bool = False,
    ):
        """
        Save the pipeline and optionally run visualization.
        For validation phase, an epoch must be provided.
        """
        assert phase in ["validation", "final"], "Phase must be either 'validation' or 'final'."
        # Unwrap models from accelerator
        unwrap_transformer_vton = unwrap_model(transformer_vton)
        unwrap_transformer_garm = unwrap_model(transformer_garm)
        unwrap_image_encoder_large = unwrap_model(image_encoder_large)
        unwrap_image_encoder_bigG = unwrap_model(image_encoder_bigG)
        unwrap_aflow = unwrap_model(aflow)
        unwrap_vae = unwrap_model(vae)

        # Create the try-on pipeline
        pipeline = StableDiffusion3TryOnPipeline(
            transformer_vton=unwrap_transformer_vton,
            transformer_garm=unwrap_transformer_garm,
            image_encoder_large=unwrap_image_encoder_large,
            image_encoder_bigG=unwrap_image_encoder_bigG,
            aflow=unwrap_aflow,
            vae=unwrap_vae,
            scheduler=noise_scheduler
        )

        # Save pipeline based on phase
        if phase == "final":
            save_dir = args.output_dir
            pipeline.save_pretrained(save_dir)
            if visualize:
                pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
                    save_dir,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
        elif phase == "validation":
            if epoch is None:
                raise ValueError("For validation phase, an epoch number must be provided.")
            save_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
            pipeline.save_pretrained(save_dir)
            if visualize:
                pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
                    save_dir,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
        else:
            raise ValueError("Unsupported phase. Choose 'validation' or 'final'.")

        # Only run visualization if explicitly requested
        if visualize:
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            visualized_images = []

            for file_name in file_names:
                garm_img = Image.open(os.path.join('zalando-hd-resized/test/cloth', file_name))
                vton_img = Image.open(os.path.join('zalando-hd-resized/test/image', file_name))
                mask = Image.open(os.path.join('zalando-hd-resized/test/rect_mask', file_name))
                pose_img = Image.open(os.path.join('zalando-hd-resized/test/image-densepose', file_name))

                image_scale = 2
                n_steps = 20
                # seed = random.randint(0, 2147483647)
                num_images_per_prompt = 1

                # Use autocast only for final phase; otherwise use nullcontext.
                autocast_ctx = (torch.autocast(accelerator.device.type, dtype=weight_dtype)
                                if phase == "final" else nullcontext())

                with autocast_ctx:
                    res = pipeline(
                        height=1024,
                        width=768,
                        guidance_scale=image_scale,
                        num_inference_steps=n_steps,
                        # generator=torch.Generator("cpu").manual_seed(seed),
                        cloth_image=garm_img,
                        model_image=vton_img,
                        mask=mask,
                        pose_image=pose_img,
                        num_images_per_prompt=num_images_per_prompt
                    ).images
                    # Combine input and generated images
                    imgs = [garm_img, vton_img, pose_img] + res
                    visualized_images.extend(imgs)

            # Log images to available trackers
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    resized_images = [np.asarray(img.resize((768, 1024))) for img in visualized_images]
                    np_images = np.stack(resized_images)
                    tracker.writer.add_images(phase, np_images, epoch if epoch is not None else 0, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log({
                        phase: [
                            wandb.Image(image, caption=f"{i}") for i, image in enumerate(visualized_images)
                        ]
                    })
        del pipeline
        free_memory()
    
    for epoch in range(first_epoch, args.num_train_epochs):
        # train_sampler.set_epoch(epoch)
        transformer_vton.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer_vton]
            with accelerator.accumulate(models_to_accumulate):
                total_loss, gen_loss, freq_loss = compute_loss(batch)
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        unwrap_modules = {
                            "transformer_vton": unwrap_model(transformer_vton),
                            "optimizer": optimizer,
                            "lr_scheduler": lr_scheduler,
                        }
                        accelerator.save_state(
                            output_dir=save_path,
                            state_dict={k: v.state_dict() if hasattr(v, 'state_dict') else v for k, v in unwrap_modules.items()},
                            safe_serialization=True,
                            include_rng=False
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": total_loss.detach().item(), 
                    "gen_loss": gen_loss.detach().item(), 
                    "freq_loss": freq_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        # Validation loop: compute and log validation loss
        if accelerator.is_main_process and (epoch % args.validation_epochs == 0):
            # transformer_vton.eval()
            # for val_step, val_batch in enumerate(val_dataloader):
            #     with torch.no_grad():
            #         val_total_loss, val_gen_loss, val_freq_loss = compute_loss(val_batch)
                
            #     logs = {"Validation Loss": val_total_loss.detach().item(), 
            #             "Validation Freq Loss": val_freq_loss.detach().item(), 
            #             "Validation Diff Loss": val_gen_loss.detach().item()}
            #     progress_bar.set_postfix(**logs)
            #     accelerator.log(logs, step=global_step + val_step)
            
            # Optionally, run image generation validation as in original code.
            visualize_and_save(phase="validation", epoch=epoch, visualize=args.visualize_outputs)


    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        visualize_and_save(phase="final", epoch=epoch, visualize=args.visualize_outputs)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
