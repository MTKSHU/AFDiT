import os
import math
import random
import shutil
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from PIL import Image

from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import free_memory

from src.appearance_flow import AppearanceFlowEncDec
from src.aflow_pipeline import AflowEncDecPipeline
from utils import VGGLoss, focal_frequency_loss_fn, ssim_loss_fn, dice_loss_fn
from vitonhd_dataset import VitonHDDataset

if is_wandb_available():
    import wandb

check_min_version("0.31.0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./output/aflow",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None, required=False, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")

    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--max_train_steps", type=int, default=None, required=False)
    
    parser.add_argument("--data_root", type=str, default="./zalando-hd-resized",)
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
    
    parser.add_argument("--pose_channels", type=int, default=3, help="Number of input channels for the AppearanceFlow.")
    parser.add_argument("--garm_channels", type=int, default=4, help="Number of input channels for the AppearanceFlow.")
    parser.add_argument("--feature_channels", type=int, default=64, help="Number of feature channels for the AppearanceFlow.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of blocks in the AppearanceFlow.")
    parser.add_argument("--act_fn", type=str, default="relu", help="Activation function for the AppearanceFlow.")
    parser.add_argument("--attn_layers", type=eval, default="[False, False, True, True]", help="Attention layers configuration for the AppearanceFlow.")
    parser.add_argument("--mid_block_attn", type=bool, default=True, help="Whether to use attention in the middle block of the AppearanceFlow.")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output channels for the AppearanceFlow.")
    parser.add_argument("--out_kernel_size", type=int, default=7, help="Kernel size for the output layer of the AppearanceFlow.")
    parser.add_argument("--out_stride", type=int, default=1, help="Stride for the output layer of the AppearanceFlow.")
    parser.add_argument("--out_padding", type=int, default=3, help="Padding for the output layer of the AppearanceFlow.")

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

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name,
        #         exist_ok=True,
        #     ).repo_id


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
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    aflow = AppearanceFlowEncDec(
        pose_channels=args.pose_channels, 
        garm_channels=args.garm_channels,
        feature_channels=args.feature_channels,
        num_blocks=args.num_blocks,
        act_fn=args.act_fn,
        attn_layers=args.attn_layers,
        mid_block_attn=args.mid_block_attn,
        out_channels=args.out_channels,
        out_kernel_size=args.out_kernel_size,
        out_stride=args.out_stride,
        out_padding=args.out_padding
    )
    aflow.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    aflow.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        aflow.enable_gradient_checkpointing()
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * \
            args.train_batch_size * accelerator.num_processes
        )
    aflow_parameters_with_lr = {
        "params": aflow.parameters(),
        "lr": args.learning_rate,
    }
    params_to_optimize = [aflow_parameters_with_lr]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = VitonHDDataset(
        dataroot_path=args.data_root,
        phase="train",
        order="paired",
        size=(args.image_width, args.image_height)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    aflow, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        aflow, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "aflow"
        accelerator.init_trackers(tracker_name)  # , config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
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
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    loss_vgg = VGGLoss(device=accelerator.device, dtype=weight_dtype)
    unwrap_aflow = accelerator.unwrap_model(aflow)
    
    free_memory()
    
    file_list = os.listdir('zalando-hd-resized/test/cloth')
    file_names = random.sample(file_list, 24)
        
    for epoch in range(first_epoch, args.num_train_epochs):
        aflow.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [aflow]
            with accelerator.accumulate(models_to_accumulate):
                garment = batch["cloth"].to(dtype=weight_dtype)
                garment_edge = batch["cloth-edge"].to(dtype=weight_dtype)
                pose_img = batch["image-densepose"].to(dtype=weight_dtype)
                image = batch["image-upper"].to(dtype=weight_dtype)
                image_mask = batch["image-upper-mask"].to(dtype=weight_dtype)
                
                flow, warped_img, mask = aflow(pose_img, garment, garment_edge)
                warped_img = warped_img.to(dtype=weight_dtype)
                
                l1 = F.l1_loss(warped_img, image)
                vgg = loss_vgg(warped_img, image)
                freq = focal_frequency_loss_fn(warped_img, image)
                ssim = ssim_loss_fn(warped_img, image)
                bce = F.binary_cross_entropy_with_logits(mask, image_mask)
                dice = dice_loss_fn(mask, image_mask)

                loss = l1 + 0.5 * vgg + 0.03 * freq + ssim + 0.5 * bce + 0.5 * dice
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        aflow.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), 
                    "l1": l1.detach().item(), "vgg": vgg.detach().item(), "freq": freq.detach().item(), "ssim": ssim.detach().item(),
                    "bce": bce.detach().item(), "dice": dice.detach().item(),
                     "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                
                unwraped_aflow = unwrap_model(aflow)

                pipeline = AflowEncDecPipeline(aflow=unwraped_aflow)
                images = []
                
                for file_name in file_names:
                    cloth_image = Image.open(f'zalando-hd-resized/test/cloth/{file_name}')
                    # cloth_mask = Image.open(f'zalando-hd-resized/test/cloth-mask/{file_name}')
                    pose_image = Image.open(f'zalando-hd-resized/test/image-densepose/{file_name}')
                    gt_image = Image.open(f'zalando-hd-resized/test/image/{file_name}')
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    # run inference
                    autocast_ctx = nullcontext()

                    with autocast_ctx:
                        image = [ # pose_image.resize((args.image_width, args.image_height)), 
                                 cloth_image.resize((args.image_width, args.image_height))]
                        res = pipeline(
                            height=args.image_height, 
                            width=args.image_width, 
                            cloth_image=cloth_image, 
                            pose_image=pose_image)
                        image += res["images"]
                        image += res["mask"]
                        image.append(gt_image.resize((args.image_width, args.image_height)))
                        images += image
                for tracker in accelerator.trackers:
                    phase_name = "validation"
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img.convert("RGB")) for img in images])
                        tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
                        
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                phase_name: [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                                ]
                            }
                        )
                pipeline.save_pretrained(args.output_dir + "/epoch-" + str(epoch))
                del pipeline
                free_memory()

    # Save the lora layers
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwraped_aflow = unwrap_model(aflow)
        pipeline = AflowEncDecPipeline(aflow=unwraped_aflow)

        # save the pipeline
        pipeline.save_pretrained(args.output_dir)
        # Final inference
        # Load previous pipeline
        pipeline = AflowEncDecPipeline.from_pretrained(
            args.output_dir,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # # run inference
        images = []
        
        for file_name in file_names:
            cloth_image = Image.open(f'zalando-hd-resized/test/cloth/{file_name}')
            # cloth_mask = Image.open(f'zalando-hd-resized/test/cloth-mask/{file_name}')
            pose_image = Image.open(f'zalando-hd-resized/test/image-densepose/{file_name}')
            gt_image = Image.open(f'zalando-hd-resized/test/image/{file_name}')

            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
            
            with autocast_ctx:
                image = [ # pose_image.resize((args.image_width, args.image_height)), 
                         cloth_image.resize((args.image_width, args.image_height))]
                res = pipeline(
                    height=args.image_height, 
                    width=args.image_width, 
                    cloth_image=cloth_image, 
                    pose_image=pose_image)
                image += res["images"]
                image += res["mask"]
                image.append(gt_image.resize((args.image_width, args.image_height)))
                images += image
        for tracker in accelerator.trackers:
            phase_name = "final"
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img.convert("RGB")) for img in images])

                tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                        ]
                    }
                )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
