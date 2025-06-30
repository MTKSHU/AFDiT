#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate afdit

CUDA_VISIBLE_DEVICES=0,1 \
    accelerate launch --main_process_port=29500 train_aflow_enc_dec.py \
        --image_height 256 \
        --image_width 192 \
        --train_batch_size 32 \
        --dataloader_num_workers 32 \
        --lr_scheduler constant_with_warmup \
        --learning_rate 1e-4 \
        --act_fn silu \
        --output_dir ./output/aflow/aflow_enc_dec_1e-4_256x192_with-mask \
        --checkpointing_steps 2000 \
        --num_train_epochs 50 \
        --validation_epochs 10 \
        --resume_from_checkpoint latest
