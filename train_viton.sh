#!/bin/bash
set -e  # Quit on error.

# activate Conda Env.
source /opt/conda/etc/profile.d/conda.sh
conda activate afdit

# Training Parameters.
IMAGE_HEIGHT=1024
IMAGE_WIDTH=768
TRAIN_BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=1e-5
CHECKPOINT_STEPS=2000
NUM_EPOCHS=50
VALIDATION_EPOCHS=10
OPTIMIZER=adamw
MIXED_PRECISION=bf16
LR_SCHEDULER=constant_with_warmup
GRADIENT_ACCUMULATION_STEPS=1
SCALE_LR=False

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=./output/vton/${TIMESTAMP}_${MIXED_PRECISION}_${OPTIMIZER}_${LR_SCHEDULER}-${LEARNING_RATE}_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_rectangle-enhance_full-train

echo "Output directory: ${OUTPUT_DIR}"

# Pretrained model and AFLOW repository paths.
REPO=pretrained_models/BoyuanJiang/FitDiT
AFLOW_REPO=output/aflow/aflow_enc_dec_1e-4_256x192_with-mask

# CUDA devices to use.
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Train the model using accelerate.
accelerate launch --main_process_port 29500 --mixed_precision ${MIXED_PRECISION} train_viton.py \
    --image_height ${IMAGE_HEIGHT} \
    --image_width ${IMAGE_WIDTH} \
    --repo ${REPO} \
    --aflow_repo ${AFLOW_REPO} \
    --mixed_precision ${MIXED_PRECISION} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler ${LR_SCHEDULER} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --scale_lr ${SCALE_LR} \
    --gradient_checkpointing \
    --output_dir ${OUTPUT_DIR} \
    --checkpointing_steps ${CHECKPOINT_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --validation_epochs ${VALIDATION_EPOCHS} \
    --resume_from_checkpoint latest \
    --allow_tf32
    # --freq_loss \
    # --visualize_outputs \
    # --optimizer prodigy \
