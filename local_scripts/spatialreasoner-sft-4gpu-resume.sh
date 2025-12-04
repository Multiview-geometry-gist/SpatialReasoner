#!/bin/bash

# SpatialReasoner SFT Training Resume with 4 GPUs (GPU 0,1,2,3)
# Resume from checkpoint-2000

model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/OpenImages_3DSR_apr23_sampled24k_NT_llava_24k

# Use GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data and checkpoint paths in /data
DATA_DIR=/data/SpatialReasoner/data
CHECKPOINT_DIR=/data/SpatialReasoner/checkpoints
RESUME_FROM=/data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-Instruct-SFT-1GPU-2025-11-06-06-45-28/checkpoint-2000

# Set a new run name for continuation
export WANDB_RUN_NAME=$(basename $model_name)-SFT-4GPU-Resume-$(date +%Y-%m-%d-%H-%M-%S)

echo "========================================="
echo "Resuming SFT Training with 4 GPUs"
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "GPUs: 0, 1, 2, 3"
echo "Resume From: $RESUME_FROM"
echo "Data Dir: $DATA_DIR"
echo "New Output Dir: $CHECKPOINT_DIR/${WANDB_RUN_NAME}"
echo "========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo.yaml \
    --output_dir ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --data_dir ${DATA_DIR}/openimages/ \
    --llava_dir ${DATA_DIR}/llava/ \
    --resume_from_checkpoint $RESUME_FROM

echo ""
echo "========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${CHECKPOINT_DIR}/${WANDB_RUN_NAME}"
echo "========================================="

# Download preprocessor_config.json and chat_template.json
echo "Downloading preprocessor config files..."
wget -P ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json
wget -P ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json

echo "Done!"
