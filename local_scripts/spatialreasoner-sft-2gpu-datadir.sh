#!/bin/bash

# SpatialReasoner SFT Training with 2 GPUs (GPU 0,1)
# All data and checkpoints stored in /data/SpatialReasoner/

model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/OpenImages_3DSR_apr23_sampled24k_NT_llava_24k

# Use only GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Increase NCCL timeout for torch compile (60 minutes)
export NCCL_TIMEOUT=3600

# Data and checkpoint paths in /data
DATA_DIR=/data/SpatialReasoner/data
CHECKPOINT_DIR=/data/SpatialReasoner/checkpoints

# WANDB Configuration (Optional - comment out if not using)
# export WANDB_BASE_URL=https://api.wandb.ai
# export WANDB_PROJECT=spatial-reasoning-2gpu
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
# export WANDB_RUN_NAME=$(basename $model_name)-SFT-2GPU-$(date +%Y-%m-%d-%H-%M-%S)
# wandb login $WANDB_API_KEY

# If not using WANDB, set a simple run name
export WANDB_RUN_NAME=$(basename $model_name)-SFT-2GPU-$(date +%Y-%m-%d-%H-%M-%S)

# HuggingFace Token (Optional - only needed for gated models)
# export HF_TOKEN=YOUR_HF_TOKEN

echo "========================================="
echo "Starting SFT Training with 2 GPUs"
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "GPUs: 0, 1"
echo "Data Dir: $DATA_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR/${WANDB_RUN_NAME}"
echo "========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_2gpu.yaml \
    --num_processes=2 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo_2gpu_lowmem.yaml \
    --output_dir ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --data_dir ${DATA_DIR}/openimages/ \
    --llava_dir ${DATA_DIR}/llava/ \
    --stop_steps 100

echo ""
echo "========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${CHECKPOINT_DIR}/${WANDB_RUN_NAME}"
echo "========================================="

# Download preprocessor_config.json and chat_template.json
# https://github.com/huggingface/transformers/issues/29790#issuecomment-2016755078
echo "Downloading preprocessor config files..."
wget -P ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json
wget -P ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json

echo "Done!"
