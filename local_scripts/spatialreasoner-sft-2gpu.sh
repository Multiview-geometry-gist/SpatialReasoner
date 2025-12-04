#!/bin/bash

# SpatialReasoner SFT Training with 2 GPUs (GPU 0,1)
# This script trains the Qwen2.5-VL-7B model using supervised fine-tuning

model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/OpenImages_3DSR_apr23_sampled24k_NT_llava_24k

# Use only GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

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
echo "Output: checkpoints/${WANDB_RUN_NAME}"
echo "========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_2gpu.yaml \
    --num_processes=2 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo.yaml \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --stop_steps 100

echo ""
echo "========================================="
echo "Training completed!"
echo "Checkpoint saved to: checkpoints/${WANDB_RUN_NAME}"
echo "========================================="

# Download preprocessor_config.json and chat_template.json
# https://github.com/huggingface/transformers/issues/29790#issuecomment-2016755078
echo "Downloading preprocessor config files..."
wget -P checkpoints/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json
wget -P checkpoints/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json

echo "Done!"
