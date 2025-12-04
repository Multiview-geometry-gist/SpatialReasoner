#!/bin/bash

# SpatialReasoner SFT Training with 4 GPUs from checkpoint-2000 weights
# Uses checkpoint-2000 as model_name_or_path (no optimizer state resume)
# This avoids world size mismatch issue

# Use checkpoint-2000 as the starting model (contains trained weights)
model_path=/data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-Instruct-SFT-1GPU-2025-11-06-06-45-28/checkpoint-2000
dataset_name=ccvl/OpenImages_3DSR_apr23_sampled24k_NT_llava_24k

# Use GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data and checkpoint paths
DATA_DIR=/data/SpatialReasoner/data
CHECKPOINT_DIR=/data/SpatialReasoner/checkpoints

# Set run name
export WANDB_RUN_NAME=Qwen2.5-VL-7B-Instruct-SFT-4GPU-FromCkpt2000-$(date +%Y-%m-%d-%H-%M-%S)

echo "========================================="
echo "Starting 4 GPU Training from checkpoint-2000"
echo "Model Path: $model_path"
echo "Dataset: $dataset_name"
echo "GPUs: 0, 1, 2, 3"
echo "DeepSpeed: ZeRO-2"
echo "Note: Using checkpoint as model_name_or_path (no optimizer resume)"
echo "Output Dir: $CHECKPOINT_DIR/${WANDB_RUN_NAME}"
echo "========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_4gpu.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo.yaml \
    --output_dir ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} \
    --model_name_or_path $model_path \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --data_dir ${DATA_DIR}/openimages/ \
    --llava_dir ${DATA_DIR}/llava/

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
