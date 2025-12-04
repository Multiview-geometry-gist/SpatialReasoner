#!/bin/bash

# SpatialReasoner SFT Training with 1 GPU (GPU 0)
# All data and checkpoints stored in /data/SpatialReasoner/

model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/OpenImages_3DSR_apr23_sampled24k_NT_llava_24k

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Data and checkpoint paths in /data
DATA_DIR=/data/SpatialReasoner/data
CHECKPOINT_DIR=/data/SpatialReasoner/checkpoints

# If not using WANDB, set a simple run name
export WANDB_RUN_NAME=$(basename $model_name)-SFT-1GPU-$(date +%Y-%m-%d-%H-%M-%S)

echo "========================================="
echo "Starting SFT Training with 1 GPU"
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "GPU: 0"
echo "Data Dir: $DATA_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR/${WANDB_RUN_NAME}"
echo "========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_1gpu_cpu_offload.yaml \
    --num_processes=1 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo_2gpu_lowmem.yaml \
    --output_dir ${CHECKPOINT_DIR}/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
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
