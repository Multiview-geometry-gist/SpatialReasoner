#!/bin/bash
# SFT Training with Multi-View Data
#
# This script runs SFT training with:
# - Multi-view data augmentation (different camera angles)
# - Rotation-aware queries emphasized
# - Standard spatial reasoning tasks
#
# Usage:
#   bash local_scripts/spatialreasoner-sft-multiview.sh

set -e

# Model and dataset
model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/OpenImages_3DSR  # Or local path

# Wandb setup
export WANDB_PROJECT=spatial-reasoning-multiview
export WANDB_RUN_NAME=$(basename $model_name)-SFT-multiview-$(date +%Y-%m-%d-%H-%M-%S)

# Optional: Set your API keys
# export WANDB_API_KEY="your_key_here"
# export HF_TOKEN="your_token_here"

echo "======================================"
echo "Multi-View SFT Training"
echo "======================================"
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "Run name: $WANDB_RUN_NAME"
echo "======================================"

# Check if config exists
CONFIG_FILE="recipes/Qwen2.5-VL-7B-Instruct/sft/config_multiview.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config "$CONFIG_FILE" \
    --output_dir "checkpoints/${WANDB_RUN_NAME}" \
    --model_name_or_path "$model_name" \
    --dataset_name "$dataset_name" \
    --run_name "$WANDB_RUN_NAME"

# Download preprocessor configs for inference
echo "Downloading preprocessor configs..."
OUTPUT_DIR="checkpoints/${WANDB_RUN_NAME}"
wget -q -P "$OUTPUT_DIR" \
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json" || true
wget -q -P "$OUTPUT_DIR" \
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json" || true

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo "Checkpoint saved to: $OUTPUT_DIR"
