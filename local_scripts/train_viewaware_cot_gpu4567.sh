#!/bin/bash
# View-Aware CoT Training on GPU 4-7
# ===================================
# This script trains a model with view-aware CoT that includes
# explicit viewpoint context for better multi-view understanding.
#
# Usage:
#   bash local_scripts/train_viewaware_cot_gpu4567.sh

set -e

cd /home/ubuntu/SpatialReasoner

echo "Starting View-Aware CoT Training on GPU 4-7..."
echo "================================================"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Optional: Set WandB settings
# export WANDB_PROJECT=spatial-reasoner-viewaware-cot

# Run training
accelerate launch \
    --config_file recipes/accelerate_configs/zero2_4gpu_4567.yaml \
    src/spatial_reasoner/sft_viewaware_cot.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_viewaware_cot.yaml

echo "================================================"
echo "Training completed!"
echo "Checkpoint saved to: /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-ViewAware-CoT"
