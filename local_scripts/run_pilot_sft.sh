#!/bin/bash
# Pilot SFT Training on GPU 0-3

set -e
cd /home/ubuntu/SpatialReasoner

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=spatial-reasoner-pilot
export WANDB_RUN_NAME=pilot-sft-$(date +%Y%m%d-%H%M%S)
export WANDB_MODE=offline

echo "========================================"
echo "Pilot SFT Training"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: recipes/Qwen2.5-VL-7B-Instruct/sft/config_pilot.yaml"
echo "Max steps: 100"
echo "========================================"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_4gpu.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_pilot.yaml

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
