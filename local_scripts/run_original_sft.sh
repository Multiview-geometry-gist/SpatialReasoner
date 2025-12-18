#!/bin/bash
# Original SpatialReasoner SFT Training
# Uses ccvl/SpatialReasonerTrain-SFT dataset (48,000 samples)
# All outputs saved to /data/SpatialReasoner/

set -e
cd /home/ubuntu/SpatialReasoner

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create log directory
LOG_DIR=/data/SpatialReasoner/logs
mkdir -p $LOG_DIR

# WandB configuration (offline mode)
export WANDB_PROJECT=spatial-reasoner-original
export WANDB_RUN_NAME=SFT-original-$(date +%Y%m%d-%H%M%S)
export WANDB_MODE=offline
export WANDB_DIR=$LOG_DIR

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/sft_original_$TIMESTAMP.log

echo "========================================"
echo "Original SpatialReasoner SFT Training"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: recipes/Qwen2.5-VL-7B-Instruct/sft/config_original.yaml"
echo "Dataset: ccvl/SpatialReasonerTrain-SFT (48,000 samples)"
echo "Output: /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-original"
echo "Log: $LOG_FILE"
echo "Early stop: 7000 steps"
echo "========================================"

# Launch training with DeepSpeed ZeRO-2
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_4gpu.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_original.yaml \
    2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "Training Complete!"
echo "Checkpoint: /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-original"
echo "Log: $LOG_FILE"
echo "========================================"
