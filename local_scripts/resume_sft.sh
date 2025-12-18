#!/bin/bash
# Resume SFT Training from checkpoint-1500
# With improved NCCL settings to prevent timeout

set -e
cd /home/ubuntu/SpatialReasoner

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL settings to prevent timeout
export NCCL_TIMEOUT=1800  # 30 minutes (default 10 min)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN

# Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix PyTorch 2.6 weights_only issue for DeepSpeed checkpoint loading
export TORCH_LOAD_WEIGHTS_ONLY=0

# WandB configuration (offline mode)
export WANDB_PROJECT=spatial-reasoner-original
export WANDB_RUN_NAME=SFT-resume-$(date +%Y%m%d-%H%M%S)
export WANDB_MODE=offline
export WANDB_DIR=/data/SpatialReasoner/logs

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=/data/SpatialReasoner/logs/sft_resume_$TIMESTAMP.log

echo "========================================"
echo "Resuming SFT Training from checkpoint-1500"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds"
echo "Resume from: checkpoint-1500"
echo "Log: $LOG_FILE"
echo "========================================"

# Launch training with resume
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_4gpu.yaml \
    --num_processes=4 \
    src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_original.yaml \
    --resume_from_checkpoint /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-original/checkpoint-1500 \
    2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
