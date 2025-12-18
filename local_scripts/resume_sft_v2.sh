#!/bin/bash
# Resume SpatialReasoner SFT Training from checkpoint-1500
# With increased NCCL timeout to prevent communication failures

set -e
cd /home/ubuntu/SpatialReasoner

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL settings to prevent timeout - INCREASED FOR VISION MODEL I/O
export NCCL_TIMEOUT=7200          # 2 hours (critical for variable image loading)
export NCCL_DEBUG=INFO            # Enable debug logging
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=23         # InfiniBand timeout
export TORCH_NCCL_BLOCKING_WAIT=0

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create log directory
LOG_DIR=/data/SpatialReasoner/logs
mkdir -p $LOG_DIR

# WandB configuration (offline mode)
export WANDB_PROJECT=spatial-reasoner-resume
export WANDB_RUN_NAME=SFT-resume-$(date +%Y%m%d-%H%M%S)
export WANDB_MODE=offline
export WANDB_DIR=$LOG_DIR

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/sft_resume_$TIMESTAMP.log

echo "========================================"
echo "Resuming SpatialReasoner SFT Training"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Resuming from: checkpoint-1500"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds (2 hours)"
echo "Dataloader workers: 4 (with prefetch)"
echo "Pixel range: 100,352 - 150,528 (narrowed)"
echo "Output: /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-v2"
echo "Log: $LOG_FILE"
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
echo "Checkpoint: /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-v2"
echo "Log: $LOG_FILE"
echo "========================================"
