#!/usr/bin/env bash
set -euo pipefail

# Multiview inference using precomputed views (moved to /data to save workspace disk).

MODEL_PATH="${MODEL_PATH:-/data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-MVGen-Multi/checkpoint-500}"
TEST_DATA="${TEST_DATA:-/data/SpatialReasoner/data/benchmark/3dsrbench_v1_vlmevalkit_circular.tsv}"
VIEWS_DIR="${VIEWS_DIR:-/data/SpatialReasoner/results/multiview_eval_600/generated_views}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/ubuntu/SpatialReasoner/results/multiview_precomputed_mvgen_multi_ckpt500.xlsx}"
GPU_IDS="${GPU_IDS:-0}"
ANGLES="${ANGLES:-10,-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"

python -m src.inference.mvgen_inference_precomputed \
  --model_path "${MODEL_PATH}" \
  --test_data "${TEST_DATA}" \
  --views_dir "${VIEWS_DIR}" \
  --output_path "${OUTPUT_PATH}" \
  --gpu_ids "${GPU_IDS}" \
  --angles "${ANGLES}" \
  --batch_size "${BATCH_SIZE}"

