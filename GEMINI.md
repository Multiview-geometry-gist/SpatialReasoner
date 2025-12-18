# SpatialReasoner Project Context

## Project Overview
**SpatialReasoner** is an official implementation of the paper "SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning". It aims to improve 3D spatial reasoning in Multimodal Large Language Models (MLLMs) by building explicit 3D representations, performing 3D computations, and reasoning about the final answer.

## Tech Stack
*   **Language:** Python 3.11+
*   **Frameworks:** PyTorch, Hugging Face Transformers, Accelerate, DeepSpeed, TRL (Transformer Reinforcement Learning), vLLM.
*   **Base Models:** Likely builds upon Qwen2.5-VL (observed in directory names).
*   **Tools:** WandB (logging), Flash Attention (optimization).

## Project Structure

### Key Directories
*   `src/spatial_reasoner/`: Core source code for the python package.
*   `local_scripts/`: Bash scripts for launching training (SFT, Zero) and inference jobs.
*   `configs/`: YAML configuration files for data generation, training, and view synthesis.
*   `recipes/`: Configuration recipes for `accelerate` and model-specific settings (e.g., Qwen2.5-VL).
*   `data/`: Storage for training and evaluation datasets (OpenImages, LLaVA, 3DSRBench).
*   `VLMEvalKit/`: A git submodule used for setting up the evaluation environment.
*   `checkpoints/`: Directory where trained model checkpoints are saved.

## Setup & Installation

### 1. Environment Setup
```bash
conda create -n spatial_reasoner python=3.11 -y
conda activate spatial_reasoner
```

### 2. Dependencies
```bash
pip3 install -e ".[dev]"
pip3 install flash-attn --no-build-isolation
pip3 install qwen_vl_utils xlsxwriter
```

### 3. Evaluation Setup
```bash
git submodule update --init --recursive
cd VLMEvalKit
pip install -e .
cd ..
```

## Usage

### Training
Training workflows are managed via scripts in `local_scripts/`.
*   **SFT (Supervised Fine-Tuning):** `bash local_scripts/spatialreasoner-sft.sh`
*   **Zero-Shot / RL:** `bash local_scripts/spatialreasoner-zero.sh`
*   **Full Pipeline:** `bash local_scripts/spatialreasoner.sh`

**Note:** Ensure data is downloaded to `data/` before training (see `README.md` for download links).

### Inference & Evaluation
Inference scripts are also located in `local_scripts/`.
*   **SFT Inference:** `bash local_scripts/infer_spatialreasoner-sft.sh`
*   **Zero Inference:** `bash local_scripts/infer_spatialreasoner-zero.sh`

Results are typically saved to `results_3DSRBench.csv` or printed to stdout.

## Development
*   **Style Checks:** Use `make style` to format code (black, isort) and `make quality` to check code quality (flake8).
*   **Configuration:** Adjust training parameters in `configs/` or `recipes/` before running scripts.
