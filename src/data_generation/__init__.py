"""Data generation pipeline for multi-view spatial reasoning.

This module provides tools for:
- Depth estimation using Depth Anything v2
- Instance segmentation using SAM2
- Novel view synthesis via depth-based warping
- 6-DOF pose estimation with quaternion representation
- Spatial reasoning QA pair generation

Usage:
    from data_generation import DataGenerationPipeline

    pipeline = DataGenerationPipeline(config)
    pipeline.process_dataset(input_dir, output_dir)
"""

from .config import DataGenerationConfig
from .pipeline import DataGenerationPipeline

__all__ = [
    "DataGenerationConfig",
    "DataGenerationPipeline",
]
