"""
Multi-view spatial reasoning data generation pipeline.

Implements depth-based view synthesis for training data augmentation
and quaternion-based pose estimation for rotation-aware reasoning.
"""

from .config import DataGenerationConfig
from .pipeline import DataGenerationPipeline

__all__ = ["DataGenerationConfig", "DataGenerationPipeline"]
