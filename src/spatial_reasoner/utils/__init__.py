"""Utility modules for SpatialReasoner.

Includes:
- quaternion: Quaternion operations for rotation handling
- geometry: 3D geometry utilities
- callbacks: Training callbacks
- evaluation: Evaluation utilities
- hub: Hub upload utilities
- utils: Conversation formatting
- wandb_logging: Weights & Biases integration
"""

from . import quaternion
from . import geometry

__all__ = [
    "quaternion",
    "geometry",
]
