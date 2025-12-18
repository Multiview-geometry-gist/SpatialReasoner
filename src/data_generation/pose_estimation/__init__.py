"""Pose Estimation Module with Pluggable Backends.

This module provides a modular architecture for 6-DOF pose estimation with:
- Abstract base class defining the estimation interface
- Multiple backend implementations (depth-based, Orient-Anything, hybrid)
- Configuration-driven backend selection
- Graceful fallback mechanisms

Architecture:
    PoseEstimatorBase (ABC)
        |
        +-- DepthBasedEstimator (PCA-based, uses depth map)
        +-- OrientAnythingEstimator (neural network-based)
        +-- HybridPoseEstimator (combines both approaches)

Usage:
    from data_generation.pose_estimation import create_pose_estimator

    # Simple factory usage (recommended)
    config = PoseEstimationConfig(backend="depth_based")
    estimator = create_pose_estimator(config)
    poses = estimator.estimate_poses(image, depth_map, masks, boxes, K)

    # Direct backend usage
    from data_generation.pose_estimation import DepthBasedEstimator
    estimator = DepthBasedEstimator(config)

Extending:
    To add a new backend:
    1. Create new_backend.py implementing PoseEstimatorBase
    2. Decorate with @PoseEstimatorRegistry.register("new_backend")
    3. Import in this __init__.py (or lazy-load via __getattr__)

    No changes to core pipeline code required.

Backward Compatibility:
    The original PoseEstimator class is aliased to DepthBasedEstimator.
    Existing code using PoseEstimator will continue to work unchanged.
"""

import logging

logger = logging.getLogger(__name__)

# Core imports - always available
from .base import (
    PoseEstimatorBase,
    ObjectPose,
    PoseEstimationConfig,
    euler_to_quaternion,
    quaternion_to_euler,
)
from .registry import (
    PoseEstimatorRegistry,
    create_pose_estimator,
    ensure_backends_loaded,
)
from .depth_based import DepthBasedEstimator

# Backward compatibility alias for existing code
PoseEstimator = DepthBasedEstimator

__all__ = [
    # Core classes
    "PoseEstimatorBase",
    "ObjectPose",
    "PoseEstimationConfig",
    # Utility functions
    "euler_to_quaternion",
    "quaternion_to_euler",
    # Registry and factory
    "PoseEstimatorRegistry",
    "create_pose_estimator",
    "ensure_backends_loaded",
    # Backend implementations
    "DepthBasedEstimator",
    # Backward compatibility alias
    "PoseEstimator",
]


def __getattr__(name: str):
    """Lazy loading for optional backend implementations.

    This allows importing backends only when they are actually used,
    avoiding ImportError for missing optional dependencies.
    """
    if name == "OrientAnythingEstimator":
        try:
            from .orient_anything import OrientAnythingEstimator
            return OrientAnythingEstimator
        except ImportError as e:
            raise ImportError(
                f"OrientAnythingEstimator requires additional dependencies. "
                f"Install with: pip install rembg transformers huggingface_hub. "
                f"Also ensure Orient-Anything repo is cloned to /home/ubuntu/Orient-Anything. "
                f"Error: {e}"
            )
    elif name == "HybridPoseEstimator":
        from .hybrid import HybridPoseEstimator
        return HybridPoseEstimator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def list_available_backends() -> list:
    """List all registered and available backends.

    Returns:
        List of backend names that can be used
    """
    ensure_backends_loaded()
    return PoseEstimatorRegistry.list_backends()


def get_backend_info(name: str) -> dict:
    """Get information about a specific backend.

    Args:
        name: Backend name

    Returns:
        Dictionary with backend metadata and capabilities
    """
    ensure_backends_loaded()
    metadata = PoseEstimatorRegistry.get_metadata(name)
    backend_cls = PoseEstimatorRegistry.get(name)

    if backend_cls is None:
        return {"error": f"Backend '{name}' not found"}

    # Get capabilities if possible
    try:
        from .base import PoseEstimationConfig
        temp_config = PoseEstimationConfig()
        temp_instance = backend_cls(temp_config)
        capabilities = temp_instance.get_capabilities()
    except Exception:
        capabilities = {}

    return {
        "name": name,
        "class": backend_cls.__name__,
        "capabilities": capabilities,
        **metadata,
    }


# Auto-load backends on import
try:
    ensure_backends_loaded()
except Exception as e:
    logger.debug(f"Some backends could not be loaded: {e}")
