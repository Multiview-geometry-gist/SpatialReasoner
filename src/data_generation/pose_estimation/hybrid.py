"""Hybrid pose estimation backend.

Combines multiple backends for robust pose estimation:
- Uses Orient-Anything when available and confident
- Falls back to depth-based when Orient-Anything fails or has low confidence

This provides the best of both worlds:
- Neural network accuracy for orientation when it works well
- Reliable fallback for edge cases
"""

import numpy as np
from typing import Optional, List, Dict, Any
import logging

from .base import PoseEstimatorBase, ObjectPose, PoseEstimationConfig
from .registry import PoseEstimatorRegistry

logger = logging.getLogger(__name__)


@PoseEstimatorRegistry.register("hybrid", priority=25)
class HybridPoseEstimator(PoseEstimatorBase):
    """Hybrid pose estimation combining multiple backends.

    Strategy:
    1. Try Orient-Anything first (if available)
    2. If confidence < threshold, use depth-based PCA
    3. If Orient-Anything fails entirely, fall back to depth-based

    Attributes:
        backend_name: "hybrid"

    Configuration:
        - confidence_threshold: Minimum Orient-Anything confidence (default 0.5)
        - primary_backend: First backend to try ("orient_anything")
        - fallback_backend: Fallback backend ("depth_based")
    """

    backend_name = "hybrid"

    def __init__(self, config: PoseEstimationConfig):
        super().__init__(config)

        self._confidence_threshold = getattr(config, 'confidence_threshold', 0.5)
        self._primary_backend_name = getattr(config, 'primary_backend', 'orient_anything')
        self._fallback_backend_name = getattr(config, 'fallback_backend', 'depth_based')

        # Lazy-loaded backends
        self._primary_backend = None
        self._fallback_backend = None
        self._primary_available = None

    def _initialize(self) -> None:
        """Initialize both backends."""
        from .depth_based import DepthBasedEstimator

        # Always initialize fallback (depth-based)
        self._fallback_backend = DepthBasedEstimator(self.config)
        self._fallback_backend._ensure_initialized()

        # Try to initialize primary (orient-anything)
        try:
            from .orient_anything import OrientAnythingEstimator, check_orient_anything_available

            if check_orient_anything_available():
                self._primary_backend = OrientAnythingEstimator(self.config)
                self._primary_backend._ensure_initialized()
                self._primary_available = True
                logger.info("Hybrid: Orient-Anything available as primary")
            else:
                self._primary_available = False
                logger.info("Hybrid: Orient-Anything not available, using depth-based only")

        except Exception as e:
            logger.warning(f"Hybrid: Could not initialize Orient-Anything: {e}")
            self._primary_available = False

    def estimate_single(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray,
        K: np.ndarray,
        label: Optional[str] = None,
        object_id: int = 0,
    ) -> Optional[ObjectPose]:
        """Estimate pose using hybrid approach.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            mask: Segmentation mask (H, W)
            box: 2D bounding box [x1, y1, x2, y2]
            K: Camera intrinsics (3, 3)
            label: Optional object label
            object_id: Unique identifier

        Returns:
            ObjectPose from best available backend
        """
        self._ensure_initialized()

        pose = None
        used_backend = "none"

        # Try primary backend (Orient-Anything)
        if self._primary_available and self._primary_backend is not None:
            try:
                pose = self._primary_backend.estimate_single(
                    image, depth_map, mask, box, K, label, object_id
                )

                if pose is not None:
                    # Check confidence threshold
                    if pose.confidence >= self._confidence_threshold:
                        used_backend = "orient_anything"
                    else:
                        logger.debug(
                            f"Object {object_id}: Orient-Anything confidence "
                            f"{pose.confidence:.2f} < {self._confidence_threshold}, "
                            "falling back to depth-based"
                        )
                        pose = None  # Will trigger fallback

            except Exception as e:
                logger.warning(f"Orient-Anything failed for object {object_id}: {e}")
                pose = None

        # Fallback to depth-based
        if pose is None:
            pose = self._fallback_backend.estimate_single(
                image, depth_map, mask, box, K, label, object_id
            )
            used_backend = "depth_based"

        # Update metadata with backend info
        if pose is not None:
            pose.metadata["hybrid_backend_used"] = used_backend
            pose.metadata["primary_available"] = self._primary_available

        return pose

    def estimate_poses(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        """Estimate poses for all objects with hybrid approach.

        Overrides base method to collect statistics about backend usage.
        """
        poses = super().estimate_poses(image, depth_map, masks, boxes, K, labels)

        # Log statistics
        if poses:
            orient_count = sum(
                1 for p in poses
                if p.metadata.get("hybrid_backend_used") == "orient_anything"
            )
            depth_count = len(poses) - orient_count

            logger.debug(
                f"Hybrid estimation: {orient_count} Orient-Anything, "
                f"{depth_count} depth-based out of {len(poses)} objects"
            )

        return poses

    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities."""
        return {
            "requires_depth": True,
            "requires_mask": True,
            "estimates_orientation": True,
            "gpu_required": False,  # Can work without GPU
            "provides_confidence": True,
            "hybrid": True,
            "primary_backend": self._primary_backend_name,
            "fallback_backend": self._fallback_backend_name,
            "primary_available": self._primary_available,
        }
