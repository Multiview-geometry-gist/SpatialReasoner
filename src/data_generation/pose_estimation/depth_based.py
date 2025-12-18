"""Depth-based pose estimation backend.

This is the original implementation refactored as a pluggable backend.
Uses depth map back-projection and PCA for orientation estimation.

Method:
1. Back-project masked depth pixels to 3D point cloud
2. Compute centroid as position
3. Compute bounding box as scale
4. Use PCA on point cloud for orientation (or camera-facing fallback)
5. Convert rotation matrix to quaternion
"""

import numpy as np
from typing import Optional, List
import logging

from .base import PoseEstimatorBase, ObjectPose, PoseEstimationConfig
from .registry import PoseEstimatorRegistry

# Import quaternion utilities
import sys
from pathlib import Path
_SPATIAL_REASONER_PATH = Path(__file__).parent.parent.parent
if str(_SPATIAL_REASONER_PATH) not in sys.path:
    sys.path.insert(0, str(_SPATIAL_REASONER_PATH))

from spatial_reasoner.utils import quaternion as quat

logger = logging.getLogger(__name__)


@PoseEstimatorRegistry.register("depth_based", priority=100)
class DepthBasedEstimator(PoseEstimatorBase):
    """Depth-based 6-DOF pose estimation.

    This backend uses depth maps to estimate object poses through:
    - 3D point cloud back-projection from depth
    - Centroid computation for position
    - Bounding box computation for scale
    - PCA-based orientation estimation

    Attributes:
        backend_name: "depth_based"

    Configuration:
        - use_pca_orientation: If True, use PCA for orientation;
                              otherwise use camera-facing heuristic
        - min_mask_pixels: Minimum pixels for valid object
    """

    backend_name = "depth_based"

    def __init__(self, config: PoseEstimationConfig):
        super().__init__(config)
        self._use_pca = getattr(config, 'use_pca_orientation', True)

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
        """Estimate pose for a single object using depth-based method.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            mask: Segmentation mask (H, W)
            box: 2D bounding box [x1, y1, x2, y2]
            K: Camera intrinsics (3, 3)
            label: Optional object label
            object_id: Unique identifier

        Returns:
            ObjectPose or None if estimation failed
        """
        object_pixels = mask > 0

        if object_pixels.sum() < getattr(self.config, 'min_mask_pixels', 10):
            return None

        H, W = depth_map.shape
        K_inv = np.linalg.inv(K)

        # Create coordinate grids
        u_coords = np.arange(W)
        v_coords = np.arange(H)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # Get 3D points for this object
        object_depths = depth_map[object_pixels]
        object_u = u_grid[object_pixels]
        object_v = v_grid[object_pixels]

        # Back-project to 3D: p_3D = D(u,v) * K^(-1) * [u, v, 1]^T
        points_2d = np.stack([
            object_u,
            object_v,
            np.ones_like(object_u)
        ], axis=-1)

        directions = (K_inv @ points_2d.T).T
        points_3d = directions * object_depths[:, np.newaxis]

        # Position: centroid of 3D points
        position = points_3d.mean(axis=0)

        # Scale: 3D axis-aligned bounding box
        scale = points_3d.max(axis=0) - points_3d.min(axis=0)
        scale = np.maximum(scale, 0.01)  # Prevent zero scale

        # Orientation
        if self._use_pca:
            rotation_matrix = self._estimate_orientation_pca(points_3d, position)
        else:
            rotation_matrix = self._camera_facing_rotation(position)

        # Convert to quaternion
        quaternion = quat.from_rotation_matrix(rotation_matrix)

        return ObjectPose(
            position=position,
            scale=scale,
            quaternion=quaternion,
            rotation_matrix=rotation_matrix,
            mask=mask,
            bbox_2d=box,
            confidence=1.0,  # Depth-based doesn't provide confidence
            label=label,
            object_id=object_id,
            metadata={
                "backend": self.backend_name,
                "orientation_method": "pca" if self._use_pca else "camera_facing",
                "num_points": len(points_3d),
            }
        )

    def _estimate_orientation_pca(
        self,
        points_3d: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """Estimate orientation using PCA on 3D points.

        Args:
            points_3d: 3D point cloud (N, 3)
            centroid: Centroid of point cloud (3,)

        Returns:
            3x3 rotation matrix
        """
        centered = points_3d - centroid

        try:
            _, _, Vt = np.linalg.svd(centered)
            axes = Vt.T  # Principal components as columns

            # Ensure right-handed coordinate system
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1

            return axes

        except np.linalg.LinAlgError:
            logger.warning("SVD failed, using identity rotation")
            return np.eye(3)

    def _camera_facing_rotation(self, position: np.ndarray) -> np.ndarray:
        """Compute rotation matrix for object facing the camera.

        The object's forward direction points toward the camera.

        Args:
            position: 3D position in camera coordinates

        Returns:
            3x3 rotation matrix
        """
        # Forward direction: pointing toward camera (negative position)
        forward = -position / (np.linalg.norm(position) + 1e-8)

        # World up direction
        up = np.array([0, 1, 0])

        # Right direction (cross product)
        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)

        if right_norm < 1e-6:
            # Object is directly above/below camera
            right = np.array([1, 0, 0])
        else:
            right = right / right_norm

        # Recompute up to ensure orthogonality
        up = np.cross(forward, right)

        # Build rotation matrix: [right | up | forward]
        R = np.stack([right, up, forward], axis=1)
        return R

    def get_capabilities(self) -> dict:
        """Return backend capabilities."""
        return {
            "requires_depth": True,
            "requires_mask": True,
            "estimates_orientation": True,
            "gpu_required": False,
            "provides_confidence": False,
            "orientation_methods": ["pca", "camera_facing"],
        }


# Backward compatibility alias
PoseEstimator = DepthBasedEstimator
