"""6-DOF pose estimation with quaternion support.

Estimates position, scale, and orientation for detected objects.
Implements the 10D object representation: s_i = [t_i, d_i, q_i]
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from .config import PoseEstimationConfig

# Import quaternion utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_reasoner.utils import quaternion as quat


@dataclass
class ObjectPose:
    """
    6-DOF pose representation for a single object.

    Attributes:
        position: 3D position (t_i) in camera coordinates
        scale: 3D bounding box dimensions (d_i) [width, height, depth]
        quaternion: Unit quaternion (q_i) [w, x, y, z]
        rotation_matrix: 3x3 rotation matrix
        mask: Binary segmentation mask
        bbox_2d: 2D bounding box [x1, y1, x2, y2]
        label: Optional object label
        object_id: Unique identifier
    """
    position: np.ndarray
    scale: np.ndarray
    quaternion: np.ndarray
    rotation_matrix: np.ndarray
    mask: np.ndarray
    bbox_2d: np.ndarray
    confidence: float = 1.0
    label: Optional[str] = None
    object_id: int = 0

    def to_10d_vector(self) -> np.ndarray:
        """
        Convert to 10D representation: s_i = [t_i, d_i, q_i]
        """
        return np.concatenate([
            self.position,   # 3D
            self.scale,      # 3D
            self.quaternion  # 4D
        ])

    @property
    def forward_direction(self) -> np.ndarray:
        """Compute forward direction vector."""
        return quat.forward_direction(self.quaternion)

    @property
    def left_direction(self) -> np.ndarray:
        """Compute left direction vector."""
        return quat.left_direction(self.quaternion)

    @property
    def up_direction(self) -> np.ndarray:
        """Compute up direction vector."""
        return quat.up_direction(self.quaternion)

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "quaternion": self.quaternion.tolist(),
            "bbox_2d": self.bbox_2d.tolist(),
            "confidence": self.confidence,
        }


class PoseEstimator:
    """Estimate 6-DOF poses for detected objects."""

    def __init__(self, config: PoseEstimationConfig):
        self.config = config

    def estimate_poses(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        """
        Estimate poses for all detected objects.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            masks: Segmentation masks (N, H, W)
            boxes: 2D bounding boxes (N, 4)
            K: Camera intrinsics (3, 3)
            labels: Optional object labels

        Returns:
            List of ObjectPose for each detected object
        """
        if self.config.method == "depth_based":
            return self._estimate_depth_based(image, depth_map, masks, boxes, K, labels)
        else:
            return self._estimate_heuristic(image, depth_map, masks, boxes, K, labels)

    def _estimate_depth_based(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        """Estimate poses using depth map and PCA for orientation."""
        poses = []
        K_inv = np.linalg.inv(K)

        H, W = depth_map.shape
        u_coords = np.arange(W)
        v_coords = np.arange(H)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            object_pixels = mask > 0

            if object_pixels.sum() < 10:
                continue

            # Get 3D points for this object
            object_depths = depth_map[object_pixels]
            object_u = u_grid[object_pixels]
            object_v = v_grid[object_pixels]

            # Back-project to 3D
            points_2d = np.stack([
                object_u,
                object_v,
                np.ones_like(object_u)
            ], axis=-1)

            directions = (K_inv @ points_2d.T).T
            points_3d = directions * object_depths[:, np.newaxis]

            # Position: centroid
            position = points_3d.mean(axis=0)

            # Scale: 3D bounding box
            scale = points_3d.max(axis=0) - points_3d.min(axis=0)
            scale = np.maximum(scale, 0.01)

            # Orientation
            if self.config.use_pca_orientation:
                rotation_matrix = self._estimate_orientation_pca(points_3d, position)
            else:
                rotation_matrix = self._camera_facing_rotation(position)

            # Convert to quaternion
            quaternion = quat.from_rotation_matrix(rotation_matrix)

            poses.append(ObjectPose(
                position=position,
                scale=scale,
                quaternion=quaternion,
                rotation_matrix=rotation_matrix,
                mask=mask,
                bbox_2d=box,
                label=labels[i] if labels else None,
                object_id=i,
            ))

        return poses

    def _estimate_heuristic(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        """Simple heuristic: assume objects face the camera."""
        poses = []
        K_inv = np.linalg.inv(K)

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            object_pixels = mask > 0

            if object_pixels.sum() < 10:
                continue

            # Position: use box center and median depth
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            object_depths = depth_map[object_pixels]
            median_depth = np.median(object_depths)

            # Back-project center
            center_2d = np.array([cx, cy, 1])
            direction = K_inv @ center_2d
            position = direction * median_depth

            # Scale from 2D box and depth
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            fx = K[0, 0]
            scale_x = box_width * median_depth / fx
            scale_y = box_height * median_depth / fx
            scale_z = min(scale_x, scale_y)
            scale = np.array([scale_x, scale_y, scale_z])

            # Camera-facing orientation
            rotation_matrix = self._camera_facing_rotation(position)
            quaternion = quat.from_rotation_matrix(rotation_matrix)

            poses.append(ObjectPose(
                position=position,
                scale=scale,
                quaternion=quaternion,
                rotation_matrix=rotation_matrix,
                mask=mask,
                bbox_2d=box,
                label=labels[i] if labels else None,
                object_id=i,
            ))

        return poses

    def _estimate_orientation_pca(
        self,
        points_3d: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """Estimate orientation using PCA on 3D points."""
        centered = points_3d - centroid

        try:
            _, _, Vt = np.linalg.svd(centered)
            axes = Vt.T

            # Ensure right-handed coordinate system
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1

            return axes
        except:
            return np.eye(3)

    def _camera_facing_rotation(self, position: np.ndarray) -> np.ndarray:
        """Compute rotation matrix for object facing the camera."""
        forward = -position / (np.linalg.norm(position) + 1e-8)
        up = np.array([0, 1, 0])

        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)

        if right_norm < 1e-6:
            right = np.array([1, 0, 0])
        else:
            right = right / right_norm

        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)
        return R

    @staticmethod
    def compute_relative_pose(pose1: ObjectPose, pose2: ObjectPose) -> Dict:
        """Compute relative pose between two objects."""
        # Relative position
        rel_position = pose2.position - pose1.position
        distance = np.linalg.norm(rel_position)

        # Direction from pose1 to pose2
        if distance > 1e-8:
            direction = rel_position / distance
        else:
            direction = np.array([0, 0, 1])

        # Relative rotation (q_rel = q2 * q1^-1)
        q1_inv = quat.inverse(pose1.quaternion)
        rel_quaternion = quat.multiply(pose2.quaternion, q1_inv)

        # Angle between forward directions
        fwd1 = pose1.forward_direction
        fwd2 = pose2.forward_direction
        angle = quat.angle_between(
            quat.from_axis_angle(np.array([0, 1, 0]), 0),
            quat.from_axis_angle(np.array([0, 1, 0]), 0)
        )

        # More useful: cosine similarity between forward directions
        fwd_similarity = np.dot(fwd1, fwd2)

        return {
            "relative_position": rel_position.tolist(),
            "distance": float(distance),
            "direction": direction.tolist(),
            "relative_quaternion": rel_quaternion.tolist(),
            "forward_similarity": float(fwd_similarity),
        }
