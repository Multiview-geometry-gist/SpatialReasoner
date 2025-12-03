"""6-DOF pose estimation and quaternion conversion."""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Optional
from dataclasses import dataclass

from .config import PoseEstimationConfig


@dataclass
class ObjectPose:
    """6-DOF pose for a single object."""
    position: np.ndarray  # (3,) - t_i
    scale: np.ndarray  # (3,) - d_i [w, h, d]
    quaternion: np.ndarray  # (4,) - q_i [qw, qx, qy, qz]
    rotation_matrix: np.ndarray  # (3, 3)
    mask: np.ndarray  # (H, W) bool
    bbox_2d: np.ndarray  # (4,) [x1, y1, x2, y2]
    object_id: int = 0
    label: Optional[str] = None

    def to_10d_vector(self) -> np.ndarray:
        """s_i = [t_i, d_i, q_i] in R^10"""
        return np.concatenate([self.position, self.scale, self.quaternion])

    @property
    def forward_direction(self) -> np.ndarray:
        """Forward vector: q * (0,0,1) * q*"""
        r = Rotation.from_quat([self.quaternion[1], self.quaternion[2],
                                self.quaternion[3], self.quaternion[0]])
        return r.apply([0, 0, 1])

    @property
    def left_direction(self) -> np.ndarray:
        """Left vector: q * (-1,0,0) * q*"""
        r = Rotation.from_quat([self.quaternion[1], self.quaternion[2],
                                self.quaternion[3], self.quaternion[0]])
        return r.apply([-1, 0, 0])


class PoseEstimator:
    """Estimate 6-DOF poses for detected objects."""

    def __init__(self, config: PoseEstimationConfig):
        self.config = config

    def estimate_poses(
        self,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        if self.config.method == "depth_based":
            return self._estimate_depth_based(depth_map, masks, boxes, K, labels)
        return self._estimate_heuristic(depth_map, masks, boxes, K, labels)

    def _estimate_depth_based(
        self, depth_map: np.ndarray, masks: np.ndarray, boxes: np.ndarray,
        K: np.ndarray, labels: Optional[List[str]] = None
    ) -> List[ObjectPose]:
        poses = []
        K_inv = np.linalg.inv(K)
        H, W = depth_map.shape
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            obj_pixels = mask > 0
            if obj_pixels.sum() < 10:
                continue

            # Get 3D points
            depths = depth_map[obj_pixels]
            u, v = u_grid[obj_pixels], v_grid[obj_pixels]
            pts_2d = np.stack([u, v, np.ones_like(u)], axis=-1)
            directions = (K_inv @ pts_2d.T).T
            pts_3d = directions * depths[:, np.newaxis]

            # Position = centroid
            position = pts_3d.mean(axis=0)

            # Scale = 3D bbox
            scale = np.maximum(pts_3d.max(0) - pts_3d.min(0), 0.01)

            # Orientation via PCA or camera-facing
            if self.config.use_pca_orientation:
                R = self._pca_orientation(pts_3d, position)
            else:
                R = self._camera_facing(position)

            quaternion = self._mat_to_quat(R)

            poses.append(ObjectPose(
                position=position, scale=scale, quaternion=quaternion,
                rotation_matrix=R, mask=mask, bbox_2d=box,
                object_id=i, label=labels[i] if labels else None
            ))

        return poses

    def _estimate_heuristic(
        self, depth_map: np.ndarray, masks: np.ndarray, boxes: np.ndarray,
        K: np.ndarray, labels: Optional[List[str]] = None
    ) -> List[ObjectPose]:
        poses = []
        K_inv = np.linalg.inv(K)

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            if mask.sum() < 10:
                continue

            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            median_depth = np.median(depth_map[mask > 0])

            direction = K_inv @ np.array([cx, cy, 1])
            position = direction * median_depth

            # Scale from 2D box
            fx = K[0, 0]
            w, h = box[2] - box[0], box[3] - box[1]
            sx = w * median_depth / fx
            sy = h * median_depth / fx
            scale = np.array([sx, sy, min(sx, sy)])

            R = self._camera_facing(position)
            quaternion = self._mat_to_quat(R)

            poses.append(ObjectPose(
                position=position, scale=scale, quaternion=quaternion,
                rotation_matrix=R, mask=mask, bbox_2d=box,
                object_id=i, label=labels[i] if labels else None
            ))

        return poses

    def _pca_orientation(self, points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        centered = points - centroid
        try:
            _, _, Vt = np.linalg.svd(centered)
            R = Vt.T
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            return R
        except:
            return np.eye(3)

    def _camera_facing(self, position: np.ndarray) -> np.ndarray:
        forward = -position / (np.linalg.norm(position) + 1e-8)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        return np.stack([right, up, forward], axis=1)

    def _mat_to_quat(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to [qw, qx, qy, qz]."""
        r = Rotation.from_matrix(R)
        q_xyzw = r.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-8)
        if q_wxyz[0] < 0:
            q_wxyz = -q_wxyz
        return q_wxyz


# Quaternion utilities
def geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic distance: arccos(|q1 . q2|)"""
    dot = np.abs(np.dot(q1, q2))
    return np.arccos(np.clip(dot, -1.0, 1.0))


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
