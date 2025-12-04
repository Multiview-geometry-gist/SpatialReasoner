"""3D geometry utilities for spatial reasoning.

Provides:
- Distance and direction computations
- Camera projection/unprojection
- 3D bounding box utilities
- Relative spatial relationship checks
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)


def direction_vector(from_point: np.ndarray, to_point: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute direction vector from one point to another.

    Args:
        from_point: Start point
        to_point: End point
        normalize: If True, return unit vector
    """
    vec = to_point - from_point
    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
    return vec


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
    """
    Compute angle between two vectors.

    Args:
        v1, v2: Input vectors
        degrees: If True, return angle in degrees
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle) if degrees else angle


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return np.dot(v1, v2) / (norm1 * norm2)


# Camera utilities

def estimate_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> Dict[str, float]:
    """
    Estimate camera intrinsics from image size and assumed FOV.

    Args:
        width, height: Image dimensions
        fov_deg: Horizontal field of view in degrees

    Returns:
        Dict with fx, fy, cx, cy
    """
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels

    return {
        "fx": fx,
        "fy": fy,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }


def intrinsics_to_matrix(intrinsics: Dict[str, float]) -> np.ndarray:
    """Convert intrinsics dict to 3x3 matrix K."""
    return np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ])


def project_point(point_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D point to 2D image coordinates.

    Args:
        point_3d: Point in camera coordinates (3,)
        K: Intrinsic matrix (3, 3)

    Returns:
        2D pixel coordinates (2,)
    """
    projected = K @ point_3d
    z = projected[2]

    if abs(z) < 1e-8:
        return np.array([0, 0])

    return projected[:2] / z


def unproject_pixel(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    """
    Unproject 2D pixel to 3D point given depth.

    Args:
        pixel: 2D pixel coordinates [u, v]
        depth: Depth value at pixel
        K: Intrinsic matrix (3, 3)

    Returns:
        3D point in camera coordinates
    """
    K_inv = np.linalg.inv(K)
    pixel_h = np.array([pixel[0], pixel[1], 1.0])
    direction = K_inv @ pixel_h
    return direction * depth


def depth_to_pointcloud(
    depth_map: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud.

    Implements Equation (1): p_3D = D(u,v) * K^(-1) * [u, v, 1]^T

    Args:
        depth_map: Depth map (H, W)
        K: Intrinsic matrix (3, 3)
        mask: Optional binary mask (H, W)

    Returns:
        Point cloud (N, 3) where N = H*W or number of masked points
    """
    H, W = depth_map.shape
    K_inv = np.linalg.inv(K)

    # Create pixel grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Homogeneous pixel coordinates
    ones = np.ones((H, W))
    pixels = np.stack([u, v, ones], axis=-1)  # (H, W, 3)

    # Back-project
    directions = np.einsum('ij,hwj->hwi', K_inv, pixels)  # (H, W, 3)
    points_3d = directions * depth_map[..., np.newaxis]  # (H, W, 3)

    if mask is not None:
        points_3d = points_3d[mask]
    else:
        points_3d = points_3d.reshape(-1, 3)

    return points_3d


# Spatial relationship checks

def is_above(obj_a_pos: np.ndarray, obj_b_pos: np.ndarray, threshold: float = 0.0) -> bool:
    """Check if object A is above object B (Y-up convention)."""
    return obj_a_pos[1] > obj_b_pos[1] + threshold


def is_left_of(obj_a_pos: np.ndarray, obj_b_pos: np.ndarray, threshold: float = 0.0) -> bool:
    """Check if object A is to the left of object B (X-right convention)."""
    return obj_a_pos[0] < obj_b_pos[0] - threshold


def is_in_front_of(obj_a_pos: np.ndarray, obj_b_pos: np.ndarray, threshold: float = 0.0) -> bool:
    """Check if object A is in front of object B (Z-forward, smaller Z = closer)."""
    return obj_a_pos[2] < obj_b_pos[2] - threshold


def is_closer_to_camera(obj_a_pos: np.ndarray, obj_b_pos: np.ndarray) -> bool:
    """Check if object A is closer to camera (assumed at origin)."""
    return np.linalg.norm(obj_a_pos) < np.linalg.norm(obj_b_pos)


def is_facing_toward(
    obj_a_pos: np.ndarray,
    obj_a_forward: np.ndarray,
    obj_b_pos: np.ndarray,
    threshold: float = 0.5
) -> bool:
    """
    Check if object A is facing toward object B.

    Computation: v_fwd . (t_B - t_A) / ||t_B - t_A|| > threshold
    """
    direction_to_b = direction_vector(obj_a_pos, obj_b_pos)
    cos_sim = cosine_similarity(obj_a_forward, direction_to_b)
    return cos_sim > threshold


def is_same_direction(
    forward_a: np.ndarray,
    forward_b: np.ndarray,
    threshold: float = 0.5
) -> bool:
    """
    Check if two objects are facing the same direction.

    Uses absolute cosine similarity to handle opposite directions.
    """
    cos_sim = abs(cosine_similarity(forward_a, forward_b))
    return cos_sim > threshold


# 3D bounding box utilities

def compute_3d_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned 3D bounding box from points.

    Returns:
        min_corner: (3,) minimum x, y, z
        max_corner: (3,) maximum x, y, z
    """
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return min_corner, max_corner


def bbox_dimensions(min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """Compute width, height, depth of bounding box."""
    return max_corner - min_corner


def bbox_center(min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """Compute center of bounding box."""
    return (min_corner + max_corner) / 2


def bbox_volume(min_corner: np.ndarray, max_corner: np.ndarray) -> float:
    """Compute volume of bounding box."""
    dims = bbox_dimensions(min_corner, max_corner)
    return np.prod(dims)


# Transform utilities

def apply_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray
) -> np.ndarray:
    """
    Apply rigid transformation to points.

    Args:
        points: Points (N, 3)
        rotation: Rotation matrix (3, 3)
        translation: Translation vector (3,)

    Returns:
        Transformed points (N, 3)
    """
    return (rotation @ points.T).T + translation


def rotation_matrix_y(angle: float, degrees: bool = True) -> np.ndarray:
    """Create rotation matrix around Y axis (yaw)."""
    if degrees:
        angle = np.radians(angle)

    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_x(angle: float, degrees: bool = True) -> np.ndarray:
    """Create rotation matrix around X axis (pitch)."""
    if degrees:
        angle = np.radians(angle)

    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])


def rotation_matrix_z(angle: float, degrees: bool = True) -> np.ndarray:
    """Create rotation matrix around Z axis (roll)."""
    if degrees:
        angle = np.radians(angle)

    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
