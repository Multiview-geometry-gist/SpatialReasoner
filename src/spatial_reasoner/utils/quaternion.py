"""Quaternion utilities for rotation-aware spatial reasoning.

Provides:
- Quaternion normalization, multiplication, conjugate
- Quaternion <-> Rotation matrix conversion
- Quaternion <-> Euler angles conversion
- Geodesic (angular) distance between quaternions
- Direction vector computation from quaternion

Convention: Quaternion format is [w, x, y, z] (scalar-first)
"""

import numpy as np
from typing import Tuple, Union, Optional


def normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + eps)


def conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate: q* = [w, -x, -y, -z]."""
    result = q.copy()
    result[..., 1:] = -result[..., 1:]
    return result


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product).
    q1, q2: [..., 4] in [w, x, y, z] format
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (same as conjugate for unit quaternions)."""
    return conjugate(normalize(q))


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        q: Quaternion array of shape (..., 4)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Rotation matrix elements
    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)

    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x*x + z*z)
    r12 = 2 * (y*z - x*w)

    r20 = 2 * (x*z - y*w)
    r21 = 2 * (y*z + x*w)
    r22 = 1 - 2 * (x*x + y*y)

    return np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1),
    ], axis=-2)


def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Ensure w > 0 for canonical form
    if q[0] < 0:
        q = -q
    return normalize(q)


def to_euler(q: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX convention (yaw-pitch-roll).
    """
    q = normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    return roll, pitch, yaw


def from_euler(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """
    Create quaternion from Euler angles (ZYX convention).
    """
    if degrees:
        roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return normalize(np.array([w, x, y, z]))


def from_axis_angle(axis: np.ndarray, angle: float, degrees: bool = True) -> np.ndarray:
    """Create quaternion from axis-angle representation."""
    if degrees:
        angle = np.radians(angle)

    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half_angle = angle / 2

    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def angle_between(q1: np.ndarray, q2: np.ndarray, degrees: bool = True) -> float:
    """
    Compute the angular distance between two quaternions.

    This is the geodesic distance on the quaternion manifold:
    angle = 2 * arccos(|q1 . q2|)
    """
    q1, q2 = normalize(q1), normalize(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(dot)

    if degrees:
        return np.degrees(angle)
    return angle


def geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute geodesic distance (in radians) between quaternions.

    Implements Equation 4 from the paper: L_rot = arccos(|q_pred . q_gt|)
    """
    q1, q2 = normalize(q1), normalize(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q1, q2: Unit quaternions
        t: Interpolation parameter in [0, 1]
    """
    q1, q2 = normalize(q1), normalize(q2)

    dot = np.dot(q1, q2)

    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Linear interpolation for very close quaternions
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return s1 * q1 + s2 * q2


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q.

    v' = q * [0, v] * q^(-1)
    """
    q = normalize(q)
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_inv = conjugate(q)

    rotated = multiply(multiply(q, v_quat), q_inv)
    return rotated[1:]


def forward_direction(q: np.ndarray) -> np.ndarray:
    """
    Compute forward direction vector from quaternion.

    Forward is defined as +Z axis in local frame: [0, 0, 1]
    v_fwd = q * [0, 0, 1] * q*
    """
    return rotate_vector(np.array([0, 0, 1]), q)


def left_direction(q: np.ndarray) -> np.ndarray:
    """
    Compute left direction vector from quaternion.

    Left is defined as -X axis in local frame: [-1, 0, 0]
    """
    return rotate_vector(np.array([-1, 0, 0]), q)


def up_direction(q: np.ndarray) -> np.ndarray:
    """
    Compute up direction vector from quaternion.

    Up is defined as +Y axis in local frame: [0, 1, 0]
    """
    return rotate_vector(np.array([0, 1, 0]), q)


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Create quaternion that orients from position toward target.

    Args:
        position: Current position
        target: Target to look at
        up: Up vector (default: [0, 1, 0])
    """
    if up is None:
        up = np.array([0, 1, 0])

    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    up = np.cross(forward, right)

    R = np.stack([right, up, forward], axis=1)
    return from_rotation_matrix(R)


# Utility for parsing quaternion from text (for reward computation)
def parse_from_text(text: str) -> Optional[np.ndarray]:
    """
    Parse quaternion values from generated text.

    Looks for patterns like:
    - "quaternion: (0.5, 0.5, 0.5, 0.5)"
    - "orientation: [0.5, 0.5, 0.5, 0.5]"
    """
    import re

    patterns = [
        r"quaternion[:\s]*[\(\[]?\s*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\s*[\)\]]?",
        r"orientation[:\s]*[\(\[]?\s*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\s*[\)\]]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            values = [float(match.group(i)) for i in range(1, 5)]
            return normalize(np.array(values))

    return None
