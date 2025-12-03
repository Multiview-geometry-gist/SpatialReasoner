"""
Quaternion utilities for rotation-aware spatial reasoning.

Implements geodesic loss (Eq. 4) and quaternion operations.
"""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Union, Optional


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion to unit norm. q: (..., 4) [w, x, y, z]"""
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)


def geodesic_loss(
    q_pred: torch.Tensor, q_gt: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Geodesic loss: L_rot = arccos(|q_pred . q_gt|)

    Args:
        q_pred, q_gt: (..., 4) in [w, x, y, z] format
        reduction: "mean", "sum", or "none"
    """
    q_pred = normalize_quaternion(q_pred)
    q_gt = normalize_quaternion(q_gt)

    dot = torch.sum(q_pred * q_gt, dim=-1)
    dot = torch.clamp(torch.abs(dot), -1.0 + 1e-7, 1.0 - 1e-7)
    loss = torch.acos(dot)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (..., 4) [w,x,y,z] to rotation matrix (..., 3, 3)."""
    q = normalize_quaternion(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)
    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y*z - x*w)
    r20 = 2 * (x*z - y*w)
    r21 = 2 * (y*z + x*w)
    r22 = 1 - 2 * (x**2 + y**2)

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def quaternion_to_forward(q: torch.Tensor) -> torch.Tensor:
    """Forward direction: q * (0,0,1) * q* → (..., 3)"""
    R = quaternion_to_rotation_matrix(q)
    return R[..., :, 2]


def quaternion_to_left(q: torch.Tensor) -> torch.Tensor:
    """Left direction: -X column of rotation matrix."""
    R = quaternion_to_rotation_matrix(q)
    return -R[..., :, 0]


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: [w, -x, -y, -z]"""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


class GeodesicLoss(nn.Module):
    """PyTorch module for geodesic loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
        return geodesic_loss(q_pred, q_gt, self.reduction)


def parse_quaternion_from_text(text: str) -> Optional[np.ndarray]:
    """
    Parse quaternion from generated text.
    Looks for: "quaternion: (w, x, y, z)" or "orientation: [w, x, y, z]"
    """
    patterns = [
        r"quaternion[:\s]*\(?([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\)?",
        r"orientation[:\s]*\[?([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            q = np.array([float(match.group(i)) for i in range(1, 5)])
            return q / (np.linalg.norm(q) + 1e-8)
    return None


# NumPy versions for data generation
def geodesic_distance_np(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic distance between two quaternions (numpy)."""
    dot = np.abs(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)))
    return np.arccos(np.clip(dot, -1.0, 1.0))
