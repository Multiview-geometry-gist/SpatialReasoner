"""Depth estimation using Depth Anything v2."""

import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F

from .config import DepthConfig


class DepthEstimator:
    """Wrapper for Depth Anything v2 model."""

    def __init__(self, config: DepthConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def estimate(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Estimate relative depth (0=close, 1=far)."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        original_size = image.size  # (W, H)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth

        # Resize to original
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.cpu().numpy()

    def estimate_metric(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Estimate approximate metric depth."""
        relative = self.estimate(image)
        # Convert: close=high value in relative, we want close=low meters
        return self.config.scale_factor * (1.0 - relative + 0.1)

    def get_point_cloud(
        self, depth_map: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """Back-project depth to 3D. Returns (H, W, 3)."""
        H, W = depth_map.shape
        K_inv = np.linalg.inv(K)

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones((H, W))
        pixels = np.stack([u, v, ones], axis=-1)

        directions = np.einsum('ij,hwj->hwi', K_inv, pixels)
        return directions * depth_map[..., np.newaxis]

    @staticmethod
    def estimate_intrinsics(width: int, height: int) -> np.ndarray:
        """Estimate camera intrinsics from image size."""
        fx = fy = max(width, height)
        cx, cy = width / 2.0, height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
