"""Depth estimation using Depth Anything v2.

Provides depth map estimation and 3D point cloud generation.
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn.functional as F

from .config import DepthConfig


class DepthEstimator:
    """Wrapper for Depth Anything v2 model."""

    def __init__(self, config: DepthConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading model until needed."""
        if self._initialized:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
        except ImportError:
            raise ImportError(
                "transformers is required for depth estimation. "
                "Install with: pip install transformers"
            )

    @torch.no_grad()
    def estimate(
        self,
        image: Union[Image.Image, np.ndarray, str],
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Estimate relative depth from a single image.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            return_tensor: If True, return torch.Tensor; else numpy array

        Returns:
            Depth map (H, W), normalized to [0, 1], 0=closest, 1=farthest
        """
        self._lazy_init()

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        original_size = image.size  # (W, H)

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Resize to original size
        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(0),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Normalize to [0, 1]
        depth_min = predicted_depth.min()
        depth_max = predicted_depth.max()
        normalized_depth = (predicted_depth - depth_min) / (depth_max - depth_min + 1e-8)

        if return_tensor:
            return normalized_depth
        return normalized_depth.cpu().numpy()

    def estimate_metric(
        self,
        image: Union[Image.Image, np.ndarray, str],
        scale_factor: float = 10.0,
    ) -> np.ndarray:
        """
        Estimate metric depth (approximate).

        Note: Monocular depth is scale-ambiguous. This applies a heuristic scale.

        Args:
            image: Input image
            scale_factor: Approximate scene depth range in meters

        Returns:
            Depth map in metric scale (meters), shape (H, W)
        """
        relative_depth = self.estimate(image, return_tensor=False)

        # Convert from inverse depth to metric depth
        # Depth Anything outputs: close = high value
        # We want: close = low value (in meters)
        metric_depth = scale_factor * (1.0 - relative_depth + 0.1)

        return metric_depth

    def get_point_cloud(
        self,
        image: Union[Image.Image, np.ndarray],
        depth_map: np.ndarray,
        K: Optional[np.ndarray] = None,
        colors: bool = True
    ) -> np.ndarray:
        """
        Back-project depth map to 3D point cloud.

        Implements Equation (1): p_3D = D(u,v) * K^(-1) * [u, v, 1]^T

        Args:
            image: Original image (for colors)
            depth_map: Depth map (H, W)
            K: Camera intrinsic matrix (3, 3). If None, estimated.
            colors: Whether to include RGB colors

        Returns:
            Point cloud (H, W, 3) or (H, W, 6) if colors=True
        """
        if isinstance(image, Image.Image):
            W, H = image.size
            image_np = np.array(image)
        else:
            H, W = image.shape[:2]
            image_np = image

        # Estimate intrinsics if not provided
        if K is None:
            K = self.estimate_intrinsics(W, H)

        K_inv = np.linalg.inv(K)

        # Create pixel coordinate grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # Homogeneous pixel coordinates
        ones = np.ones_like(u)
        pixels = np.stack([u, v, ones], axis=-1)  # (H, W, 3)

        # Back-project: p_3D = D(u,v) * K^(-1) * [u, v, 1]^T
        directions = np.einsum('ij,hwj->hwi', K_inv, pixels)  # (H, W, 3)
        point_cloud = directions * depth_map[..., np.newaxis]  # (H, W, 3)

        if colors and image_np is not None:
            colors_normalized = image_np.astype(np.float32) / 255.0
            point_cloud = np.concatenate([point_cloud, colors_normalized], axis=-1)

        return point_cloud

    @staticmethod
    def estimate_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
        """
        Estimate camera intrinsics from image size.

        Args:
            width, height: Image dimensions
            fov_deg: Assumed horizontal field of view in degrees

        Returns:
            K: Camera intrinsic matrix (3, 3)
        """
        # Estimate focal length from FOV
        fov_rad = np.radians(fov_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels

        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ]
        ], dtype=np.float64)

        return K
