"""Depth-based warping backend for novel view synthesis.

This is the original implementation refactored as a pluggable backend.
Fast, CPU-friendly, but cannot hallucinate disoccluded regions.

Method:
1. Back-project image to 3D using depth (Eq. 1): p_3D = D(u,v) * K^(-1) * [u, v, 1]^T
2. Apply rotation transform (Eq. 2): [u', v', z'] = K * R * p_3D
3. Warp pixels with z-buffer occlusion handling
4. Inpaint holes using OpenCV Telea algorithm
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from .base import ViewSynthesizerBase, SynthesizedView, SynthesisConfig
from .registry import ViewSynthesizerRegistry


@ViewSynthesizerRegistry.register("depth_warping")
class DepthWarpingSynthesizer(ViewSynthesizerBase):
    """Depth-based warping for novel view synthesis.

    This backend uses depth maps to warp images to new viewpoints.
    It is fast and does not require GPU, but quality degrades with
    complex scenes and cannot generate new content for disoccluded regions.

    Attributes:
        backend_name: "depth_warping"

    Configuration:
        - rotation_axis: "vertical" or "horizontal"
        - inpainting_method: "opencv", "none"
        - max_hole_ratio: Maximum acceptable hole ratio (default 0.15)
    """

    backend_name = "depth_warping"

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self._rotation_axis = getattr(config, 'rotation_axis', 'vertical')
        self._inpainting_method = getattr(config, 'inpainting_method', 'opencv')
        self._max_hole_ratio = getattr(config, 'max_hole_ratio', 0.15)

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize a single novel view using depth-based warping.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Depth map (H, W) in metric scale. Required.
            K: Camera intrinsic matrix (3, 3)
            angle: Rotation angle in degrees

        Returns:
            SynthesizedView with warped image

        Raises:
            ValueError: If depth_map is None
        """
        if depth_map is None:
            raise ValueError(
                f"{self.backend_name} requires depth_map. "
                "Provide depth_map or use a diffusion-based backend."
            )

        H, W = image.shape[:2]

        # Step 1: Back-project to 3D point cloud
        point_cloud = self._back_project(depth_map, K)

        # Step 2: Apply rotation
        R = self._rotation_matrix(np.deg2rad(angle), axis=self._rotation_axis)

        # Rotate all points: (3, 3) @ (N, 3).T -> (3, N)
        points_flat = point_cloud.reshape(-1, 3)
        rotated_points = (R @ points_flat.T).T
        rotated_points = rotated_points.reshape(H, W, 3)

        # Step 3: Re-project to 2D
        new_coords = self._project(rotated_points, K)
        new_depth = rotated_points[..., 2]

        # Step 4: Warp image with z-buffer
        warped_image, valid_mask = self._warp_image(
            image, new_coords, new_depth, depth_map
        )

        # Step 5: Calculate hole ratio and optionally inpaint
        hole_ratio = 1.0 - valid_mask.mean()

        if hole_ratio > 0 and hole_ratio < self._max_hole_ratio:
            if self._inpainting_method == "opencv":
                final_image = self._inpaint_opencv(warped_image, valid_mask)
            else:
                final_image = warped_image
        else:
            final_image = warped_image

        return SynthesizedView(
            image=final_image,
            angle=angle,
            mask=valid_mask,
            hole_ratio=hole_ratio,
            backend=self.backend_name,
            quality_score=1.0 - hole_ratio,  # Simple quality metric
            metadata={
                "inpainting_method": self._inpainting_method,
                "rotation_axis": self._rotation_axis,
            }
        )

    def _back_project(self, depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Back-project depth map to 3D point cloud.

        Implements: p_3D = D(u,v) * K^(-1) * [u, v, 1]^T

        Args:
            depth_map: Depth map (H, W)
            K: Camera intrinsic matrix (3, 3)

        Returns:
            Point cloud (H, W, 3)
        """
        H, W = depth_map.shape
        K_inv = np.linalg.inv(K)

        # Create pixel coordinate grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # Homogeneous coordinates [u, v, 1]
        ones = np.ones((H, W))
        pixels = np.stack([u, v, ones], axis=-1)  # (H, W, 3)

        # Back-project: directions = K^(-1) @ [u, v, 1]
        directions = np.einsum('ij,hwj->hwi', K_inv, pixels)

        # Scale by depth
        point_cloud = directions * depth_map[..., np.newaxis]

        return point_cloud

    def _project(self, points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates.

        Args:
            points_3d: 3D points (H, W, 3)
            K: Camera intrinsic matrix (3, 3)

        Returns:
            2D coordinates (H, W, 2)
        """
        # Project: K @ points_3d
        projected = np.einsum('ij,hwj->hwi', K, points_3d)

        # Perspective division
        z = projected[..., 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)  # Avoid division by zero

        uv = projected[..., :2] / z
        return uv

    def _rotation_matrix(self, angle: float, axis: str = "vertical") -> np.ndarray:
        """Create 3D rotation matrix.

        Args:
            angle: Rotation angle in radians
            axis: "vertical" (Y-axis/yaw) or "horizontal" (X-axis/pitch)

        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(angle), np.sin(angle)

        if axis == "vertical":
            # Rotation around Y-axis (yaw) - left/right viewing
            R = np.array([
                [c,  0, s],
                [0,  1, 0],
                [-s, 0, c]
            ])
        else:
            # Rotation around X-axis (pitch) - up/down viewing
            R = np.array([
                [1, 0,  0],
                [0, c, -s],
                [0, s,  c]
            ])

        return R

    def _warp_image(
        self,
        image: np.ndarray,
        new_coords: np.ndarray,
        new_depth: np.ndarray,
        original_depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp image using computed coordinates with z-buffer occlusion.

        Uses depth-based z-buffering to handle occlusions correctly.
        Renders from far to near so closer pixels overwrite farther ones.

        Args:
            image: Source image (H, W, 3)
            new_coords: Target coordinates (H, W, 2)
            new_depth: Depth after rotation (H, W)
            original_depth: Original depth map (H, W)

        Returns:
            Tuple of (warped_image, valid_mask)
        """
        H, W = image.shape[:2]

        # Initialize output buffers
        warped = np.zeros_like(image)
        valid_mask = np.zeros((H, W), dtype=bool)
        depth_buffer = np.full((H, W), np.inf)

        # Round to integer coordinates
        u_new = new_coords[..., 0].astype(np.int32)
        v_new = new_coords[..., 1].astype(np.int32)

        # Valid coordinate mask
        valid = (
            (u_new >= 0) & (u_new < W) &
            (v_new >= 0) & (v_new < H) &
            (new_depth > 0)
        )

        # Source coordinate grids
        v_src, u_src = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Flatten for processing
        u_src_flat = u_src[valid]
        v_src_flat = v_src[valid]
        u_new_flat = u_new[valid]
        v_new_flat = v_new[valid]
        depth_flat = new_depth[valid]

        # Sort by depth (far to near) for z-buffer
        sort_idx = np.argsort(-depth_flat)

        # Render with z-buffer (vectorized where possible)
        for idx in sort_idx:
            u_t, v_t = u_new_flat[idx], v_new_flat[idx]
            u_s, v_s = u_src_flat[idx], v_src_flat[idx]
            d = depth_flat[idx]

            if d < depth_buffer[v_t, u_t]:
                warped[v_t, u_t] = image[v_s, u_s]
                depth_buffer[v_t, u_t] = d
                valid_mask[v_t, u_t] = True

        return warped, valid_mask

    def _inpaint_opencv(
        self,
        image: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint holes using OpenCV Telea algorithm.

        Args:
            image: Image with holes (H, W, 3)
            valid_mask: Mask of valid pixels (H, W)

        Returns:
            Inpainted image
        """
        # Convert mask to uint8 (0 = valid, 255 = need inpainting)
        inpaint_mask = (~valid_mask).astype(np.uint8) * 255

        # Dilate mask slightly to ensure clean borders
        kernel = np.ones((3, 3), np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

        # Apply Telea inpainting
        inpainted = cv2.inpaint(
            image,
            inpaint_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )

        return inpainted

    def get_capabilities(self) -> dict:
        """Return backend capabilities."""
        return {
            "requires_depth": True,
            "can_hallucinate": False,
            "supports_arbitrary_angles": True,
            "gpu_required": False,
            "inpainting_methods": ["opencv", "none"],
        }


# Backward compatibility alias
ViewSynthesizer = DepthWarpingSynthesizer
