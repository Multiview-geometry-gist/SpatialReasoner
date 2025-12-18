"""Novel view synthesis using depth-based warping.

Implements depth-based view synthesis as described in Section 2.2:
1. Back-project image to 3D using depth (Eq. 1)
2. Apply rotation transform (Eq. 2)
3. Re-project to new view
4. Inpaint sparse regions
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .config import ViewSynthesisConfig


@dataclass
class SynthesizedView:
    """Container for a synthesized view."""
    image: np.ndarray  # RGB image (H, W, 3)
    angle: float  # Rotation angle in degrees
    mask: np.ndarray  # Valid pixel mask (H, W), True = valid
    hole_ratio: float  # Ratio of invalid pixels

    @property
    def is_valid(self) -> bool:
        """Check if view passes quality threshold."""
        return self.hole_ratio < 0.15


class ViewSynthesizer:
    """Synthesize novel views using depth-based warping."""

    def __init__(self, config: ViewSynthesisConfig):
        self.config = config
        self.rotation_angles = config.rotation_angles

    def synthesize(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> Dict[float, SynthesizedView]:
        """
        Synthesize novel views at specified rotation angles.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Depth map (H, W) in metric scale
            K: Camera intrinsic matrix (3, 3). If None, estimated.

        Returns:
            Dictionary mapping angle -> SynthesizedView
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        H, W = image.shape[:2]

        # Estimate intrinsics if not provided
        if K is None:
            K = self._estimate_intrinsics(W, H)

        # Generate views
        views = {}
        for angle in self.rotation_angles:
            if angle == 0:
                # Original view
                views[angle] = SynthesizedView(
                    image=image.copy(),
                    angle=angle,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0,
                )
            else:
                # Synthesize rotated view
                views[angle] = self._synthesize_single_view(image, depth_map, K, angle)

        return views

    def _synthesize_single_view(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """
        Synthesize a single novel view.

        Implements:
        1. Eq. 1: p_3D = D(u,v) * K^(-1) * [u, v, 1]^T
        2. Eq. 2: [u', v', z'] = K * R * p_3D
        """
        H, W = image.shape[:2]

        # Step 1: Back-project to 3D
        point_cloud = self._back_project(depth_map, K)

        # Step 2: Apply rotation
        R = self._rotation_matrix(np.deg2rad(angle), axis=self.config.rotation_axis)

        # Rotate all points
        points_flat = point_cloud.reshape(-1, 3)
        rotated_points = (R @ points_flat.T).T
        rotated_points = rotated_points.reshape(H, W, 3)

        # Step 3: Re-project to 2D
        new_coords = self._project(rotated_points, K)
        new_depth = rotated_points[..., 2]

        # Step 4: Warp image
        warped_image, valid_mask = self._warp_image(image, new_coords, new_depth, depth_map)

        # Step 5: Inpaint holes
        hole_ratio = 1.0 - valid_mask.mean()

        if hole_ratio > 0 and hole_ratio < self.config.max_hole_ratio:
            if self.config.inpainting_method != "none":
                inpainted_image = self._inpaint(warped_image, valid_mask)
            else:
                inpainted_image = warped_image
        else:
            inpainted_image = warped_image

        return SynthesizedView(
            image=inpainted_image,
            angle=angle,
            mask=valid_mask,
            hole_ratio=hole_ratio,
        )

    def _back_project(self, depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Back-project depth map to 3D point cloud."""
        H, W = depth_map.shape
        K_inv = np.linalg.inv(K)

        # Create pixel coordinate grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # Homogeneous coordinates
        ones = np.ones((H, W))
        pixels = np.stack([u, v, ones], axis=-1)

        # Back-project
        directions = np.einsum('ij,hwj->hwi', K_inv, pixels)
        point_cloud = directions * depth_map[..., np.newaxis]

        return point_cloud

    def _project(self, points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        # Project: pixels = K @ points_3d
        projected = np.einsum('ij,hwj->hwi', K, points_3d)

        # Perspective division
        z = projected[..., 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)

        uv = projected[..., :2] / z
        return uv

    def _rotation_matrix(self, angle: float, axis: str = "vertical") -> np.ndarray:
        """Create 3D rotation matrix."""
        c, s = np.cos(angle), np.sin(angle)

        if axis == "vertical":
            # Rotation around Y-axis (yaw)
            R = np.array([
                [c,  0, s],
                [0,  1, 0],
                [-s, 0, c]
            ])
        else:
            # Rotation around X-axis (pitch)
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
        """Warp image using computed coordinates with depth-based occlusion handling."""
        H, W = image.shape[:2]

        # Initialize output
        warped = np.zeros_like(image)
        valid_mask = np.zeros((H, W), dtype=bool)
        depth_buffer = np.full((H, W), np.inf)

        # Round coordinates
        u_new = new_coords[..., 0].astype(np.int32)
        v_new = new_coords[..., 1].astype(np.int32)

        # Valid coordinates
        valid = (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H)
        valid &= (new_depth > 0)

        # Source coordinates
        v_src, u_src = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Flatten for processing
        u_src_flat = u_src[valid]
        v_src_flat = v_src[valid]
        u_new_flat = u_new[valid]
        v_new_flat = v_new[valid]
        depth_flat = new_depth[valid]

        # Sort by depth (far to near) for z-buffer
        sort_idx = np.argsort(-depth_flat)

        for idx in sort_idx:
            u_t, v_t = u_new_flat[idx], v_new_flat[idx]
            u_s, v_s = u_src_flat[idx], v_src_flat[idx]
            d = depth_flat[idx]

            if d < depth_buffer[v_t, u_t]:
                warped[v_t, u_t] = image[v_s, u_s]
                depth_buffer[v_t, u_t] = d
                valid_mask[v_t, u_t] = True

        return warped, valid_mask

    def _inpaint(self, image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Inpaint invalid regions."""
        if self.config.inpainting_method == "opencv":
            return self._inpaint_opencv(image, valid_mask)
        return image

    def _inpaint_opencv(self, image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """OpenCV inpainting."""
        # Convert mask to uint8
        inpaint_mask = (~valid_mask).astype(np.uint8) * 255

        # Dilate mask slightly
        kernel = np.ones((3, 3), np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

        # Inpaint
        inpainted = cv2.inpaint(
            image,
            inpaint_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )

        return inpainted

    def _estimate_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Estimate camera intrinsics from image size."""
        fx = fy = max(width, height)
        cx, cy = width / 2, height / 2

        return np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ]
        ], dtype=np.float64)
