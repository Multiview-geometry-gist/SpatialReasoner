"""Depth-based novel view synthesis."""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from .config import ViewSynthesisConfig


@dataclass
class SynthesizedView:
    image: np.ndarray  # RGB (H, W, 3)
    angle: float
    mask: np.ndarray  # Valid pixels (H, W)
    hole_ratio: float

    @property
    def is_valid(self) -> bool:
        return self.hole_ratio < 0.15


class ViewSynthesizer:
    """Synthesize novel views using depth-based warping."""

    def __init__(self, config: ViewSynthesisConfig):
        self.config = config

    def synthesize(
        self,
        image: Union[Image.Image, np.ndarray],
        depth_map: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> Dict[float, SynthesizedView]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        H, W = image.shape[:2]
        if K is None:
            K = self._estimate_intrinsics(W, H)

        views = {}
        for angle in self.config.rotation_angles:
            if angle == 0:
                views[angle] = SynthesizedView(
                    image=image.copy(),
                    angle=angle,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0,
                )
            else:
                views[angle] = self._synthesize_view(image, depth_map, K, angle)
        return views

    def _synthesize_view(
        self, image: np.ndarray, depth_map: np.ndarray, K: np.ndarray, angle: float
    ) -> SynthesizedView:
        H, W = image.shape[:2]

        # Back-project to 3D
        point_cloud = self._back_project(depth_map, K)

        # Rotate
        R = self._rotation_matrix(np.deg2rad(angle))
        points_flat = point_cloud.reshape(-1, 3)
        rotated = (R @ points_flat.T).T.reshape(H, W, 3)

        # Re-project
        new_coords = self._project(rotated, K)
        new_depth = rotated[..., 2]

        # Warp
        warped, valid_mask = self._warp_image(image, new_coords, new_depth)
        hole_ratio = 1.0 - valid_mask.mean()

        # Inpaint if needed
        if 0 < hole_ratio < self.config.max_hole_ratio:
            warped = self._inpaint(warped, valid_mask)

        return SynthesizedView(
            image=warped,
            angle=angle,
            mask=valid_mask,
            hole_ratio=hole_ratio,
        )

    def _back_project(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        K_inv = np.linalg.inv(K)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels = np.stack([u, v, np.ones((H, W))], axis=-1)
        directions = np.einsum('ij,hwj->hwi', K_inv, pixels)
        return directions * depth[..., np.newaxis]

    def _project(self, points: np.ndarray, K: np.ndarray) -> np.ndarray:
        projected = np.einsum('ij,hwj->hwi', K, points)
        z = projected[..., 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return projected[..., :2] / z

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        if self.config.rotation_axis == "vertical":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _warp_image(
        self, image: np.ndarray, new_coords: np.ndarray, new_depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = image.shape[:2]
        warped = np.zeros_like(image)
        valid_mask = np.zeros((H, W), dtype=bool)
        depth_buffer = np.full((H, W), np.inf)

        u_new = new_coords[..., 0].astype(np.int32)
        v_new = new_coords[..., 1].astype(np.int32)
        valid = (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H) & (new_depth > 0)

        v_src, u_src = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        indices = np.where(valid)

        # Sort by depth (far to near)
        depths = new_depth[indices]
        order = np.argsort(-depths)

        for idx in order:
            i, j = indices[0][idx], indices[1][idx]
            u_t, v_t = u_new[i, j], v_new[i, j]
            d = new_depth[i, j]
            if d < depth_buffer[v_t, u_t]:
                warped[v_t, u_t] = image[i, j]
                depth_buffer[v_t, u_t] = d
                valid_mask[v_t, u_t] = True

        return warped, valid_mask

    def _inpaint(self, image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        inpaint_mask = (~valid_mask).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
        return cv2.inpaint(image, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def _estimate_intrinsics(self, width: int, height: int) -> np.ndarray:
        fx = fy = self.config.focal_length or max(width, height)
        return np.array([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]], dtype=np.float64)
