"""
Multi-View Spatial Reasoner Pipeline

Inference-time pipeline that generates novel views from a single image
and uses them for improved spatial reasoning.

Pipeline:
    1. Single Image → Depth Estimation (Depth Anything V2)
    2. Single Image → Novel Views (Zero123++)
    3. [Original, Novel View, Depth] → Spatial VLM → Prediction
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiViewInput:
    """Container for multi-view input to VLM."""
    original: Image.Image
    novel_views: List[Image.Image]
    depth_map: Optional[np.ndarray] = None
    view_angles: List[float] = None

    def to_vlm_images(self, num_views: int = 2) -> List[Image.Image]:
        """Get images formatted for VLM input."""
        images = [self.original]
        images.extend(self.novel_views[:num_views-1])
        return images


class MultiViewSpatialReasoner:
    """
    Pipeline for spatial reasoning using synthesized multi-view images.

    This class combines:
    - Depth estimation (Depth Anything V2)
    - Novel view synthesis (Zero123++)
    - Optional: Spatial reasoning VLM

    Usage:
        reasoner = MultiViewSpatialReasoner(device='cuda:0')
        result = reasoner.generate_multiview(image)
        # result.original, result.novel_views, result.depth_map
    """

    # Zero123++ fixed output azimuths (relative to input view)
    ZERO123_AZIMUTHS = [30, 90, 150, 210, 270, 330]

    def __init__(
        self,
        device: str = 'cuda:0',
        num_inference_steps: int = 30,
        enable_depth: bool = True,
        selected_angles: List[int] = None,
    ):
        """
        Initialize the multi-view spatial reasoner.

        Args:
            device: CUDA device to use
            num_inference_steps: Diffusion steps for Zero123++
            enable_depth: Whether to estimate depth
            selected_angles: Which Zero123++ angles to use (default: [30, 330])
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.enable_depth = enable_depth
        self.selected_angles = selected_angles or [30, 330]  # ±30° views

        self._depth_estimator = None
        self._view_synthesizer = None

    def _load_depth_estimator(self):
        """Lazy load depth estimation model."""
        if self._depth_estimator is None:
            logger.info("Loading Depth Anything V2...")
            from transformers import pipeline
            self._depth_estimator = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device
            )
            logger.info("Depth estimator loaded")

    def _load_view_synthesizer(self):
        """Lazy load Zero123++ model."""
        if self._view_synthesizer is None:
            logger.info("Loading Zero123++ pipeline...")
            from diffusers import DiffusionPipeline

            self._view_synthesizer = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1",
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
            )
            self._view_synthesizer.to(self.device)
            logger.info("Zero123++ loaded")

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image: Input PIL Image

        Returns:
            Depth map as numpy array (H, W), normalized to [0, 1]
        """
        self._load_depth_estimator()

        result = self._depth_estimator(image)
        depth = np.array(result['depth'].resize(image.size))
        depth = depth.astype(np.float32) / 255.0

        return depth

    def synthesize_views(self, image: Image.Image) -> Tuple[List[Image.Image], List[int]]:
        """
        Generate novel views using Zero123++.

        Args:
            image: Input PIL Image

        Returns:
            Tuple of (list of view images, list of azimuth angles)
        """
        self._load_view_synthesizer()

        # Ensure RGBA for Zero123++
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Generate 6 views in a grid
        with torch.no_grad():
            result = self._view_synthesizer(
                image,
                num_inference_steps=self.num_inference_steps,
            )

        # Extract individual views from grid
        grid_image = result.images[0]
        views = self._extract_views_from_grid(grid_image)

        # Select requested views
        selected_views = []
        selected_angles = []

        for angle in self.selected_angles:
            if angle in self.ZERO123_AZIMUTHS:
                idx = self.ZERO123_AZIMUTHS.index(angle)
                selected_views.append(views[idx])
                selected_angles.append(angle)

        return selected_views, selected_angles

    def _extract_views_from_grid(self, grid_image: Image.Image) -> List[Image.Image]:
        """Extract 6 individual views from Zero123++ grid output."""
        W, H = grid_image.size

        # Grid is 2 columns x 3 rows
        view_w = W // 2
        view_h = H // 3

        views = []
        for row in range(3):
            for col in range(2):
                left = col * view_w
                upper = row * view_h
                right = left + view_w
                lower = upper + view_h
                view = grid_image.crop((left, upper, right, lower))
                views.append(view)

        return views

    def generate_multiview(
        self,
        image: Image.Image,
        include_depth: bool = None,
    ) -> MultiViewInput:
        """
        Generate multi-view input from a single image.

        Args:
            image: Input PIL Image
            include_depth: Whether to include depth map (default: self.enable_depth)

        Returns:
            MultiViewInput containing original, novel views, and optionally depth
        """
        if include_depth is None:
            include_depth = self.enable_depth

        # Synthesize novel views
        novel_views, angles = self.synthesize_views(image)

        # Estimate depth if requested
        depth_map = None
        if include_depth:
            depth_map = self.estimate_depth(image)

        return MultiViewInput(
            original=image,
            novel_views=novel_views,
            depth_map=depth_map,
            view_angles=angles,
        )

    def prepare_vlm_input(
        self,
        multiview: MultiViewInput,
        resize_to: Tuple[int, int] = None,
    ) -> Dict[str, Any]:
        """
        Prepare input for VLM spatial reasoning.

        Args:
            multiview: MultiViewInput from generate_multiview()
            resize_to: Optional resize dimensions (width, height)

        Returns:
            Dictionary with 'images' and 'depth' keys ready for VLM
        """
        images = multiview.to_vlm_images()

        if resize_to:
            images = [img.resize(resize_to) for img in images]

        return {
            'images': images,
            'depth': multiview.depth_map,
            'view_angles': [0] + multiview.view_angles,  # 0 for original
        }


# Convenience function for quick inference
def generate_multiview_for_reasoning(
    image_path: str,
    output_dir: str = None,
    device: str = 'cuda:0',
) -> MultiViewInput:
    """
    Quick function to generate multi-view from an image file.

    Args:
        image_path: Path to input image
        output_dir: Optional directory to save results
        device: CUDA device

    Returns:
        MultiViewInput with original, novel views, and depth
    """
    import os

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Create reasoner and generate
    reasoner = MultiViewSpatialReasoner(device=device)
    result = reasoner.generate_multiview(image)

    # Save if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        result.original.save(os.path.join(output_dir, 'original.png'))

        for i, (view, angle) in enumerate(zip(result.novel_views, result.view_angles)):
            view.save(os.path.join(output_dir, f'view_{angle}deg.png'))

        if result.depth_map is not None:
            depth_img = Image.fromarray((result.depth_map * 255).astype(np.uint8))
            depth_img.save(os.path.join(output_dir, 'depth.png'))

        logger.info(f"Saved results to {output_dir}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multiview_spatial_reasoner.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./multiview_output"

    logging.basicConfig(level=logging.INFO)

    result = generate_multiview_for_reasoning(image_path, output_dir)
    print(f"Generated {len(result.novel_views)} novel views")
    print(f"View angles: {result.view_angles}")
    print(f"Depth map shape: {result.depth_map.shape if result.depth_map is not None else 'N/A'}")
