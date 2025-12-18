"""Zero123++ backend for object-centric novel view synthesis.

Zero123++ is specifically designed for object-centric images with clean backgrounds.
It generates multiple consistent views of objects in a single forward pass.

Paper: "Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model"
GitHub: https://github.com/SUDO-AI-3D/zero123plus

Best for:
    - Product images with clean backgrounds
    - Single object images
    - When 6 views are needed simultaneously

Not ideal for:
    - Complex scene-level images
    - Images with multiple objects
    - Natural photographs with busy backgrounds
"""

import numpy as np
from typing import Optional, Dict, Any, List
import logging

from .base import ViewSynthesizerBase, SynthesizedView, SynthesisConfig
from .registry import ViewSynthesizerRegistry

logger = logging.getLogger(__name__)


@ViewSynthesizerRegistry.register(
    "zero123pp",
    priority=60,
    description="Object-centric multi-view synthesis",
    requirements=["diffusers", "torch"],
)
class Zero123Synthesizer(ViewSynthesizerBase):
    """Zero123++ for object-centric novel view synthesis.

    Generates 6 consistent views of an object in a single forward pass.
    The views are at fixed elevations and azimuths.

    Output views (Zero123++ default):
        - Front (0 deg), Front-Right (30 deg), Right (90 deg)
        - Back (180 deg), Left (270 deg), Front-Left (330 deg)

    Configuration:
        - model_name: HuggingFace model path (default: "sudo-ai/zero123plus-v1.2")
        - num_inference_steps: Diffusion steps (default 75)
        - device: "cuda" or "cpu"
    """

    backend_name = "zero123pp"

    # Fixed output views from Zero123++
    FIXED_AZIMUTHS = [0, 30, 90, 180, 270, 330]

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)

        self._model_name = getattr(
            config, 'zero123_model',
            "sudo-ai/zero123plus-v1.2"
        )
        self._num_inference_steps = getattr(config, 'num_inference_steps', 75)
        self._device = getattr(config, 'device', 'cuda')

        self._pipeline = None
        self._cached_views = {}  # Cache multi-view outputs

    def _initialize(self) -> None:
        """Load the Zero123++ pipeline."""
        try:
            import torch
            from diffusers import DiffusionPipeline

            logger.info(f"Loading Zero123++ model: {self._model_name}")

            # Use v1.1 with custom pipeline for best compatibility
            self._pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1",
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
            )
            self._pipeline.to(self._device)

            logger.info("Zero123++ model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Zero123++ requires diffusers. Install with: "
                f"pip install diffusers transformers. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Zero123++ model: {e}")

    def synthesize(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> Dict[float, SynthesizedView]:
        """Override synthesize to generate all views at once.

        Zero123++ generates 6 views in a single forward pass, so we
        generate all views together and then extract the requested angles.
        """
        self._ensure_initialized()

        from PIL import Image as PILImage

        H, W = image.shape[:2]

        # Generate all 6 views at once
        input_image = PILImage.fromarray(image)
        all_views = self._generate_multiview(input_image)

        # Get requested rotation angles
        rotation_angles = self._get_rotation_angles()

        # Map requested angles to closest available views
        views = {}
        for angle in rotation_angles:
            if angle == 0:
                # Original view
                views[angle] = SynthesizedView(
                    image=image.copy(),
                    angle=angle,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0,
                    backend=self.backend_name,
                )
            else:
                # Find closest generated view
                closest_idx = self._find_closest_view(angle)
                generated = all_views[closest_idx]

                # Resize to original dimensions
                if generated.size != (W, H):
                    generated = generated.resize((W, H), PILImage.LANCZOS)

                views[angle] = SynthesizedView(
                    image=np.array(generated),
                    angle=angle,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0,
                    backend=self.backend_name,
                    quality_score=0.85,
                    metadata={
                        "actual_azimuth": self.FIXED_AZIMUTHS[closest_idx],
                        "requested_angle": angle,
                    }
                )

        return views

    def _generate_multiview(self, image) -> List:
        """Generate 6 views using Zero123++.

        Args:
            image: PIL Image input (will be converted to RGBA)

        Returns:
            List of 6 PIL Images extracted from the 3x2 grid output
        """
        import torch
        from PIL import Image as PILImage

        # Ensure RGBA format for Zero123++
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        with torch.no_grad():
            result = self._pipeline(
                image,
                num_inference_steps=self._num_inference_steps,
            )

        # Zero123++ returns a single image with 6 views in a 3x2 grid
        # Output size is typically 640x960 (each view is ~320x320)
        grid_image = result.images[0]
        W, H = grid_image.size

        # Calculate single view dimensions (3 columns, 2 rows)
        view_w = W // 2
        view_h = H // 3

        # Extract 6 views from the grid
        # Layout: top-left, top-right, mid-left, mid-right, bot-left, bot-right
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

    def _find_closest_view(self, angle: float) -> int:
        """Find the index of the closest fixed view to the requested angle.

        Args:
            angle: Requested rotation angle in degrees

        Returns:
            Index into FIXED_AZIMUTHS
        """
        # Normalize angle to [0, 360)
        normalized = angle % 360

        # Find closest azimuth
        min_diff = float('inf')
        closest_idx = 0

        for i, azimuth in enumerate(self.FIXED_AZIMUTHS):
            diff = min(
                abs(normalized - azimuth),
                abs(normalized - azimuth + 360),
                abs(normalized - azimuth - 360)
            )
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        return closest_idx

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize single view - calls full pipeline for efficiency.

        Note: Zero123++ generates 6 views at once, so calling this
        repeatedly is inefficient. Use synthesize() instead.
        """
        from PIL import Image as PILImage

        H, W = image.shape[:2]
        input_image = PILImage.fromarray(image)

        # Generate all views
        all_views = self._generate_multiview(input_image)

        # Get closest view
        closest_idx = self._find_closest_view(angle)
        generated = all_views[closest_idx]

        if generated.size != (W, H):
            generated = generated.resize((W, H), PILImage.LANCZOS)

        return SynthesizedView(
            image=np.array(generated),
            angle=angle,
            mask=np.ones((H, W), dtype=bool),
            hole_ratio=0.0,
            backend=self.backend_name,
            metadata={"actual_azimuth": self.FIXED_AZIMUTHS[closest_idx]}
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities."""
        return {
            "requires_depth": False,
            "can_hallucinate": True,
            "supports_arbitrary_angles": False,  # Only fixed angles
            "fixed_azimuths": self.FIXED_AZIMUTHS,
            "gpu_required": True,
            "vram_requirement_gb": 16,
            "batch_size": 6,  # Generates 6 views at once
            "best_for": "object_centric",
        }
