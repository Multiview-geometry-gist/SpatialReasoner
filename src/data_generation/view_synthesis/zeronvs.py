"""ZeroNVS backend for high-quality scene-level novel view synthesis.

ZeroNVS is a diffusion-based model trained on scene-level data (RealEstate10K).
It can hallucinate plausible content for disoccluded regions.

Paper: "ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image"
GitHub: https://github.com/kylesargent/zeronvs

Requirements:
    pip install diffusers transformers accelerate
    # ZeroNVS specific dependencies

Note:
    This backend requires GPU with ~24GB VRAM for inference.
    Inference is slower than depth-warping but produces higher quality results.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging

from .base import ViewSynthesizerBase, SynthesizedView, SynthesisConfig
from .registry import ViewSynthesizerRegistry

logger = logging.getLogger(__name__)


@ViewSynthesizerRegistry.register(
    "zeronvs",
    priority=50,
    description="Diffusion-based scene-level view synthesis",
    requirements=["diffusers", "torch"],
)
class ZeroNVSSynthesizer(ViewSynthesizerBase):
    """ZeroNVS diffusion model for novel view synthesis.

    Uses a pre-trained diffusion model to synthesize novel views.
    Can hallucinate plausible content for regions not visible in the input.

    Configuration:
        - model_name: HuggingFace model identifier
        - num_inference_steps: Diffusion steps (default 50)
        - guidance_scale: Classifier-free guidance scale (default 3.0)
        - device: "cuda" or "cpu"
    """

    backend_name = "zeronvs"

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)

        # ZeroNVS specific config
        self._model_name = getattr(
            config, 'zeronvs_model',
            "stabilityai/stable-zero123"  # Fallback to publicly available model
        )
        self._num_inference_steps = getattr(config, 'num_inference_steps', 50)
        self._guidance_scale = getattr(config, 'guidance_scale', 3.0)
        self._device = getattr(config, 'device', 'cuda')

        # Model will be loaded lazily
        self._pipeline = None

    def _initialize(self) -> None:
        """Load the ZeroNVS/Zero123 pipeline."""
        try:
            import torch
            from diffusers import StableZero123Pipeline

            logger.info(f"Loading ZeroNVS model: {self._model_name}")

            self._pipeline = StableZero123Pipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
            )
            self._pipeline.to(self._device)

            # Enable memory optimizations
            if hasattr(self._pipeline, 'enable_attention_slicing'):
                self._pipeline.enable_attention_slicing()

            logger.info("ZeroNVS model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"ZeroNVS requires diffusers library. Install with: "
                f"pip install diffusers transformers accelerate. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ZeroNVS model: {e}")

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize a novel view using ZeroNVS diffusion model.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Optional depth map (not required for diffusion)
            K: Camera intrinsic matrix (used for conditioning)
            angle: Rotation angle in degrees

        Returns:
            SynthesizedView with synthesized image
        """
        import torch
        from PIL import Image as PILImage

        H, W = image.shape[:2]

        # Convert to PIL for diffusers
        input_image = PILImage.fromarray(image)

        # Convert angle to camera pose change
        # ZeroNVS/Zero123 uses elevation and azimuth
        elevation = 0.0  # Keep same elevation
        azimuth = angle  # Use rotation angle as azimuth change

        # Run diffusion
        with torch.no_grad():
            output = self._pipeline(
                input_image,
                elevation=elevation,
                azimuth=azimuth,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
            )

        # Extract generated image
        generated_image = output.images[0]

        # Resize to original dimensions if needed
        if generated_image.size != (W, H):
            generated_image = generated_image.resize((W, H), PILImage.LANCZOS)

        output_array = np.array(generated_image)

        # Diffusion models produce full images without holes
        valid_mask = np.ones((H, W), dtype=bool)

        return SynthesizedView(
            image=output_array,
            angle=angle,
            mask=valid_mask,
            hole_ratio=0.0,  # No holes with diffusion
            backend=self.backend_name,
            quality_score=0.9,  # High quality from diffusion
            metadata={
                "elevation": elevation,
                "azimuth": azimuth,
                "num_inference_steps": self._num_inference_steps,
                "guidance_scale": self._guidance_scale,
            }
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities."""
        return {
            "requires_depth": False,
            "can_hallucinate": True,
            "supports_arbitrary_angles": True,
            "gpu_required": True,
            "vram_requirement_gb": 24,
            "inference_time_seconds": 10,  # Approximate
        }


class ZeroNVSLiteSynthesizer(ZeroNVSSynthesizer):
    """Lightweight version of ZeroNVS with reduced memory footprint.

    Uses fp16, attention slicing, and optionally sequential CPU offload
    to run on GPUs with less VRAM (~12GB).
    """

    backend_name = "zeronvs_lite"

    def _initialize(self) -> None:
        """Load model with memory optimizations."""
        try:
            import torch
            from diffusers import StableZero123Pipeline

            logger.info(f"Loading ZeroNVS-Lite model: {self._model_name}")

            self._pipeline = StableZero123Pipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            # Aggressive memory optimizations
            self._pipeline.enable_attention_slicing(1)

            # Use sequential CPU offload for lower VRAM
            if hasattr(self._pipeline, 'enable_sequential_cpu_offload'):
                self._pipeline.enable_sequential_cpu_offload()
            else:
                self._pipeline.to(self._device)

            logger.info("ZeroNVS-Lite model loaded with memory optimizations")

        except ImportError as e:
            raise ImportError(f"ZeroNVS requires diffusers library: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            "vram_requirement_gb": 12,
            "inference_time_seconds": 15,  # Slower due to offloading
        })
        return capabilities


# Register the lite version
ViewSynthesizerRegistry.register(
    "zeronvs_lite",
    priority=55,
    description="Memory-efficient ZeroNVS for GPUs with 12GB+ VRAM"
)(ZeroNVSLiteSynthesizer)
