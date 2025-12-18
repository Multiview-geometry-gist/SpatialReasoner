"""Abstract base class for view synthesis backends.

Defines the interface that all view synthesis implementations must follow.
This enables plug-and-play backend switching without modifying pipeline code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np
from PIL import Image


@dataclass
class SynthesizedView:
    """Container for a synthesized view.

    Attributes:
        image: RGB image array (H, W, 3), uint8
        angle: Rotation angle in degrees (positive = clockwise from above)
        mask: Valid pixel mask (H, W), True = valid pixel
        hole_ratio: Ratio of invalid/hallucinated pixels (0.0 = perfect)
        backend: Name of the backend that produced this view
        quality_score: Optional quality metric (higher = better)
        metadata: Additional backend-specific metadata
    """
    image: np.ndarray
    angle: float
    mask: np.ndarray
    hole_ratio: float
    backend: str = "unknown"
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if view passes quality threshold (default: <15% holes)."""
        return self.hole_ratio < 0.15

    def passes_threshold(self, max_hole_ratio: float) -> bool:
        """Check if view passes custom hole ratio threshold."""
        return self.hole_ratio <= max_hole_ratio

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        return Image.fromarray(self.image)

    def save(self, path: str, quality: int = 95) -> None:
        """Save view to disk."""
        self.to_pil().save(path, quality=quality)


@dataclass
class SynthesisConfig:
    """Base configuration for view synthesis backends.

    Subclasses can extend this with backend-specific parameters.
    This base config contains common parameters shared across all backends.
    """
    # Backend selection
    backend: str = "depth_warping"
    enabled: bool = True
    rotation_angles: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
    rotation_axis: str = "vertical"  # "vertical" (y-axis) or "horizontal" (x-axis)
    max_hole_ratio: float = 0.15
    device: str = "cuda"

    # Quality thresholds for fallback decisions
    fallback_hole_threshold: float = 0.10  # Trigger fallback if holes > 10%

    # MVGenMaster-specific settings
    mvgenmaster_root: str = "/home/ubuntu/MVGenMaster"
    mvgenmaster_model_dir: str = "/home/ubuntu/MVGenMaster/check_points/pretrained_model"
    mvgenmaster_num_frames: int = 28
    mvgenmaster_guidance_scale: float = 2.0

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for compatibility."""
        return getattr(self, key, default)


class ViewSynthesizerBase(ABC):
    """Abstract base class for all view synthesis backends.

    All view synthesis implementations must inherit from this class and
    implement the required abstract methods. This ensures consistent
    interface across different backends (depth-warping, ZeroNVS, etc.).

    Example:
        class MyCustomSynthesizer(ViewSynthesizerBase):
            backend_name = "my_custom"

            def synthesize_single(self, image, depth_map, K, angle):
                # Custom implementation
                ...
    """

    # Class attribute - subclasses must override
    backend_name: str = "base"

    def __init__(self, config: SynthesisConfig):
        """Initialize the synthesizer with configuration.

        Args:
            config: SynthesisConfig or compatible configuration object
        """
        self.config = config
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization hook - called before first synthesis.

        Subclasses can override to load models on demand.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Perform actual initialization. Override in subclasses."""
        pass

    def synthesize(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> Dict[float, SynthesizedView]:
        """Synthesize novel views at all configured rotation angles.

        This is the main entry point for view synthesis. It handles:
        - Input validation and preprocessing
        - Calling synthesize_single for each angle
        - Post-processing and quality checks

        Args:
            image: Original RGB image (H, W, 3) as numpy array or PIL Image
            depth_map: Optional depth map (H, W). Required by some backends.
            K: Optional camera intrinsic matrix (3, 3). Estimated if not provided.

        Returns:
            Dictionary mapping angle (float) -> SynthesizedView
        """
        self._ensure_initialized()

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        H, W = image.shape[:2]

        # Get rotation angles from config
        rotation_angles = self._get_rotation_angles()

        # Estimate intrinsics if not provided
        if K is None:
            K = self._estimate_intrinsics(W, H)

        # Generate views for each angle
        views = {}
        for angle in rotation_angles:
            if angle == 0:
                # Original view - no synthesis needed
                views[angle] = SynthesizedView(
                    image=image.copy(),
                    angle=angle,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0,
                    backend=self.backend_name,
                    quality_score=1.0,
                )
            else:
                # Synthesize rotated view
                views[angle] = self.synthesize_single(
                    image=image,
                    depth_map=depth_map,
                    K=K,
                    angle=angle,
                )

        return views

    @abstractmethod
    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize a single novel view at the specified angle.

        This is the core method that subclasses must implement.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Depth map (H, W), may be None for diffusion backends
            K: Camera intrinsic matrix (3, 3)
            angle: Rotation angle in degrees

        Returns:
            SynthesizedView containing the rotated view
        """
        raise NotImplementedError

    def _get_rotation_angles(self) -> List[float]:
        """Get rotation angles from config."""
        if hasattr(self.config, 'rotation_angles'):
            return self.config.rotation_angles
        return [-5.0, 0.0, 5.0]

    def _estimate_intrinsics(self, width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
        """Estimate camera intrinsics from image size.

        Uses standard pinhole camera model with assumed horizontal FOV.

        Args:
            width, height: Image dimensions
            fov_deg: Assumed horizontal field of view in degrees

        Returns:
            Camera intrinsic matrix K (3, 3)
        """
        fov_rad = np.radians(fov_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx, cy = width / 2.0, height / 2.0

        return np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ]
        ], dtype=np.float64)

    def get_capabilities(self) -> Dict[str, bool]:
        """Return backend capabilities for feature discovery.

        Returns:
            Dictionary of capability flags
        """
        return {
            "requires_depth": True,
            "can_hallucinate": False,
            "supports_arbitrary_angles": True,
            "gpu_required": False,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend_name})"
