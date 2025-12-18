"""Hybrid view synthesizer with quality-based fallback.

This orchestrator tries fast backends first and falls back to higher-quality
(but slower) backends if quality thresholds are not met.

Fallback Strategy:
1. Try depth_warping (fastest, no GPU required)
2. If hole_ratio > threshold, try zeronvs (scene-level)
3. If still failing, try zero123++ (object-centric)

Configuration allows customizing:
- Which backends to try
- Quality thresholds for fallback
- Parallel vs sequential execution
"""

import numpy as np
from typing import Optional, Dict, Any, List
import logging

from .base import ViewSynthesizerBase, SynthesizedView, SynthesisConfig
from .registry import ViewSynthesizerRegistry, create_view_synthesizer

logger = logging.getLogger(__name__)


@ViewSynthesizerRegistry.register(
    "hybrid",
    priority=10,  # Low priority = preferred when using auto-select
    description="Quality-based fallback across multiple backends",
)
class HybridSynthesizer(ViewSynthesizerBase):
    """Hybrid synthesizer with intelligent backend selection.

    Tries backends in order of speed and falls back to higher-quality
    backends if the result doesn't meet quality thresholds.

    Configuration:
        - fallback_backends: List of backends to try in order
        - hole_threshold: Trigger fallback if hole_ratio exceeds this
        - quality_threshold: Minimum quality score required
        - enable_fallback: Whether to use fallback (False = use first backend only)
    """

    backend_name = "hybrid"

    # Default fallback order (fast to slow)
    DEFAULT_FALLBACK_ORDER = ["depth_warping", "zeronvs", "zero123pp"]

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)

        # Get fallback settings from config
        self._fallback_backends = getattr(
            config, 'fallback_backends',
            self.DEFAULT_FALLBACK_ORDER
        )
        self._hole_threshold = getattr(config, 'fallback_hole_threshold', 0.10)
        self._quality_threshold = getattr(config, 'quality_threshold', 0.8)
        self._enable_fallback = getattr(config, 'enable_fallback', True)

        # Cache instantiated backends
        self._backend_instances: Dict[str, ViewSynthesizerBase] = {}

    def _initialize(self) -> None:
        """Initialize the primary backend."""
        if self._fallback_backends:
            primary = self._fallback_backends[0]
            self._get_or_create_backend(primary)

    def _get_or_create_backend(self, name: str) -> Optional[ViewSynthesizerBase]:
        """Get or create a backend instance.

        Uses lazy initialization to avoid loading unused models.

        Args:
            name: Backend name

        Returns:
            Backend instance or None if unavailable
        """
        if name in self._backend_instances:
            return self._backend_instances[name]

        try:
            backend = ViewSynthesizerRegistry.create(name, self.config)
            self._backend_instances[name] = backend
            return backend
        except (KeyError, ImportError, RuntimeError) as e:
            logger.warning(f"Backend '{name}' unavailable: {e}")
            return None

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize with fallback mechanism.

        Tries backends in order until quality threshold is met.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Optional depth map
            K: Camera intrinsic matrix
            angle: Rotation angle in degrees

        Returns:
            Best SynthesizedView across tried backends
        """
        best_result: Optional[SynthesizedView] = None
        tried_backends: List[str] = []

        for backend_name in self._fallback_backends:
            backend = self._get_or_create_backend(backend_name)

            if backend is None:
                continue

            # Check if backend requires depth and we don't have it
            capabilities = backend.get_capabilities()
            if capabilities.get("requires_depth", False) and depth_map is None:
                logger.debug(
                    f"Skipping {backend_name}: requires depth map"
                )
                continue

            try:
                logger.debug(f"Trying backend: {backend_name}")
                result = backend.synthesize_single(image, depth_map, K, angle)
                tried_backends.append(backend_name)

                # Track best result
                if best_result is None or self._is_better(result, best_result):
                    best_result = result

                # Check if quality is sufficient
                if self._meets_threshold(result):
                    logger.debug(
                        f"Backend {backend_name} succeeded with "
                        f"hole_ratio={result.hole_ratio:.3f}"
                    )
                    break

                # Fallback not enabled - use first result
                if not self._enable_fallback:
                    break

                logger.debug(
                    f"Backend {backend_name} quality insufficient "
                    f"(hole_ratio={result.hole_ratio:.3f}), trying fallback"
                )

            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                continue

        if best_result is None:
            # All backends failed - return empty view
            H, W = image.shape[:2]
            logger.error("All backends failed, returning empty view")
            return SynthesizedView(
                image=np.zeros_like(image),
                angle=angle,
                mask=np.zeros((H, W), dtype=bool),
                hole_ratio=1.0,
                backend="none",
                metadata={"tried_backends": tried_backends, "all_failed": True}
            )

        # Add metadata about fallback attempts
        best_result.metadata["tried_backends"] = tried_backends

        return best_result

    def _meets_threshold(self, view: SynthesizedView) -> bool:
        """Check if view meets quality thresholds.

        Args:
            view: Synthesized view to evaluate

        Returns:
            True if view meets all quality thresholds
        """
        # Check hole ratio
        if view.hole_ratio > self._hole_threshold:
            return False

        # Check quality score if available
        if view.quality_score is not None:
            if view.quality_score < self._quality_threshold:
                return False

        return True

    def _is_better(
        self,
        candidate: SynthesizedView,
        current_best: SynthesizedView
    ) -> bool:
        """Compare two views to determine which is better.

        Args:
            candidate: New candidate view
            current_best: Current best view

        Returns:
            True if candidate is better than current_best
        """
        # Prefer lower hole ratio
        if candidate.hole_ratio < current_best.hole_ratio:
            return True

        # If hole ratios are similar, prefer higher quality score
        if abs(candidate.hole_ratio - current_best.hole_ratio) < 0.01:
            if (candidate.quality_score or 0) > (current_best.quality_score or 0):
                return True

        return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Return combined capabilities of all available backends."""
        capabilities = {
            "requires_depth": False,  # Can work without if diffusion backend available
            "can_hallucinate": True,  # Via diffusion fallback
            "supports_arbitrary_angles": True,
            "gpu_required": False,  # depth_warping doesn't need GPU
            "fallback_backends": self._fallback_backends,
            "available_backends": [],
        }

        # Check which backends are actually available
        for name in self._fallback_backends:
            if ViewSynthesizerRegistry.is_available(name):
                capabilities["available_backends"].append(name)

        return capabilities


class AdaptiveSynthesizer(HybridSynthesizer):
    """Adaptive synthesizer that learns which backend works best.

    Extends HybridSynthesizer with statistics tracking to optimize
    backend selection over time.
    """

    backend_name = "adaptive"

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)

        # Track success rates per backend
        self._backend_stats: Dict[str, Dict[str, float]] = {}
        for name in self._fallback_backends:
            self._backend_stats[name] = {
                "attempts": 0,
                "successes": 0,
                "avg_hole_ratio": 0.0,
                "avg_time": 0.0,
            }

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize with adaptive backend selection."""
        import time

        result = None

        for backend_name in self._get_sorted_backends():
            backend = self._get_or_create_backend(backend_name)
            if backend is None:
                continue

            start_time = time.time()

            try:
                result = backend.synthesize_single(image, depth_map, K, angle)
                elapsed = time.time() - start_time

                # Update statistics
                self._update_stats(backend_name, result, elapsed, success=True)

                if self._meets_threshold(result):
                    break

            except Exception as e:
                elapsed = time.time() - start_time
                self._update_stats(backend_name, None, elapsed, success=False)
                logger.warning(f"Backend {backend_name} failed: {e}")

        return result or self._create_fallback_view(image, angle)

    def _get_sorted_backends(self) -> List[str]:
        """Sort backends by success rate and speed."""
        def score(name: str) -> float:
            stats = self._backend_stats.get(name, {})
            attempts = stats.get("attempts", 0)
            if attempts == 0:
                return 0.5  # Unknown - try it

            success_rate = stats.get("successes", 0) / attempts
            avg_hole = stats.get("avg_hole_ratio", 0.5)

            # Higher score = try first
            return success_rate * (1 - avg_hole)

        return sorted(self._fallback_backends, key=score, reverse=True)

    def _update_stats(
        self,
        backend_name: str,
        result: Optional[SynthesizedView],
        elapsed: float,
        success: bool
    ) -> None:
        """Update backend statistics."""
        stats = self._backend_stats[backend_name]
        stats["attempts"] += 1

        if success and result:
            stats["successes"] += 1
            # Running average for hole ratio
            n = stats["attempts"]
            stats["avg_hole_ratio"] = (
                stats["avg_hole_ratio"] * (n - 1) + result.hole_ratio
            ) / n

        # Running average for time
        n = stats["attempts"]
        stats["avg_time"] = (stats["avg_time"] * (n - 1) + elapsed) / n

    def _create_fallback_view(
        self,
        image: np.ndarray,
        angle: float
    ) -> SynthesizedView:
        """Create fallback view when all backends fail."""
        H, W = image.shape[:2]
        return SynthesizedView(
            image=np.zeros_like(image),
            angle=angle,
            mask=np.zeros((H, W), dtype=bool),
            hole_ratio=1.0,
            backend="fallback",
        )

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get backend performance statistics."""
        return dict(self._backend_stats)


# Register adaptive synthesizer
ViewSynthesizerRegistry.register(
    "adaptive",
    priority=5,
    description="Self-optimizing backend selection",
)(AdaptiveSynthesizer)
