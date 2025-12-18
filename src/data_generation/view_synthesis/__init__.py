"""Novel View Synthesis Module with Pluggable Backends.

This module provides a modular architecture for novel view synthesis with:
- Abstract base class defining the synthesis interface
- Multiple backend implementations (depth-warping, ZeroNVS, Zero123++)
- Quality-based fallback mechanism
- Configuration-driven backend selection

Architecture:
    ViewSynthesizerBase (ABC)
        |
        +-- DepthWarpingSynthesizer (fast, CPU-friendly)
        +-- ZeroNVSSynthesizer (high-quality, scene-level)
        +-- Zero123Synthesizer (object-centric)
        +-- HybridSynthesizer (quality-based fallback)

Usage:
    from data_generation.view_synthesis import create_view_synthesizer

    # Simple factory usage
    synthesizer = create_view_synthesizer(config)
    views = synthesizer.synthesize(image, depth_map)

    # Direct backend usage
    from data_generation.view_synthesis import DepthWarpingSynthesizer
    synthesizer = DepthWarpingSynthesizer(config)

Backward Compatibility:
    The original ViewSynthesizer class is aliased to DepthWarpingSynthesizer.
    Existing code using ViewSynthesizer will continue to work unchanged.
"""

import logging

logger = logging.getLogger(__name__)

# Core imports - always available
from .base import ViewSynthesizerBase, SynthesizedView
from .registry import ViewSynthesizerRegistry, create_view_synthesizer
from .depth_warping import DepthWarpingSynthesizer

# Re-export for backward compatibility with existing code
ViewSynthesizer = DepthWarpingSynthesizer

__all__ = [
    # Core classes
    "ViewSynthesizerBase",
    "SynthesizedView",
    # Registry and factory
    "ViewSynthesizerRegistry",
    "create_view_synthesizer",
    # Backend implementations
    "DepthWarpingSynthesizer",
    # Backward compatibility alias
    "ViewSynthesizer",
]


def __getattr__(name: str):
    """Lazy loading for optional backend implementations.

    This allows importing backends only when they are actually used,
    avoiding ImportError for missing optional dependencies.
    """
    if name == "ZeroNVSSynthesizer":
        try:
            from .zeronvs import ZeroNVSSynthesizer
            return ZeroNVSSynthesizer
        except ImportError as e:
            raise ImportError(
                f"ZeroNVSSynthesizer requires diffusers. "
                f"Install with: pip install diffusers transformers. Error: {e}"
            )
    elif name == "Zero123Synthesizer":
        try:
            from .zero123 import Zero123Synthesizer
            return Zero123Synthesizer
        except ImportError as e:
            raise ImportError(
                f"Zero123Synthesizer requires diffusers. "
                f"Install with: pip install diffusers transformers. Error: {e}"
            )
    elif name == "HybridSynthesizer":
        from .hybrid import HybridSynthesizer
        return HybridSynthesizer
    elif name == "AdaptiveSynthesizer":
        from .hybrid import AdaptiveSynthesizer
        return AdaptiveSynthesizer
    elif name == "MVGenMasterSynthesizer":
        try:
            from .mvgenmaster import MVGenMasterSynthesizer
            return MVGenMasterSynthesizer
        except ImportError as e:
            raise ImportError(
                f"MVGenMasterSynthesizer requires MVGenMaster installation. "
                f"Ensure MVGenMaster is cloned to /home/ubuntu/MVGenMaster. Error: {e}"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def list_available_backends() -> list:
    """List all registered and available backends.

    Returns:
        List of backend names that can be used
    """
    return ViewSynthesizerRegistry.list_backends()


def get_backend_info(name: str) -> dict:
    """Get information about a specific backend.

    Args:
        name: Backend name

    Returns:
        Dictionary with backend metadata and capabilities
    """
    metadata = ViewSynthesizerRegistry.get_metadata(name)
    backend_cls = ViewSynthesizerRegistry.get(name)

    if backend_cls is None:
        return {"error": f"Backend '{name}' not found"}

    # Get capabilities if possible
    try:
        # Create temporary instance to get capabilities
        from .base import SynthesisConfig
        temp_config = SynthesisConfig()
        temp_instance = backend_cls(temp_config)
        capabilities = temp_instance.get_capabilities()
    except Exception:
        capabilities = {}

    return {
        "name": name,
        "class": backend_cls.__name__,
        "capabilities": capabilities,
        **metadata,
    }
