"""Extended configuration for multi-backend view synthesis.

This module extends the base ViewSynthesisConfig with backend-specific
options and validation.

Configuration Hierarchy:
    view_synthesis:
        backend: "hybrid"           # Backend selector
        rotation_angles: [-5, 0, 5]
        max_hole_ratio: 0.15

        # Backend-specific sections
        depth_warping:
            inpainting_method: "opencv"

        zeronvs:
            model_name: "stabilityai/stable-zero123"
            num_inference_steps: 50

        zero123:
            model_name: "sudo-ai/zero123plus-v1.2"

        hybrid:
            fallback_backends: ["depth_warping", "zeronvs"]
            fallback_hole_threshold: 0.10
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class DepthWarpingConfig:
    """Configuration for depth-warping backend."""
    inpainting_method: str = "opencv"  # "opencv", "none"
    rotation_axis: str = "vertical"


@dataclass
class ZeroNVSConfig:
    """Configuration for ZeroNVS backend."""
    model_name: str = "stabilityai/stable-zero123"
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    use_lite: bool = False  # Use memory-optimized version


@dataclass
class Zero123Config:
    """Configuration for Zero123++ backend."""
    model_name: str = "sudo-ai/zero123plus-v1.2"
    num_inference_steps: int = 75


@dataclass
class HybridConfig:
    """Configuration for hybrid fallback system."""
    fallback_backends: List[str] = field(
        default_factory=lambda: ["depth_warping", "zeronvs"]
    )
    fallback_hole_threshold: float = 0.10
    quality_threshold: float = 0.8
    enable_fallback: bool = True


@dataclass
class ViewSynthesisConfigV2:
    """Extended configuration for novel view synthesis with multi-backend support.

    This configuration supports multiple backends with backend-specific options.
    It is backward-compatible with the original ViewSynthesisConfig.

    Attributes:
        enabled: Whether view synthesis is enabled
        backend: Primary backend ("depth_warping", "zeronvs", "zero123pp", "hybrid")
        rotation_angles: List of rotation angles in degrees
        rotation_axis: Axis of rotation ("vertical" or "horizontal")
        max_hole_ratio: Maximum acceptable hole ratio
        device: Compute device ("cuda", "cpu")

        depth_warping: Depth-warping backend config
        zeronvs: ZeroNVS backend config
        zero123: Zero123++ backend config
        hybrid: Hybrid fallback config
    """
    # Core settings (backward compatible)
    enabled: bool = True
    backend: str = "depth_warping"
    rotation_angles: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
    rotation_axis: str = "vertical"
    max_hole_ratio: float = 0.15
    device: str = "cuda"

    # Backward compatibility alias
    inpainting_method: str = "opencv"

    # Backend-specific configs
    depth_warping: DepthWarpingConfig = field(default_factory=DepthWarpingConfig)
    zeronvs: ZeroNVSConfig = field(default_factory=ZeroNVSConfig)
    zero123: Zero123Config = field(default_factory=Zero123Config)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # Derived attributes for backend access
    @property
    def fallback_backends(self) -> List[str]:
        return self.hybrid.fallback_backends

    @property
    def fallback_hole_threshold(self) -> float:
        return self.hybrid.fallback_hole_threshold

    @property
    def zeronvs_model(self) -> str:
        return self.zeronvs.model_name

    @property
    def zero123_model(self) -> str:
        return self.zero123.model_name

    @property
    def num_inference_steps(self) -> int:
        """Get inference steps for active diffusion backend."""
        if self.backend in ("zeronvs", "zeronvs_lite"):
            return self.zeronvs.num_inference_steps
        elif self.backend == "zero123pp":
            return self.zero123.num_inference_steps
        return 50  # Default

    @property
    def guidance_scale(self) -> float:
        return self.zeronvs.guidance_scale

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ViewSynthesisConfigV2":
        """Create config from dictionary.

        Handles both flat and nested configuration formats.
        """
        # Handle nested backend configs
        depth_cfg = DepthWarpingConfig(
            **config_dict.pop("depth_warping", {})
        )
        zeronvs_cfg = ZeroNVSConfig(
            **config_dict.pop("zeronvs", {})
        )
        zero123_cfg = Zero123Config(
            **config_dict.pop("zero123", {})
        )
        hybrid_cfg = HybridConfig(
            **config_dict.pop("hybrid", {})
        )

        return cls(
            depth_warping=depth_cfg,
            zeronvs=zeronvs_cfg,
            zero123=zero123_cfg,
            hybrid=hybrid_cfg,
            **config_dict
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ViewSynthesisConfigV2":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict.get("view_synthesis", config_dict))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "rotation_angles": self.rotation_angles,
            "rotation_axis": self.rotation_axis,
            "max_hole_ratio": self.max_hole_ratio,
            "device": self.device,
            "inpainting_method": self.inpainting_method,
            "depth_warping": asdict(self.depth_warping),
            "zeronvs": asdict(self.zeronvs),
            "zero123": asdict(self.zero123),
            "hybrid": asdict(self.hybrid),
        }

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(
                {"view_synthesis": self.to_dict()},
                f,
                default_flow_style=False,
                sort_keys=False
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for backward compatibility."""
        return getattr(self, key, default)


def upgrade_legacy_config(legacy_config: Any) -> ViewSynthesisConfigV2:
    """Upgrade legacy ViewSynthesisConfig to ViewSynthesisConfigV2.

    Args:
        legacy_config: Old-style ViewSynthesisConfig or dict

    Returns:
        ViewSynthesisConfigV2 instance
    """
    if isinstance(legacy_config, ViewSynthesisConfigV2):
        return legacy_config

    if isinstance(legacy_config, dict):
        # Add backend field if missing
        if "backend" not in legacy_config:
            legacy_config["backend"] = "depth_warping"
        return ViewSynthesisConfigV2.from_dict(legacy_config)

    # Assume it's an old dataclass-style config
    config_dict = {
        "enabled": getattr(legacy_config, "enabled", True),
        "backend": "depth_warping",
        "rotation_angles": getattr(legacy_config, "rotation_angles", [-5.0, 0.0, 5.0]),
        "rotation_axis": getattr(legacy_config, "rotation_axis", "vertical"),
        "max_hole_ratio": getattr(legacy_config, "max_hole_ratio", 0.15),
        "inpainting_method": getattr(legacy_config, "inpainting_method", "opencv"),
    }

    return ViewSynthesisConfigV2.from_dict(config_dict)
