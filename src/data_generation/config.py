"""Configuration for data generation pipeline.

All configurations are defined as dataclasses for easy serialization/deserialization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
import yaml
from pathlib import Path


@dataclass
class DepthConfig:
    """Depth estimation configuration."""
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    device: str = "cuda"
    max_depth: float = 100.0  # meters (for metric depth scaling)


@dataclass
class SegmentationConfig:
    """SAM2 segmentation configuration."""
    model_name: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_mask_region_area: int = 100  # minimum pixel area


@dataclass
class ViewSynthesisConfig:
    """Novel view synthesis configuration."""
    enabled: bool = True
    rotation_angles: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
    rotation_axis: str = "vertical"  # "vertical" (y-axis) or "horizontal" (x-axis)
    inpainting_method: str = "opencv"  # "opencv", "lama", "none"
    max_hole_ratio: float = 0.15  # Skip if holes > 15%


@dataclass
class PoseEstimationConfig:
    """6-DOF pose estimation configuration."""
    method: str = "depth_based"  # "depth_based" or "heuristic"
    use_pca_orientation: bool = True


@dataclass
class QAGenerationConfig:
    """QA pair generation configuration."""
    # Query type ratios (should sum to 1.0)
    distance_ratio: float = 0.30
    directional_ratio: float = 0.35
    rotation_ratio: float = 0.35  # Emphasized for rotation-aware training

    num_qa_per_image: int = 10
    min_objects_per_image: int = 2
    max_objects_per_image: int = 10

    # Rotation-specific settings
    rotation_threshold: float = 0.5  # cosine threshold for "facing toward" queries


@dataclass
class DataGenerationConfig:
    """Master configuration for data generation pipeline."""

    # Sub-configs
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    view_synthesis: ViewSynthesisConfig = field(default_factory=ViewSynthesisConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    qa_generation: QAGenerationConfig = field(default_factory=QAGenerationConfig)

    # I/O settings
    input_dir: str = "./data/openimages"
    output_dir: str = "./data/multiview"

    # Processing settings
    num_workers: int = 4
    batch_size: int = 1
    num_images: int = 20000
    seed: int = 42

    # Quality filtering
    min_image_size: Tuple[int, int] = (256, 256)
    max_image_size: Tuple[int, int] = (2048, 2048)

    @classmethod
    def from_yaml(cls, path: str) -> "DataGenerationConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configs
        depth_cfg = DepthConfig(**config_dict.pop("depth", {}))
        seg_cfg = SegmentationConfig(**config_dict.pop("segmentation", {}))
        view_cfg = ViewSynthesisConfig(**config_dict.pop("view_synthesis", {}))
        pose_cfg = PoseEstimationConfig(**config_dict.pop("pose_estimation", {}))
        qa_cfg = QAGenerationConfig(**config_dict.pop("qa_generation", {}))

        return cls(
            depth=depth_cfg,
            segmentation=seg_cfg,
            view_synthesis=view_cfg,
            pose_estimation=pose_cfg,
            qa_generation=qa_cfg,
            **config_dict
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "depth": asdict(self.depth),
            "segmentation": asdict(self.segmentation),
            "view_synthesis": asdict(self.view_synthesis),
            "pose_estimation": asdict(self.pose_estimation),
            "qa_generation": asdict(self.qa_generation),
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "num_images": self.num_images,
            "seed": self.seed,
            "min_image_size": list(self.min_image_size),
            "max_image_size": list(self.max_image_size),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "depth": asdict(self.depth),
            "segmentation": asdict(self.segmentation),
            "view_synthesis": asdict(self.view_synthesis),
            "pose_estimation": asdict(self.pose_estimation),
            "qa_generation": asdict(self.qa_generation),
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "num_images": self.num_images,
            "seed": self.seed,
            "min_image_size": self.min_image_size,
            "max_image_size": self.max_image_size,
        }
