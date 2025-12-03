"""Configuration for data generation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class DepthConfig:
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    device: str = "cuda"
    scale_factor: float = 10.0  # Approximate scene depth range in meters


@dataclass
class SegmentationConfig:
    model_name: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_mask_region_area: int = 100


@dataclass
class ViewSynthesisConfig:
    rotation_angles: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
    rotation_axis: str = "vertical"  # "vertical" (y-axis) or "horizontal" (x-axis)
    focal_length: Optional[float] = None
    max_hole_ratio: float = 0.15


@dataclass
class PoseEstimationConfig:
    method: str = "depth_based"  # "depth_based" or "heuristic"
    use_pca_orientation: bool = True


@dataclass
class QAGenerationConfig:
    distance_ratio: float = 0.35
    directional_ratio: float = 0.35
    rotation_ratio: float = 0.30
    num_qa_per_image: int = 10
    min_objects_per_image: int = 2
    rotation_threshold: float = 0.5  # cos(theta) threshold for "facing toward"


@dataclass
class DataGenerationConfig:
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    view_synthesis: ViewSynthesisConfig = field(default_factory=ViewSynthesisConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    qa_generation: QAGenerationConfig = field(default_factory=QAGenerationConfig)

    input_dir: str = "./data/openimages"
    output_dir: str = "./data/multiview"
    num_workers: int = 4
    num_images: int = 20000
    seed: int = 42
    min_image_size: Tuple[int, int] = (256, 256)
    max_image_size: Tuple[int, int] = (2048, 2048)

    @classmethod
    def from_yaml(cls, path: str) -> "DataGenerationConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        depth = DepthConfig(**data.pop("depth", {}))
        seg = SegmentationConfig(**data.pop("segmentation", {}))
        view = ViewSynthesisConfig(**data.pop("view_synthesis", {}))
        pose = PoseEstimationConfig(**data.pop("pose_estimation", {}))
        qa = QAGenerationConfig(**data.pop("qa_generation", {}))

        return cls(
            depth=depth, segmentation=seg, view_synthesis=view,
            pose_estimation=pose, qa_generation=qa, **data
        )

    def to_yaml(self, path: str):
        data = {
            "depth": self.depth.__dict__,
            "segmentation": self.segmentation.__dict__,
            "view_synthesis": self.view_synthesis.__dict__,
            "pose_estimation": self.pose_estimation.__dict__,
            "qa_generation": self.qa_generation.__dict__,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "num_images": self.num_images,
            "seed": self.seed,
            "min_image_size": self.min_image_size,
            "max_image_size": self.max_image_size,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
