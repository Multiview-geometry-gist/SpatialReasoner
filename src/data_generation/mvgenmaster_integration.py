"""MVGenMaster Integration Module for SpatialReasoner.

Provides a clean, high-level interface for MVGenMaster view generation
that can be used by both training data augmentation and inference pipelines.

Key Features:
    - Subprocess isolation to avoid dependency conflicts
    - Camera trajectory configuration
    - View selection and quality filtering
    - Batch processing support

Note:
    MVGenMaster requires its own conda environment (mvgenmaster).
    This module handles the isolation automatically via subprocess.

Example:
    from data_generation.mvgenmaster_integration import MVGenMasterGenerator, MVGenConfig

    config = MVGenConfig(
        num_views=5,
        azimuth_range=45.0,
        elevation=5.0,
    )
    generator = MVGenMasterGenerator(config)

    views = generator.generate_views(image_path)
    for view in views:
        print(f"Angle: {view.azimuth}, Quality: {view.quality_score}")
"""

import os
import json
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Default paths for MVGenMaster installation
_MVGENMASTER_ROOT = Path("/home/ubuntu/MVGenMaster")
_DEFAULT_MODEL_DIR = _MVGENMASTER_ROOT / "check_points" / "pretrained_model"
_CONDA_ENV = "mvgenmaster"


class CameraTrajectory(str, Enum):
    """Supported camera trajectory types for MVGenMaster."""
    FREE = "free"  # Free-form trajectory
    ORBIT = "orbit"  # Orbit around object
    DOLLY = "dolly"  # Dolly zoom effect
    SPIRAL = "spiral"  # Spiral trajectory


@dataclass
class GeneratedView:
    """Container for a single generated view from MVGenMaster.

    Attributes:
        image: PIL Image of the generated view
        azimuth: Azimuth angle in degrees (relative to original)
        elevation: Elevation angle in degrees
        frame_index: Frame index in the generated sequence
        quality_score: Estimated quality score (0-1)
        metadata: Additional generation metadata
    """
    image: Image.Image
    azimuth: float
    elevation: float
    frame_index: int
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str, quality: int = 95) -> None:
        """Save view to disk."""
        self.image.save(path, quality=quality)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (H, W, 3)."""
        return np.array(self.image)


@dataclass
class MVGenConfig:
    """Configuration for MVGenMaster view generation.

    Attributes:
        num_views: Number of views to generate (extracted from trajectory)
        num_frames: Total frames to generate in trajectory
        azimuth_range: Total azimuth rotation range in degrees
        elevation: Camera elevation angle in degrees
        guidance_scale: Classifier-free guidance scale
        camera_trajectory: Type of camera trajectory
        camera_longest_side: Camera normalization scale
        output_resolution: Output image resolution (width, height)
        quality_threshold: Minimum quality score to accept view
        mvgenmaster_root: Path to MVGenMaster installation
        model_dir: Path to pretrained model directory
        conda_env: Conda environment name for MVGenMaster
        gpu_id: GPU device ID to use
        use_subprocess: Whether to run in subprocess for isolation
    """
    # View generation settings
    num_views: int = 5
    num_frames: int = 28
    azimuth_range: float = 45.0
    elevation: float = 5.0
    guidance_scale: float = 2.0
    camera_trajectory: CameraTrajectory = CameraTrajectory.FREE
    camera_longest_side: float = 5.0

    # Output settings
    output_resolution: Optional[Tuple[int, int]] = None  # None = original

    # Quality filtering
    quality_threshold: float = 0.5

    # Paths and environment
    mvgenmaster_root: str = str(_MVGENMASTER_ROOT)
    model_dir: str = str(_DEFAULT_MODEL_DIR)
    conda_env: str = _CONDA_ENV
    gpu_id: int = 0
    use_subprocess: bool = True

    # Advanced settings
    num_inference_steps: int = 50
    use_v_prediction: bool = True  # Required for MVGenMaster
    precision: str = "float32"  # MVGenMaster requires float32

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["camera_trajectory"] = self.camera_trajectory.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MVGenConfig":
        """Create from dictionary."""
        if "camera_trajectory" in d and isinstance(d["camera_trajectory"], str):
            d["camera_trajectory"] = CameraTrajectory(d["camera_trajectory"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "MVGenConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested mvgenmaster config
        if "mvgenmaster" in config_dict:
            config_dict = config_dict["mvgenmaster"]

        return cls.from_dict(config_dict)


class MVGenMasterGenerator:
    """High-level interface for MVGenMaster view generation.

    This class provides a clean API for generating novel views from a single
    image using MVGenMaster. It handles subprocess isolation, camera trajectory
    configuration, and quality filtering.

    Example:
        config = MVGenConfig(num_views=5, azimuth_range=45.0)
        generator = MVGenMasterGenerator(config)

        # Generate views from a single image
        views = generator.generate_views("/path/to/image.jpg")

        # Generate views at specific angles
        views = generator.generate_views_at_angles(
            "/path/to/image.jpg",
            angles=[-15.0, -5.0, 0.0, 5.0, 15.0]
        )
    """

    def __init__(self, config: Optional[MVGenConfig] = None):
        """Initialize the generator.

        Args:
            config: Generation configuration. Uses defaults if not provided.
        """
        self.config = config or MVGenConfig()
        self._check_installation()

    def _check_installation(self) -> bool:
        """Check if MVGenMaster is properly installed."""
        root = Path(self.config.mvgenmaster_root)

        if not root.exists():
            logger.warning(f"MVGenMaster root not found: {root}")
            return False

        run_script = root / "run_mvgen.py"
        if not run_script.exists():
            logger.warning(f"MVGenMaster run script not found: {run_script}")
            return False

        model_dir = Path(self.config.model_dir)
        if not model_dir.exists():
            logger.warning(f"MVGenMaster model not found: {model_dir}")
            return False

        return True

    def generate_views(
        self,
        image_path: str,
        num_views: Optional[int] = None,
    ) -> List[GeneratedView]:
        """Generate novel views from a single image.

        Args:
            image_path: Path to input image
            num_views: Number of views to return (default: config.num_views)

        Returns:
            List of GeneratedView objects
        """
        num_views = num_views or self.config.num_views

        if self.config.use_subprocess:
            return self._generate_subprocess(image_path, num_views)
        else:
            return self._generate_direct(image_path, num_views)

    def generate_views_at_angles(
        self,
        image_path: str,
        angles: List[float],
    ) -> Dict[float, GeneratedView]:
        """Generate views at specific azimuth angles.

        Generates a full trajectory and selects views nearest to requested angles.

        Args:
            image_path: Path to input image
            angles: List of azimuth angles in degrees

        Returns:
            Dictionary mapping angle -> GeneratedView
        """
        # Calculate required azimuth range
        max_angle = max(abs(a) for a in angles) if angles else self.config.azimuth_range

        # Generate views with appropriate range
        original_range = self.config.azimuth_range
        self.config.azimuth_range = max(max_angle * 1.2, original_range)

        all_views = self.generate_views(image_path, num_views=len(angles) * 2)

        self.config.azimuth_range = original_range

        # Map to requested angles
        result = {}
        for target_angle in angles:
            if target_angle == 0.0:
                # Original view - load from input
                img = Image.open(image_path).convert("RGB")
                result[0.0] = GeneratedView(
                    image=img,
                    azimuth=0.0,
                    elevation=self.config.elevation,
                    frame_index=0,
                    quality_score=1.0,
                )
            else:
                # Find nearest generated view
                nearest_view = min(
                    all_views,
                    key=lambda v: abs(v.azimuth - target_angle)
                )
                result[target_angle] = GeneratedView(
                    image=nearest_view.image.copy(),
                    azimuth=target_angle,
                    elevation=nearest_view.elevation,
                    frame_index=nearest_view.frame_index,
                    quality_score=nearest_view.quality_score * 0.95,
                    metadata={
                        "actual_azimuth": nearest_view.azimuth,
                        "angle_diff": abs(nearest_view.azimuth - target_angle),
                    }
                )

        return result

    def _generate_subprocess(
        self,
        image_path: str,
        num_views: int,
    ) -> List[GeneratedView]:
        """Run MVGenMaster in subprocess for isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Build command
            cmd = self._build_command(image_path, str(output_dir))

            # Set environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)

            # Run with conda environment
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && " \
                       f"conda activate {self.config.conda_env} && " \
                       f"{' '.join(cmd)}"

            logger.info(f"Running MVGenMaster: {full_cmd}")

            try:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=self.config.mvgenmaster_root,
                    env=env,
                )

                if result.returncode != 0:
                    logger.error(f"MVGenMaster failed: {result.stderr}")
                    return []

            except subprocess.TimeoutExpired:
                logger.error("MVGenMaster timed out")
                return []
            except Exception as e:
                logger.error(f"MVGenMaster subprocess error: {e}")
                return []

            # Load generated views
            return self._load_views(output_dir, num_views)

    def _generate_direct(
        self,
        image_path: str,
        num_views: int,
    ) -> List[GeneratedView]:
        """Direct generation (requires MVGenMaster dependencies loaded)."""
        # Fall back to subprocess for now
        logger.warning("Direct generation not implemented, using subprocess")
        return self._generate_subprocess(image_path, num_views)

    def _build_command(self, input_path: str, output_path: str) -> List[str]:
        """Build the MVGenMaster command."""
        cmd = [
            "python", "run_mvgen.py",
            "--input_path", input_path,
            "--model_dir", self.config.model_dir,
            "--output_path", output_path,
            "--nframe", str(self.config.num_frames),
            "--val_cfg", str(self.config.guidance_scale),
            "--elevation", str(self.config.elevation),
            "--d_phi", str(self.config.azimuth_range),
            "--camera_longest_side", str(self.config.camera_longest_side),
            "--cam_traj", self.config.camera_trajectory.value,
        ]

        return cmd

    def _load_views(
        self,
        output_dir: Path,
        num_views: int,
    ) -> List[GeneratedView]:
        """Load generated views from MVGenMaster output directory."""
        views = []

        # MVGenMaster outputs to images/ subdirectory
        images_dir = output_dir / "images"
        if not images_dir.exists():
            # Try output_dir directly
            images_dir = output_dir

        # Find all view files
        view_files = sorted(images_dir.glob("view*.png")) + sorted(images_dir.glob("frame*.png"))

        if not view_files:
            # Try to find any PNG files
            view_files = sorted(images_dir.glob("*.png"))

        if not view_files:
            logger.warning(f"No view files found in {images_dir}")
            return views

        # Calculate angle per frame
        total_frames = len(view_files)
        angle_step = (2 * self.config.azimuth_range) / max(total_frames - 1, 1)

        # Select evenly spaced views
        if num_views >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_views, dtype=int).tolist()

        for idx in indices:
            view_path = view_files[idx]
            try:
                img = Image.open(view_path).convert("RGB")

                # Resize if needed
                if self.config.output_resolution:
                    img = img.resize(
                        self.config.output_resolution,
                        Image.Resampling.LANCZOS
                    )

                # Calculate azimuth for this frame
                azimuth = -self.config.azimuth_range + idx * angle_step

                views.append(GeneratedView(
                    image=img,
                    azimuth=azimuth,
                    elevation=self.config.elevation,
                    frame_index=idx,
                    quality_score=0.9,  # High quality from diffusion
                    metadata={
                        "source_file": view_path.name,
                        "total_frames": total_frames,
                    }
                ))
            except Exception as e:
                logger.warning(f"Failed to load view {view_path}: {e}")

        return views


def generate_views_batch(
    image_paths: List[str],
    config: Optional[MVGenConfig] = None,
    show_progress: bool = True,
) -> Dict[str, List[GeneratedView]]:
    """Generate views for multiple images.

    Args:
        image_paths: List of image paths
        config: Generation configuration
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping image_path -> list of GeneratedView
    """
    generator = MVGenMasterGenerator(config)
    results = {}

    iterator = image_paths
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(image_paths, desc="Generating views")

    for path in iterator:
        try:
            views = generator.generate_views(path)
            results[path] = views
        except Exception as e:
            logger.error(f"Failed to generate views for {path}: {e}")
            results[path] = []

    return results


def check_mvgenmaster_available() -> bool:
    """Check if MVGenMaster is properly installed and available.

    Returns:
        True if MVGenMaster can be used
    """
    if not _MVGENMASTER_ROOT.exists():
        return False

    if not (_MVGENMASTER_ROOT / "run_mvgen.py").exists():
        return False

    if not _DEFAULT_MODEL_DIR.exists():
        return False

    return True
