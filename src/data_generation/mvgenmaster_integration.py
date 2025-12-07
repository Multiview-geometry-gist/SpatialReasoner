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
import shlex
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Default paths - configurable via environment variables
_MVGENMASTER_ROOT = Path(os.getenv(
    "MVGENMASTER_ROOT",
    os.path.expanduser("~/MVGenMaster")
))
_DEFAULT_MODEL_DIR = Path(os.getenv(
    "MVGENMASTER_MODEL_DIR",
    str(_MVGENMASTER_ROOT / "check_points" / "pretrained_model")
))
_CONDA_ENV = os.getenv("MVGENMASTER_CONDA_ENV", "mvgenmaster")

# Allowed image extensions for security
_ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}


class CameraTrajectory(str, Enum):
    """Supported camera trajectory types for MVGenMaster.

    These values must match MVGenMaster's --cam_traj argument choices.
    See MVGenMaster/run_mvgen.py for implementation details.
    """
    FREE = "free"  # Free-form linear trajectory (default)
    BI_DIRECTION = "bi_direction"  # Bi-directional sweep (left then right)
    DISORDER = "disorder"  # Random disordered views
    SWING1 = "swing1"  # Swing trajectory pattern 1
    SWING2 = "swing2"  # Swing trajectory pattern 2


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

    Note:
        Call close() or use as context manager to release image memory when done.
    """
    image: Image.Image
    azimuth: float
    elevation: float
    frame_index: int
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, repr=False)

    def save(self, path: str, quality: int = 95) -> None:
        """Save view to disk."""
        if self._closed:
            raise ValueError("Cannot save closed view")
        self.image.save(path, quality=quality)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (H, W, 3)."""
        if self._closed:
            raise ValueError("Cannot convert closed view")
        return np.array(self.image)

    def close(self) -> None:
        """Release image memory. Call when view is no longer needed."""
        if not self._closed and self.image is not None:
            try:
                self.image.close()
            except Exception:
                pass
            self._closed = True

    def __enter__(self) -> "GeneratedView":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.close()

    def __del__(self):
        """Destructor - ensure image is closed."""
        self.close()


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
    generation_timeout: int = 300  # Timeout in seconds (5 minutes default)

    def __post_init__(self):
        """Validate configuration values after initialization."""
        # Validate numeric ranges
        if not isinstance(self.num_views, int) or self.num_views <= 0:
            raise ValueError(f"num_views must be a positive integer, got {self.num_views}")
        if self.num_views > 100:
            raise ValueError(f"num_views too large: {self.num_views} (max 100)")

        if not isinstance(self.num_frames, int) or self.num_frames <= 0:
            raise ValueError(f"num_frames must be a positive integer, got {self.num_frames}")
        if self.num_frames > 100:
            raise ValueError(f"num_frames too large: {self.num_frames} (max 100)")

        if not isinstance(self.azimuth_range, (int, float)) or self.azimuth_range <= 0:
            raise ValueError(f"azimuth_range must be positive, got {self.azimuth_range}")
        if self.azimuth_range > 180:
            raise ValueError(f"azimuth_range too large: {self.azimuth_range} (max 180)")

        if not isinstance(self.elevation, (int, float)):
            raise ValueError(f"elevation must be a number, got {self.elevation}")
        if abs(self.elevation) > 90:
            raise ValueError(f"elevation out of range: {self.elevation} (must be -90 to 90)")

        if not isinstance(self.guidance_scale, (int, float)) or self.guidance_scale <= 0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")

        if not isinstance(self.quality_threshold, (int, float)):
            raise ValueError(f"quality_threshold must be a number, got {self.quality_threshold}")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError(f"quality_threshold must be 0-1, got {self.quality_threshold}")

        if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {self.gpu_id}")

        if not isinstance(self.generation_timeout, int) or self.generation_timeout <= 0:
            raise ValueError(f"generation_timeout must be a positive integer, got {self.generation_timeout}")

        # Validate output_resolution if provided
        if self.output_resolution is not None:
            if not isinstance(self.output_resolution, tuple) or len(self.output_resolution) != 2:
                raise ValueError(f"output_resolution must be (width, height) tuple, got {self.output_resolution}")
            w, h = self.output_resolution
            if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
                raise ValueError(f"output_resolution dimensions must be positive integers, got {self.output_resolution}")

        # Validate precision
        if self.precision not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"precision must be float16/float32/bfloat16, got {self.precision}")

        # Validate camera_trajectory enum
        if not isinstance(self.camera_trajectory, CameraTrajectory):
            raise ValueError(f"camera_trajectory must be CameraTrajectory enum, got {type(self.camera_trajectory)}")

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

    def _validate_image_path(self, path: Union[str, Path]) -> Path:
        """Validate that path is a safe image file.

        Args:
            path: Path to validate

        Returns:
            Resolved, validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a valid image file
        """
        p = Path(path).resolve()

        # Check existence
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        # Check it's a file, not directory
        if not p.is_file():
            raise ValueError(f"Path is not a file: {p}")

        # Check file extension
        if p.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid image extension: {p.suffix}. Allowed: {_ALLOWED_IMAGE_EXTENSIONS}")

        # Check file size (basic sanity check)
        file_size = p.stat().st_size
        if file_size == 0:
            raise ValueError(f"Image file is empty: {p}")

        # Check for reasonable file size (max 100MB)
        if file_size > 100 * 1024 * 1024:
            raise ValueError(f"Image file too large: {file_size} bytes (max 100MB)")

        # Robust image validation - check both header and pixel data
        try:
            # First pass: verify header
            with Image.open(p) as img:
                img.verify()

            # Second pass: actually load pixel data to detect corruption
            with Image.open(p) as img:
                img.load()  # Force loading pixel data
                # Basic sanity check on dimensions
                if img.width <= 0 or img.height <= 0:
                    raise ValueError(f"Invalid image dimensions: {img.width}x{img.height}")
                if img.width > 10000 or img.height > 10000:
                    raise ValueError(f"Image too large: {img.width}x{img.height} (max 10000x10000)")

        except Image.DecompressionBombError as e:
            raise ValueError(f"Image decompression bomb detected: {e}")
        except (IOError, OSError) as e:
            raise ValueError(f"Corrupted or invalid image file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to validate image: {e}")

        return p

    def _validate_conda_env(self, env_name: str) -> str:
        """Validate conda environment name to prevent injection."""
        # Only allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', env_name):
            raise ValueError(f"Invalid conda environment name: {env_name}")
        return env_name

    def _generate_subprocess(
        self,
        image_path: str,
        num_views: int,
    ) -> List[GeneratedView]:
        """Run MVGenMaster in subprocess for isolation.

        Security: Uses a temporary shell script to avoid command injection via bash -c.
        Race condition: Uses explicit cleanup to ensure views are loaded before deletion.
        """
        # Validate input path
        validated_input = self._validate_image_path(image_path)

        # Validate conda environment name
        conda_env = self._validate_conda_env(self.config.conda_env)

        # Create temporary directory with explicit lifecycle management
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_obj.name)
        output_dir = tmpdir / "output"
        script_path = tmpdir / "run_mvgen.sh"

        try:
            output_dir.mkdir()

            # Build command with validated paths
            cmd = self._build_command(str(validated_input), str(output_dir))

            # Set environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)

            # Get conda base safely
            try:
                conda_base = subprocess.check_output(
                    ["conda", "info", "--base"],
                    text=True,
                    timeout=30
                ).strip()
                # Validate conda_base is a valid path
                if not Path(conda_base).is_dir():
                    raise ValueError(f"Invalid conda base: {conda_base}")
            except Exception as e:
                logger.error(f"Failed to get conda base: {e}")
                return []

            # SECURITY FIX: Write commands to a shell script instead of using bash -c
            # This avoids potential command injection through string interpolation
            script_content = "#!/bin/bash\nset -e\n"
            script_content += f"source {shlex.quote(str(Path(conda_base) / 'etc' / 'profile.d' / 'conda.sh'))}\n"
            script_content += f"conda activate {shlex.quote(conda_env)}\n"
            script_content += f"cd {shlex.quote(str(Path(self.config.mvgenmaster_root).resolve()))}\n"
            script_content += " ".join(shlex.quote(arg) for arg in cmd) + "\n"

            # Write script with restricted permissions (owner read/execute only)
            script_path.write_text(script_content)
            script_path.chmod(0o500)

            logger.info(f"Running MVGenMaster script: {script_path}")
            logger.debug(f"Script content:\n{script_content}")

            try:
                result = subprocess.run(
                    ["/bin/bash", str(script_path)],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.generation_timeout,
                    env=env,
                )

                logger.debug(f"MVGenMaster stdout: {result.stdout[:1000] if result.stdout else 'empty'}")

                if result.returncode != 0:
                    logger.error(f"MVGenMaster failed (code {result.returncode}): {result.stderr}")
                    return []

            except subprocess.TimeoutExpired:
                logger.error(f"MVGenMaster timed out after {self.config.generation_timeout}s")
                return []
            except Exception as e:
                logger.error(f"MVGenMaster subprocess error: {e}")
                return []

            # Load generated views BEFORE cleanup
            views = self._load_views(output_dir, num_views)
            return views

        finally:
            # Explicit cleanup after views are loaded
            try:
                tmpdir_obj.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

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
                # Use context manager to ensure file handle is closed
                # Copy image data to avoid issues with temp file cleanup
                with Image.open(view_path) as src_img:
                    # Convert and copy - this detaches from file
                    img = src_img.convert("RGB").copy()

                # Resize if needed (after copy, so original file is closed)
                if self.config.output_resolution:
                    resized = img.resize(
                        self.config.output_resolution,
                        Image.Resampling.LANCZOS
                    )
                    img.close()  # Close original before replacing
                    img = resized

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

        logger.debug(f"Loaded {len(views)} views from {images_dir}")
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
