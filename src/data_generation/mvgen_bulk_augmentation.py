"""Bulk multi-view augmentation using MVGenMaster.

This module generates 50-100 different angle views from each image in a dataset,
suitable for training spatial reasoning models with view-aware augmentation.

Architecture:
    - Uses MVGenMaster subprocess wrapper for isolation
    - Supports both HuggingFace datasets and local directories
    - Generates views in batches for efficiency
    - Outputs augmented dataset with view metadata

Usage:
    python -m src.data_generation.mvgen_bulk_augmentation \
        --input_dataset ccvl/SpatialReasonerTrain-SFT \
        --output_dir ./data/mvgen_augmented \
        --num_views 50

Configuration:
    See configs/mvgen_bulk.yaml for detailed settings.
"""

import os
import json
import subprocess
import tempfile
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import re
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CoT Coordinate Transformation
# ============================================================================

def transform_cot_coordinates(
    answer_cot: str,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
) -> str:
    """Transform 3D coordinates in CoT answer based on camera rotation.

    When the camera rotates, the 3D coordinates in camera frame change.
    This function applies the inverse rotation to update coordinates correctly.

    Args:
        answer_cot: Original chain-of-thought answer with 3D coordinates
        azimuth_deg: Camera azimuth rotation in degrees (positive = right)
        elevation_deg: Camera elevation rotation in degrees (positive = up)

    Returns:
        CoT with transformed coordinates

    Example:
        Original: "The location of the object is (1.0, 0.5, 2.0)."
        After +15 deg azimuth: "The location of the object is (0.79, 0.5, 2.12)."
    """
    if not answer_cot:
        return answer_cot

    # Skip transformation for near-zero rotations
    if abs(azimuth_deg) < 0.5 and abs(elevation_deg) < 0.5:
        return answer_cot

    # Build rotation matrix
    # When camera moves by +azimuth, object coordinates in camera frame
    # rotate by -azimuth (inverse transformation)
    theta_y = np.radians(-azimuth_deg)  # Azimuth: rotation around Y axis
    theta_x = np.radians(-elevation_deg)  # Elevation: rotation around X axis

    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)

    # Combined rotation: R = Rx @ Ry
    R = np.array([
        [cy, 0, sy],
        [sx * sy, cx, -sx * cy],
        [-cx * sy, sx, cx * cy]
    ])

    def rotate_coords(match):
        """Apply rotation to matched coordinates."""
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))

            # Apply rotation
            vec = np.array([x, y, z])
            new_vec = R @ vec

            # Format with same precision as original
            return f"({new_vec[0]:.1f}, {new_vec[1]:.1f}, {new_vec[2]:.1f})"
        except ValueError:
            return match.group(0)

    # Regex pattern to match (x.x, y.y, z.z) coordinate tuples
    # Handles optional minus signs and decimal points
    coord_pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"

    transformed = re.sub(coord_pattern, rotate_coords, answer_cot)

    return transformed


def add_view_context_to_cot(
    answer_cot: str,
    azimuth_deg: float,
    strategy: str = "prefix",
) -> str:
    """Add view angle context to CoT answer.

    Args:
        answer_cot: Chain-of-thought answer
        azimuth_deg: View angle in degrees
        strategy: How to add context ("prefix", "suffix", "none")

    Returns:
        CoT with view context added
    """
    if not answer_cot or strategy == "none":
        return answer_cot

    abs_angle = abs(azimuth_deg)
    if abs_angle < 1.0:
        view_desc = "From a frontal view"
    elif azimuth_deg > 0:
        if abs_angle < 15:
            view_desc = f"From a slightly right-rotated view ({azimuth_deg:.0f}°)"
        elif abs_angle < 45:
            view_desc = f"From a right-rotated view ({azimuth_deg:.0f}°)"
        else:
            view_desc = f"From a significantly right-rotated view ({azimuth_deg:.0f}°)"
    else:
        if abs_angle < 15:
            view_desc = f"From a slightly left-rotated view ({azimuth_deg:.0f}°)"
        elif abs_angle < 45:
            view_desc = f"From a left-rotated view ({azimuth_deg:.0f}°)"
        else:
            view_desc = f"From a significantly left-rotated view ({azimuth_deg:.0f}°)"

    if strategy == "prefix":
        return f"{view_desc}, analyzing the spatial relationships: {answer_cot}"
    elif strategy == "suffix":
        return f"{answer_cot} (Observed from {azimuth_deg:.0f}° rotation)"
    else:
        return answer_cot


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MVGenBulkConfig:
    """Configuration for bulk MVGenMaster augmentation.

    Attributes:
        mvgenmaster_root: Path to MVGenMaster installation
        model_dir: Path to MVGenMaster model checkpoints
        num_views: Total number of views to generate per image (50-100 recommended)
        azimuth_range: Range of azimuth rotation in degrees (e.g., 60 = -60 to +60)
        elevation_range: Range of elevation rotation in degrees
        num_trajectories: Number of different trajectory runs to achieve target views
        frames_per_trajectory: Frames per MVGenMaster run (max 28 default)
        guidance_scale: CFG scale for diffusion
        seed: Random seed for reproducibility
        camera_longest_side: Camera normalization scale
        timeout_per_batch: Timeout in seconds for each MVGenMaster run
        max_workers: Number of parallel workers for processing
        save_video: Whether to save video outputs
        output_format: Image output format ('jpg' or 'png')
        output_quality: JPEG quality (1-100)
        transform_cot_coordinates: Whether to transform 3D coordinates in CoT
        cot_view_context_strategy: How to add view context ("prefix", "suffix", "none")
    """
    # MVGenMaster paths
    mvgenmaster_root: str = "/home/ubuntu/MVGenMaster"
    model_dir: str = "/home/ubuntu/MVGenMaster/check_points/pretrained_model"

    # View generation settings
    num_views: int = 50  # Target number of views per image
    azimuth_range: float = 60.0  # +/- degrees for azimuth
    elevation_range: float = 15.0  # +/- degrees for elevation
    num_trajectories: int = 3  # Run multiple trajectories to get more views
    frames_per_trajectory: int = 28  # MVGenMaster default

    # Model settings
    guidance_scale: float = 2.0
    seed: int = 42
    camera_longest_side: float = 5.0

    # Processing settings
    timeout_per_batch: int = 600  # 10 minutes
    max_workers: int = 1  # MVGenMaster is GPU-bound, typically 1

    # Output settings
    save_video: bool = False
    output_format: str = "jpg"
    output_quality: int = 95

    # CoT augmentation settings
    transform_cot_coordinates: bool = True  # Transform 3D coordinates based on view angle
    cot_view_context_strategy: str = "prefix"  # "prefix", "suffix", or "none"

    @classmethod
    def from_yaml(cls, path: str) -> "MVGenBulkConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ============================================================================
# Trajectory Generation
# ============================================================================

@dataclass
class CameraTrajectory:
    """Defines a camera trajectory for view generation.

    Attributes:
        name: Trajectory identifier
        d_phi: Azimuth rotation angle
        d_theta: Elevation rotation angle
        trajectory_type: MVGenMaster trajectory type
    """
    name: str
    d_phi: float  # Azimuth
    d_theta: float  # Elevation
    trajectory_type: str = "free"


def generate_trajectory_configs(
    config: MVGenBulkConfig,
    seed: int = 42
) -> List[CameraTrajectory]:
    """Generate diverse camera trajectories to cover target number of views.

    This creates multiple trajectory configurations that, when combined,
    produce the target number of unique views with good angular coverage.

    Args:
        config: Bulk augmentation configuration
        seed: Random seed for trajectory generation

    Returns:
        List of CameraTrajectory objects
    """
    random.seed(seed)
    trajectories = []

    # Strategy: Create trajectories that sweep different azimuth ranges
    # to maximize angular diversity

    # Calculate how many trajectories needed
    views_per_traj = config.frames_per_trajectory - 1  # First frame is reference
    num_traj_needed = max(1, (config.num_views + views_per_traj - 1) // views_per_traj)
    num_traj_needed = min(num_traj_needed, config.num_trajectories)

    # Create trajectories with different azimuth sweeps
    azimuth_step = (2 * config.azimuth_range) / max(num_traj_needed, 1)

    for i in range(num_traj_needed):
        # Vary the azimuth center for each trajectory
        if num_traj_needed == 1:
            d_phi = config.azimuth_range
            d_theta = config.elevation_range / 2
        else:
            # Distribute trajectories across azimuth range
            d_phi = config.azimuth_range * (0.5 + 0.5 * (i / (num_traj_needed - 1)))
            # Alternate elevation direction
            d_theta = config.elevation_range * (0.3 if i % 2 == 0 else -0.3)

        trajectories.append(CameraTrajectory(
            name=f"traj_{i:02d}_phi{d_phi:.0f}_theta{d_theta:.0f}",
            d_phi=d_phi,
            d_theta=d_theta,
            trajectory_type="free"
        ))

    # Add swing trajectories for more natural motion if needed
    if config.num_views > 50:
        trajectories.append(CameraTrajectory(
            name="swing1",
            d_phi=config.azimuth_range * 0.7,
            d_theta=config.elevation_range * 0.5,
            trajectory_type="swing1"
        ))

    return trajectories


# ============================================================================
# MVGenMaster Runner
# ============================================================================

class MVGenMasterBulkRunner:
    """Runs MVGenMaster to generate multiple views from a single image.

    This class handles:
    - Subprocess management for MVGenMaster
    - Multiple trajectory runs to achieve target view count
    - Output collection and organization
    - Error handling and recovery
    """

    def __init__(self, config: MVGenBulkConfig):
        """Initialize the bulk runner.

        Args:
            config: Bulk augmentation configuration
        """
        self.config = config
        self._root = Path(config.mvgenmaster_root)
        self._model_dir = Path(config.model_dir)
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Verify MVGenMaster is properly installed."""
        if not self._root.exists():
            raise RuntimeError(f"MVGenMaster not found at {self._root}")

        run_script = self._root / "run_mvgen.py"
        if not run_script.exists():
            raise RuntimeError(f"run_mvgen.py not found at {run_script}")

        if not self._model_dir.exists():
            raise RuntimeError(f"Model checkpoints not found at {self._model_dir}")

    def _validate_image_path(self, path: str) -> str:
        """Validate image path to prevent command injection and path traversal.

        Args:
            path: Image file path to validate

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is invalid or suspicious
        """
        path_obj = Path(path).resolve()

        # Check path exists and is a file
        if not path_obj.exists():
            raise ValueError(f"Image file not found: {path}")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
        if path_obj.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image extension: {path_obj.suffix}")

        # Prevent path traversal attacks
        if ".." in str(path):
            raise ValueError(f"Path traversal detected in: {path}")

        return str(path_obj)

    def generate_views(
        self,
        image_path: str,
        output_dir: str,
        image_id: str,
    ) -> Dict[str, Any]:
        """Generate multiple views from a single image.

        Args:
            image_path: Path to input image
            output_dir: Directory for output views
            image_id: Unique identifier for the image

        Returns:
            Dictionary with:
                - 'views': Dict mapping angle_str -> relative path
                - 'metadata': Generation metadata
                - 'success': Boolean indicating success
        """
        trajectories = generate_trajectory_configs(self.config, self.config.seed)

        all_views = {}
        all_angles = set()
        metadata = {
            "image_id": image_id,
            "source_image": image_path,
            "trajectories": [],
            "total_views": 0,
        }

        # Create output directory for this image
        image_output_dir = Path(output_dir) / image_id
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy original image as view_0
        original_dest = image_output_dir / f"view_+0.0.{self.config.output_format}"
        self._copy_image(image_path, str(original_dest))
        all_views["0.0"] = str(original_dest.relative_to(output_dir))
        all_angles.add(0.0)

        # Run each trajectory
        for traj in trajectories:
            try:
                traj_views = self._run_trajectory(
                    image_path=image_path,
                    trajectory=traj,
                    output_dir=str(image_output_dir),
                    image_id=image_id,
                )

                for angle_str, view_path in traj_views.items():
                    angle = float(angle_str)
                    if angle not in all_angles:
                        all_views[angle_str] = str(
                            Path(view_path).relative_to(output_dir)
                        )
                        all_angles.add(angle)

                metadata["trajectories"].append({
                    "name": traj.name,
                    "d_phi": traj.d_phi,
                    "d_theta": traj.d_theta,
                    "views_generated": len(traj_views),
                })

            except Exception as e:
                logger.warning(f"Trajectory {traj.name} failed for {image_id}: {e}")
                metadata["trajectories"].append({
                    "name": traj.name,
                    "error": str(e),
                })

            # Check if we have enough views
            if len(all_views) >= self.config.num_views:
                break

        metadata["total_views"] = len(all_views)

        return {
            "views": all_views,
            "metadata": metadata,
            "success": len(all_views) > 1,
        }

    def _run_trajectory(
        self,
        image_path: str,
        trajectory: CameraTrajectory,
        output_dir: str,
        image_id: str,
    ) -> Dict[str, str]:
        """Run a single MVGenMaster trajectory.

        Args:
            image_path: Path to input image
            trajectory: Trajectory configuration
            output_dir: Output directory for views
            image_id: Image identifier

        Returns:
            Dictionary mapping angle_str -> view_path
        """
        # Validate image path before subprocess execution
        validated_path = self._validate_image_path(image_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = Path(tmpdir) / "output"
            tmp_output.mkdir()

            # Build MVGenMaster command with validated path
            cmd = [
                "python", str(self._root / "run_mvgen.py"),
                "--input_path", validated_path,
                "--model_dir", str(self._model_dir),
                "--output_path", str(tmp_output),
                "--nframe", str(self.config.frames_per_trajectory),
                "--val_cfg", str(self.config.guidance_scale),
                "--d_phi", str(trajectory.d_phi),
                "--d_theta", str(trajectory.d_theta),
                "--camera_longest_side", str(self.config.camera_longest_side),
                "--cam_traj", trajectory.trajectory_type,
                "--seed", str(self.config.seed),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_per_batch,
                    cwd=str(self._root),
                )

                if result.returncode != 0:
                    logger.debug(f"MVGenMaster stderr: {result.stderr[:500]}")
                    raise RuntimeError(f"MVGenMaster failed: {result.returncode}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("MVGenMaster timed out")

            # Collect generated views
            views = self._collect_views(
                tmp_output,
                output_dir,
                image_id,
                trajectory,
            )

            return views

    def _collect_views(
        self,
        tmp_output: Path,
        output_dir: str,
        image_id: str,
        trajectory: CameraTrajectory,
    ) -> Dict[str, str]:
        """Collect and rename generated views from MVGenMaster output.

        MVGenMaster outputs views named view000.png, view001.png, etc.
        We rename them based on estimated rotation angles.

        Args:
            tmp_output: Temporary output directory from MVGenMaster
            output_dir: Final output directory
            image_id: Image identifier
            trajectory: Trajectory that generated these views

        Returns:
            Dictionary mapping angle_str -> final_view_path
        """
        views = {}
        images_dir = tmp_output / "images"

        if not images_dir.exists():
            logger.warning(f"No images directory found in {tmp_output}")
            return views

        view_files = sorted(images_dir.glob("view*.png"))
        if not view_files:
            return views

        # Skip the reference frame (view000_ref.png or first frame)
        ref_files = [f for f in view_files if "_ref" in f.name]
        gen_files = [f for f in view_files if "_ref" not in f.name]

        num_frames = len(gen_files)
        if num_frames == 0:
            return views

        # Estimate angles for each frame based on trajectory
        # MVGenMaster generates frames linearly along the trajectory
        for i, view_file in enumerate(gen_files):
            # Linear interpolation of angle
            t = (i + 1) / num_frames
            angle = trajectory.d_phi * t

            # Create a unique angle string (with sign)
            angle_str = f"{angle:+.1f}"

            # Destination path
            dest_name = f"view_{angle_str}.{self.config.output_format}"
            dest_path = Path(output_dir) / dest_name

            # Avoid overwriting existing views with similar angles
            if dest_path.exists():
                angle_str = f"{angle:+.2f}"
                dest_name = f"view_{angle_str}.{self.config.output_format}"
                dest_path = Path(output_dir) / dest_name

            self._copy_image(str(view_file), str(dest_path))
            views[angle_str.replace("+", "")] = str(dest_path)

        return views

    def _copy_image(self, src: str, dest: str) -> None:
        """Copy and optionally convert image.

        Args:
            src: Source image path
            dest: Destination image path
        """
        img = Image.open(src).convert("RGB")

        if self.config.output_format.lower() == "jpg":
            img.save(dest, "JPEG", quality=self.config.output_quality)
        else:
            img.save(dest, "PNG")


# ============================================================================
# Dataset Augmentation Pipeline
# ============================================================================

@dataclass
class AugmentedSample:
    """Container for an augmented training sample.

    Attributes:
        original_sample: Original dataset sample
        view_images: Dict mapping angle -> image_path
        augmented_cots: Dict mapping angle -> transformed answer_cot
        selected_angle: Currently selected view angle
        metadata: Augmentation metadata
    """
    original_sample: Dict[str, Any]
    view_images: Dict[str, str]
    augmented_cots: Dict[str, str]  # angle_str -> transformed CoT
    selected_angle: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        The output includes:
        - All original fields (question, answer, A, B, C, D, category, etc.)
        - view_images: Dict mapping angle to image path
        - augmented_cots: Dict mapping angle to transformed CoT
        - answer_cot: Original CoT (for reference)
        - selected_angle: Default selected angle
        """
        result = {
            **self.original_sample,
            "view_images": self.view_images,
            "augmented_cots": self.augmented_cots,
            "selected_angle": self.selected_angle,
            "augmentation_metadata": self.metadata,
        }
        return result


class MVGenBulkAugmentor:
    """Augments a dataset with multi-view images from MVGenMaster.

    This is the main entry point for bulk augmentation. It:
    - Loads datasets from HuggingFace or local directories
    - Generates multiple views for each image
    - Creates an augmented dataset with view metadata
    - Saves the augmented dataset for training

    Example:
        config = MVGenBulkConfig(num_views=50)
        augmentor = MVGenBulkAugmentor(config)
        augmentor.augment_dataset(
            input_dataset="ccvl/SpatialReasonerTrain-SFT",
            output_dir="./data/mvgen_augmented",
            image_base_dir="./data/openimages",
        )
    """

    def __init__(self, config: MVGenBulkConfig):
        """Initialize the augmentor.

        Args:
            config: Bulk augmentation configuration
        """
        self.config = config
        self.runner = MVGenMasterBulkRunner(config)

    def augment_dataset(
        self,
        input_dataset: str,
        output_dir: str,
        image_base_dir: str,
        split: str = "train",
        limit: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> str:
        """Augment an entire dataset with multi-view images.

        Args:
            input_dataset: HuggingFace dataset name or local JSON path
            output_dir: Output directory for augmented data
            image_base_dir: Base directory for original images
            split: Dataset split to process
            limit: Optional limit on number of samples
            resume_from: Optional path to resume processing

        Returns:
            Path to augmented dataset JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load dataset
        samples = self._load_dataset(input_dataset, split, limit)
        logger.info(f"Loaded {len(samples)} samples from {input_dataset}")

        # Load progress if resuming
        processed_ids = set()
        augmented_samples = []
        if resume_from and Path(resume_from).exists():
            with open(resume_from, "r") as f:
                augmented_samples = json.load(f)
            processed_ids = {s["question_index"] for s in augmented_samples}
            logger.info(f"Resuming from {len(processed_ids)} processed samples")

        # Process each sample
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        for sample in tqdm(samples, desc="Augmenting dataset"):
            sample_id = sample.get("question_index", sample.get("id", "unknown"))

            if sample_id in processed_ids:
                continue

            try:
                augmented = self._augment_sample(
                    sample,
                    image_base_dir,
                    str(images_dir),
                )
                augmented_samples.append(augmented.to_dict())

                # Save progress periodically
                if len(augmented_samples) % 10 == 0:
                    self._save_progress(augmented_samples, output_path)

            except Exception as e:
                logger.error(f"Failed to augment {sample_id}: {e}")
                # Add sample with original view only
                original_cot = sample.get("answer_cot", "")
                fallback = AugmentedSample(
                    original_sample=sample,
                    view_images={"0.0": sample.get("image_filename", "")},
                    augmented_cots={"0.0": original_cot},  # Keep original CoT
                    selected_angle=0.0,
                    metadata={"error": str(e)},
                )
                augmented_samples.append(fallback.to_dict())

        # Save final augmented dataset
        output_json = output_path / "augmented_dataset.json"
        with open(output_json, "w") as f:
            json.dump(augmented_samples, f, indent=2)

        # Save configuration
        self.config.to_yaml(str(output_path / "augmentation_config.yaml"))

        # Save metadata
        metadata = {
            "num_samples": len(augmented_samples),
            "num_views_target": self.config.num_views,
            "source_dataset": input_dataset,
            "split": split,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Augmented dataset saved to {output_json}")
        return str(output_json)

    def _load_dataset(
        self,
        input_dataset: str,
        split: str,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace or local file.

        Args:
            input_dataset: Dataset name or path
            split: Dataset split
            limit: Optional sample limit

        Returns:
            List of sample dictionaries
        """
        if input_dataset.endswith(".json"):
            # Local JSON file
            with open(input_dataset, "r") as f:
                samples = json.load(f)
        elif HF_AVAILABLE and "/" in input_dataset:
            # HuggingFace dataset
            dataset = load_dataset(input_dataset, split=split)
            samples = [dict(s) for s in dataset]
        else:
            raise ValueError(f"Cannot load dataset: {input_dataset}")

        if limit:
            samples = samples[:limit]

        return samples

    def _augment_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str,
        images_output_dir: str,
    ) -> AugmentedSample:
        """Augment a single sample with multi-view images and transformed CoTs.

        This method:
        1. Generates multiple view images using MVGenMaster
        2. Transforms the answer_cot coordinates for each view angle
        3. Adds view context to the CoT (if configured)

        Args:
            sample: Original sample dictionary
            image_base_dir: Base directory for original images
            images_output_dir: Output directory for views

        Returns:
            AugmentedSample with view images and transformed CoTs
        """
        # Get image path
        image_filename = sample.get("image_filename") or sample.get("image_path", "")
        if not image_filename:
            raise ValueError("Sample has no image_filename or image_path")

        image_path = os.path.join(image_base_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate image ID
        sample_id = sample.get("question_index", Path(image_filename).stem)

        # Generate views
        result = self.runner.generate_views(
            image_path=image_path,
            output_dir=images_output_dir,
            image_id=sample_id,
        )

        # Get original answer_cot
        original_cot = sample.get("answer_cot", "")

        # Generate transformed CoT for each view angle
        augmented_cots = {}
        for angle_str, image_path in result["views"].items():
            angle = float(angle_str)

            if self.config.transform_cot_coordinates and original_cot:
                # Transform 3D coordinates based on camera rotation
                transformed_cot = transform_cot_coordinates(
                    answer_cot=original_cot,
                    azimuth_deg=angle,
                    elevation_deg=0.0,  # Assuming mostly azimuth rotation
                )

                # Add view context if configured
                if self.config.cot_view_context_strategy != "none":
                    transformed_cot = add_view_context_to_cot(
                        answer_cot=transformed_cot,
                        azimuth_deg=angle,
                        strategy=self.config.cot_view_context_strategy,
                    )

                augmented_cots[angle_str] = transformed_cot
            else:
                # No transformation, use original
                augmented_cots[angle_str] = original_cot

        return AugmentedSample(
            original_sample=sample,
            view_images=result["views"],
            augmented_cots=augmented_cots,
            selected_angle=0.0,  # Default to original view
            metadata=result["metadata"],
        )

    def _save_progress(
        self,
        samples: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save progress to a checkpoint file.

        Args:
            samples: Processed samples so far
            output_path: Output directory
        """
        checkpoint_path = output_path / "augmented_dataset_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(samples, f)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for bulk augmentation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-view augmented dataset using MVGenMaster"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., ccvl/SpatialReasonerTrain-SFT) or local JSON path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for augmented dataset",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="./data/openimages",
        help="Base directory for original images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=50,
        help="Number of views to generate per image (default: 50)",
    )
    parser.add_argument(
        "--azimuth_range",
        type=float,
        default=60.0,
        help="Azimuth rotation range in degrees (default: 60)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load or create config
    if args.config and Path(args.config).exists():
        config = MVGenBulkConfig.from_yaml(args.config)
    else:
        config = MVGenBulkConfig(
            num_views=args.num_views,
            azimuth_range=args.azimuth_range,
            seed=args.seed,
        )

    # Run augmentation
    augmentor = MVGenBulkAugmentor(config)
    output_path = augmentor.augment_dataset(
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        image_base_dir=args.image_base_dir,
        split=args.split,
        limit=args.limit,
        resume_from=args.resume,
    )

    print(f"Augmented dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
