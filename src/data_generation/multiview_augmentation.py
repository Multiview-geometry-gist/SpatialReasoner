"""Multi-View Augmentation Module for SpatialReasoner (Approach 1).

Implements single image training with view augmentation strategy:
- Generate multiple views from each training image
- Each view becomes a separate training sample
- Maintains relationship metadata between views

This approach increases training data diversity while preserving
the single-image input format expected by standard VLM training.

Example:
    from data_generation.multiview_augmentation import (
        ViewAugmentedDataGenerator,
        AugmentationConfig
    )

    config = AugmentationConfig(
        views_per_image=5,
        rotation_angles=[-15, -5, 0, 5, 15],
    )
    generator = ViewAugmentedDataGenerator(config)

    # Generate augmented samples from a single image
    samples = generator.augment_image(
        image_path="input.jpg",
        qa_pairs=original_qa_pairs
    )
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterator
from PIL import Image
import numpy as np
from tqdm import tqdm

from .mvgenmaster_integration import (
    MVGenMasterGenerator,
    MVGenConfig,
    GeneratedView,
    check_mvgenmaster_available,
)

logger = logging.getLogger(__name__)


@dataclass
class ViewMetadata:
    """Metadata for a generated view.

    Attributes:
        view_id: Unique identifier for this view
        source_image_id: ID of the original source image
        azimuth: Azimuth angle relative to original (degrees)
        elevation: Elevation angle (degrees)
        is_original: Whether this is the original (unaugmented) view
        generation_method: How the view was generated
        quality_score: View quality score (0-1)
    """
    view_id: str
    source_image_id: str
    azimuth: float
    elevation: float
    is_original: bool = False
    generation_method: str = "mvgenmaster"
    quality_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ViewMetadata":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class AugmentedSample:
    """A single augmented training sample.

    Attributes:
        image_path: Path to the view image
        question: Question text
        answer: Answer text (with chain-of-thought if applicable)
        view_metadata: Metadata about the view
        original_qa_metadata: Metadata from original QA pair
        spatial_context: Optional spatial context information
    """
    image_path: str
    question: str
    answer: str
    view_metadata: ViewMetadata
    original_qa_metadata: Dict[str, Any] = field(default_factory=dict)
    spatial_context: Optional[str] = None

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to standard training format.

        Returns format compatible with SFT training collator.
        """
        return {
            "image_filename": os.path.basename(self.image_path),
            "question": self.question,
            "answer_cot": self.answer,
            "view_metadata": self.view_metadata.to_dict(),
            **self.original_qa_metadata,
        }


@dataclass
class AugmentationConfig:
    """Configuration for view augmentation.

    Attributes:
        views_per_image: Number of views to generate per image
        rotation_angles: Specific rotation angles to generate (degrees)
        include_original: Whether to include the original (0-degree) view
        azimuth_range: Total azimuth range for generation
        elevation: Camera elevation angle
        quality_threshold: Minimum quality score to include view
        modify_questions_for_view: Whether to modify questions based on view angle
        add_view_context: Whether to add view context to questions
        output_dir: Directory to save augmented images
        save_metadata: Whether to save view metadata
        mvgenmaster_config: Configuration for MVGenMaster
    """
    # View generation settings
    views_per_image: int = 5
    rotation_angles: List[float] = field(
        default_factory=lambda: [-15.0, -5.0, 0.0, 5.0, 15.0]
    )
    include_original: bool = True
    azimuth_range: float = 45.0
    elevation: float = 5.0

    # Quality filtering
    quality_threshold: float = 0.5

    # Sample modification
    modify_questions_for_view: bool = False
    add_view_context: bool = True

    # Output settings
    output_dir: str = "./augmented_views"
    save_metadata: bool = True

    # MVGenMaster config
    mvgenmaster_config: Optional[MVGenConfig] = None

    def get_mvgenmaster_config(self) -> MVGenConfig:
        """Get or create MVGenMaster configuration."""
        if self.mvgenmaster_config is not None:
            return self.mvgenmaster_config

        return MVGenConfig(
            num_views=self.views_per_image,
            azimuth_range=self.azimuth_range,
            elevation=self.elevation,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AugmentationConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested structure
        if "augmentation" in config_dict:
            config_dict = config_dict["augmentation"]

        # Handle mvgenmaster_config separately
        mvgen_dict = config_dict.pop("mvgenmaster_config", None)
        if mvgen_dict:
            config_dict["mvgenmaster_config"] = MVGenConfig.from_dict(mvgen_dict)

        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        config_dict = asdict(self)
        if self.mvgenmaster_config:
            config_dict["mvgenmaster_config"] = self.mvgenmaster_config.to_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {"augmentation": config_dict},
                f,
                default_flow_style=False,
                sort_keys=False
            )


class ViewAugmentedDataGenerator:
    """Generates view-augmented training data from single images.

    This class implements Approach 1: Single Image Training with View Augmentation.
    It takes original training samples and creates multiple training samples
    from each image by generating novel views with MVGenMaster.

    Example:
        config = AugmentationConfig(
            views_per_image=5,
            rotation_angles=[-15, -5, 0, 5, 15],
        )
        generator = ViewAugmentedDataGenerator(config)

        # Augment a single image
        samples = generator.augment_image(
            image_path="train/image1.jpg",
            qa_pairs=[
                {"question": "What is to the left?", "answer": "A chair"},
            ]
        )

        # Augment an entire dataset
        all_samples = generator.augment_dataset(
            qa_pairs_file="train/qa_pairs.json",
            images_dir="train/images",
        )
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize the generator.

        Args:
            config: Augmentation configuration. Uses defaults if not provided.
        """
        self.config = config or AugmentationConfig()
        self._mvgen_generator = None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    @property
    def mvgen_generator(self) -> MVGenMasterGenerator:
        """Lazy initialization of MVGenMaster generator."""
        if self._mvgen_generator is None:
            mvgen_config = self.config.get_mvgenmaster_config()
            self._mvgen_generator = MVGenMasterGenerator(mvgen_config)
        return self._mvgen_generator

    def augment_image(
        self,
        image_path: str,
        qa_pairs: List[Dict[str, Any]],
        image_id: Optional[str] = None,
    ) -> List[AugmentedSample]:
        """Generate augmented samples from a single image.

        Args:
            image_path: Path to source image
            qa_pairs: List of QA pairs for this image
            image_id: Optional image identifier (derived from path if not provided)

        Returns:
            List of AugmentedSample objects, one for each (view, qa) combination
        """
        if image_id is None:
            image_id = Path(image_path).stem

        # Generate views at specified angles
        views = self._generate_views(image_path, image_id)

        if not views:
            logger.warning(f"No views generated for {image_path}")
            return []

        # Create augmented samples
        samples = []
        for view_meta, view_path in views:
            for qa in qa_pairs:
                sample = self._create_augmented_sample(
                    view_path=view_path,
                    view_metadata=view_meta,
                    qa_pair=qa,
                )
                samples.append(sample)

        return samples

    def _generate_views(
        self,
        image_path: str,
        image_id: str,
    ) -> List[Tuple[ViewMetadata, str]]:
        """Generate views for a single image.

        Returns:
            List of (ViewMetadata, view_path) tuples
        """
        results = []

        # Include original view if requested
        if self.config.include_original and 0.0 in self.config.rotation_angles:
            # Copy original to output directory
            original_path = self._save_original(image_path, image_id)
            original_meta = ViewMetadata(
                view_id=f"{image_id}_view_0.0",
                source_image_id=image_id,
                azimuth=0.0,
                elevation=self.config.elevation,
                is_original=True,
                generation_method="original",
                quality_score=1.0,
            )
            results.append((original_meta, original_path))

        # Generate novel views
        angles_to_generate = [
            a for a in self.config.rotation_angles
            if a != 0.0
        ]

        if angles_to_generate:
            try:
                views = self.mvgen_generator.generate_views_at_angles(
                    image_path,
                    angles=angles_to_generate,
                )

                for angle, view in views.items():
                    if angle == 0.0:
                        continue  # Already handled

                    # Filter by quality
                    if view.quality_score < self.config.quality_threshold:
                        logger.debug(
                            f"Skipping view at {angle} deg due to low quality "
                            f"({view.quality_score:.2f} < {self.config.quality_threshold})"
                        )
                        continue

                    # Save view
                    view_filename = f"{image_id}_view_{angle:+.1f}.jpg"
                    view_path = os.path.join(self.config.output_dir, view_filename)
                    view.save(view_path)

                    # Create metadata
                    view_meta = ViewMetadata(
                        view_id=f"{image_id}_view_{angle}",
                        source_image_id=image_id,
                        azimuth=angle,
                        elevation=view.elevation,
                        is_original=False,
                        generation_method="mvgenmaster",
                        quality_score=view.quality_score,
                    )
                    results.append((view_meta, view_path))

            except Exception as e:
                logger.error(f"Failed to generate views for {image_path}: {e}")

        return results

    def _save_original(self, image_path: str, image_id: str) -> str:
        """Save/copy original image to output directory."""
        original_filename = f"{image_id}_view_+0.0.jpg"
        original_path = os.path.join(self.config.output_dir, original_filename)

        # Load and save (to ensure consistent format)
        img = Image.open(image_path).convert("RGB")
        img.save(original_path, quality=95)

        return original_path

    def _create_augmented_sample(
        self,
        view_path: str,
        view_metadata: ViewMetadata,
        qa_pair: Dict[str, Any],
    ) -> AugmentedSample:
        """Create an augmented sample from a view and QA pair."""
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer_cot", qa_pair.get("answer", ""))

        # Optionally modify question based on view
        if self.config.modify_questions_for_view and not view_metadata.is_original:
            question = self._modify_question_for_view(question, view_metadata)

        # Optionally add view context
        spatial_context = None
        if self.config.add_view_context and not view_metadata.is_original:
            spatial_context = self._create_view_context(view_metadata)

        return AugmentedSample(
            image_path=view_path,
            question=question,
            answer=answer,
            view_metadata=view_metadata,
            original_qa_metadata={
                k: v for k, v in qa_pair.items()
                if k not in ["question", "answer", "answer_cot"]
            },
            spatial_context=spatial_context,
        )

    def _modify_question_for_view(
        self,
        question: str,
        view_metadata: ViewMetadata,
    ) -> str:
        """Modify question to account for view change.

        This can adjust directional references based on the camera rotation.
        """
        # For now, keep questions unchanged
        # Future: implement directional reference adjustment
        return question

    def _create_view_context(self, view_metadata: ViewMetadata) -> str:
        """Create contextual description of the view angle."""
        azimuth = view_metadata.azimuth

        if azimuth > 0:
            direction = "right"
        else:
            direction = "left"

        return (
            f"This image shows the scene from a camera rotated "
            f"{abs(azimuth):.0f} degrees to the {direction}."
        )

    def augment_dataset(
        self,
        qa_pairs_file: str,
        images_dir: str,
        output_qa_file: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[AugmentedSample]:
        """Augment an entire dataset of QA pairs.

        Args:
            qa_pairs_file: Path to JSON file with QA pairs
            images_dir: Directory containing source images
            output_qa_file: Optional path to save augmented QA pairs
            show_progress: Whether to show progress bar

        Returns:
            List of all augmented samples
        """
        # Load QA pairs
        with open(qa_pairs_file, "r") as f:
            qa_pairs = json.load(f)

        # Group by image
        image_qa_map: Dict[str, List[Dict]] = {}
        for qa in qa_pairs:
            image_file = qa.get("image_filename", "")
            if image_file not in image_qa_map:
                image_qa_map[image_file] = []
            image_qa_map[image_file].append(qa)

        # Process each image
        all_samples = []
        iterator = image_qa_map.items()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Augmenting images")

        for image_file, qa_list in iterator:
            image_path = os.path.join(images_dir, image_file)

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            samples = self.augment_image(
                image_path=image_path,
                qa_pairs=qa_list,
            )
            all_samples.extend(samples)

        # Save augmented QA pairs
        if output_qa_file:
            output_data = [s.to_training_format() for s in all_samples]
            os.makedirs(os.path.dirname(output_qa_file) or ".", exist_ok=True)
            with open(output_qa_file, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved {len(output_data)} augmented samples to {output_qa_file}")

        # Save metadata
        if self.config.save_metadata:
            metadata_file = os.path.join(self.config.output_dir, "augmentation_metadata.json")
            metadata = {
                "num_samples": len(all_samples),
                "num_original_images": len(image_qa_map),
                "views_per_image": self.config.views_per_image,
                "rotation_angles": self.config.rotation_angles,
                "config": asdict(self.config),
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(
            f"Generated {len(all_samples)} augmented samples from "
            f"{len(image_qa_map)} images"
        )

        return all_samples


def augment_training_data(
    input_qa_file: str,
    input_images_dir: str,
    output_dir: str,
    config: Optional[AugmentationConfig] = None,
) -> str:
    """Convenience function to augment training data.

    Args:
        input_qa_file: Path to original QA pairs JSON
        input_images_dir: Directory with source images
        output_dir: Directory for augmented output
        config: Augmentation configuration

    Returns:
        Path to augmented QA pairs file
    """
    if config is None:
        config = AugmentationConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    generator = ViewAugmentedDataGenerator(config)

    output_qa_file = os.path.join(output_dir, "augmented_qa_pairs.json")
    generator.augment_dataset(
        qa_pairs_file=input_qa_file,
        images_dir=input_images_dir,
        output_qa_file=output_qa_file,
    )

    return output_qa_file
