"""Multi-View Dataset Module for SpatialReasoner (Approach 2).

Implements multi-image training and inference strategy:
- Train model with multiple views as input simultaneously
- Support Qwen2.5-VL multi-image input format
- During inference, generate views with MVGenMaster and feed to model

This approach leverages the VLM's ability to reason across multiple
images simultaneously for improved spatial understanding.

Example:
    from data_generation.multiview_dataset import (
        MultiViewDatasetBuilder,
        MultiViewConfig,
        MultiViewPromptFormatter,
    )

    config = MultiViewConfig(num_views=3)
    builder = MultiViewDatasetBuilder(config)

    # Build multi-view dataset from single-view data
    builder.build_dataset(
        input_qa_file="qa_pairs.json",
        input_images_dir="images/",
        output_dir="multiview_data/",
    )

    # Format prompt for Qwen2.5-VL
    formatter = MultiViewPromptFormatter()
    messages = formatter.format_multiview_prompt(
        images=["view1.jpg", "view2.jpg", "view3.jpg"],
        question="What is to the left of the chair?",
        view_descriptions=["Front view", "Left 15 degrees", "Right 15 degrees"],
    )
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import numpy as np

from .mvgenmaster_integration import (
    MVGenMasterGenerator,
    MVGenConfig,
    GeneratedView,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiViewSample:
    """Container for a multi-view training sample.

    Attributes:
        sample_id: Unique identifier for this sample
        view_images: List of image paths (ordered by angle)
        view_angles: List of azimuth angles for each view
        question: Question text
        answer: Answer text (with chain-of-thought)
        primary_view_idx: Index of the primary (reference) view
        metadata: Additional sample metadata
    """
    sample_id: str
    view_images: List[str]
    view_angles: List[float]
    question: str
    answer: str
    primary_view_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        """Number of views in this sample."""
        return len(self.view_images)

    @property
    def primary_view(self) -> str:
        """Path to primary view image."""
        return self.view_images[self.primary_view_idx]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sample_id": self.sample_id,
            "view_images": self.view_images,
            "view_angles": self.view_angles,
            "question": self.question,
            "answer": self.answer,
            "primary_view_idx": self.primary_view_idx,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiViewSample":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class MultiViewConfig:
    """Configuration for multi-view dataset building.

    Attributes:
        num_views: Number of views per sample
        view_angles: Specific view angles to include
        include_original: Whether to include the original view
        symmetric_views: Whether to use symmetric left/right angles
        max_angle: Maximum azimuth angle for views
        view_selection: Strategy for selecting views ("uniform", "centered", "random")
        output_dir: Directory for output data
        image_format: Output image format ("jpg", "png")
        mvgenmaster_config: Configuration for MVGenMaster
    """
    # View settings
    num_views: int = 3
    view_angles: Optional[List[float]] = None
    include_original: bool = True
    symmetric_views: bool = True
    max_angle: float = 15.0
    view_selection: str = "centered"  # "uniform", "centered", "random"

    # Output settings
    output_dir: str = "./multiview_dataset"
    image_format: str = "jpg"

    # MVGenMaster config
    mvgenmaster_config: Optional[MVGenConfig] = None

    def get_view_angles(self) -> List[float]:
        """Get view angles based on configuration."""
        if self.view_angles is not None:
            return sorted(self.view_angles)

        if self.view_selection == "centered":
            # Center-weighted angles with 0 in middle
            if self.num_views == 1:
                return [0.0]
            elif self.num_views == 2:
                return [-self.max_angle, self.max_angle]
            elif self.num_views == 3:
                return [-self.max_angle, 0.0, self.max_angle]
            else:
                # Generate symmetric angles
                half_n = (self.num_views - 1) // 2
                step = self.max_angle / half_n if half_n > 0 else self.max_angle
                angles = []
                for i in range(-half_n, half_n + 1):
                    angles.append(i * step)
                if self.num_views % 2 == 0:
                    angles = angles[:-1]  # Remove last if even
                return sorted(set(angles))

        elif self.view_selection == "uniform":
            # Uniform spacing across range
            angles = np.linspace(-self.max_angle, self.max_angle, self.num_views)
            return sorted(angles.tolist())

        else:
            raise ValueError(f"Unknown view_selection: {self.view_selection}")

    def get_mvgenmaster_config(self) -> MVGenConfig:
        """Get or create MVGenMaster configuration."""
        if self.mvgenmaster_config is not None:
            return self.mvgenmaster_config

        angles = self.get_view_angles()
        max_angle = max(abs(a) for a in angles) if angles else self.max_angle

        return MVGenConfig(
            num_views=len(angles),
            azimuth_range=max_angle * 1.2,  # Slightly wider for safety
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MultiViewConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested structure
        if "multiview" in config_dict:
            config_dict = config_dict["multiview"]

        # Handle mvgenmaster_config separately
        mvgen_dict = config_dict.pop("mvgenmaster_config", None)
        if mvgen_dict:
            config_dict["mvgenmaster_config"] = MVGenConfig.from_dict(mvgen_dict)

        return cls(**config_dict)


class MultiViewPromptFormatter:
    """Formats multi-view prompts for Qwen2.5-VL.

    Handles the proper formatting of multi-image inputs for VLM processing,
    including view descriptions and spatial context.

    Example:
        formatter = MultiViewPromptFormatter()

        # Format for training
        messages = formatter.format_multiview_prompt(
            images=["view1.jpg", "view2.jpg", "view3.jpg"],
            question="What is to the left of the chair?",
            view_descriptions=["Front view", "Left 15 deg", "Right 15 deg"],
        )

        # Format for inference (with PIL Images)
        messages = formatter.format_for_processor(
            images=[pil_img1, pil_img2, pil_img3],
            question="Where is the table relative to the lamp?",
        )
    """

    # Default view description templates
    DEFAULT_VIEW_DESCRIPTIONS = {
        0.0: "front view",
        -5.0: "slightly left view",
        5.0: "slightly right view",
        -10.0: "left view (10 degrees)",
        10.0: "right view (10 degrees)",
        -15.0: "left view (15 degrees)",
        15.0: "right view (15 degrees)",
        -30.0: "left view (30 degrees)",
        30.0: "right view (30 degrees)",
    }

    def __init__(
        self,
        include_view_descriptions: bool = True,
        view_description_template: str = "Image {idx}: {description}",
        spatial_reasoning_prompt: Optional[str] = None,
    ):
        """Initialize the formatter.

        Args:
            include_view_descriptions: Whether to include descriptions of each view
            view_description_template: Template for view descriptions
            spatial_reasoning_prompt: Optional custom prompt for spatial reasoning
        """
        self.include_view_descriptions = include_view_descriptions
        self.view_description_template = view_description_template
        self.spatial_reasoning_prompt = spatial_reasoning_prompt or (
            "You are viewing a scene from multiple angles. "
            "Use all views to reason about spatial relationships."
        )

    def get_view_description(self, angle: float) -> str:
        """Get description for a specific view angle."""
        if angle in self.DEFAULT_VIEW_DESCRIPTIONS:
            return self.DEFAULT_VIEW_DESCRIPTIONS[angle]

        # Generate description for non-standard angles
        if angle == 0:
            return "front view"
        elif angle < 0:
            return f"left view ({abs(angle):.0f} degrees)"
        else:
            return f"right view ({angle:.0f} degrees)"

    def format_multiview_prompt(
        self,
        images: List[Union[str, Image.Image]],
        question: str,
        view_angles: Optional[List[float]] = None,
        view_descriptions: Optional[List[str]] = None,
        include_system_prompt: bool = True,
    ) -> List[Dict[str, Any]]:
        """Format a multi-view prompt for Qwen2.5-VL.

        Args:
            images: List of image paths or PIL Images
            question: Question text
            view_angles: Optional list of view angles (used for auto-description)
            view_descriptions: Optional custom view descriptions
            include_system_prompt: Whether to include system prompt

        Returns:
            List of message dictionaries in Qwen2.5-VL format
        """
        messages = []

        # Generate view descriptions if not provided
        if view_descriptions is None and view_angles is not None:
            view_descriptions = [
                self.get_view_description(angle)
                for angle in view_angles
            ]

        # Build user content
        user_content = []

        # Add view description preamble
        if self.include_view_descriptions and view_descriptions:
            description_text = self.spatial_reasoning_prompt + "\n\n"
            description_text += "Views provided:\n"
            for idx, desc in enumerate(view_descriptions, 1):
                description_text += self.view_description_template.format(
                    idx=idx,
                    description=desc,
                )
                description_text += "\n"
            user_content.append({"type": "text", "text": description_text})

        # Add images
        for img in images:
            if isinstance(img, str):
                # Image path
                user_content.append({"type": "image", "image": img})
            else:
                # PIL Image
                user_content.append({"type": "image", "image": img})

        # Add question
        user_content.append({"type": "text", "text": f"\nQuestion: {question}"})

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return messages

    def format_for_training(
        self,
        sample: MultiViewSample,
        images_dir: str = "",
    ) -> Dict[str, Any]:
        """Format a multi-view sample for training.

        Returns format compatible with Qwen2.5-VL processor.

        Args:
            sample: MultiViewSample to format
            images_dir: Base directory for image paths

        Returns:
            Training sample dictionary
        """
        # Build messages
        messages = self.format_multiview_prompt(
            images=[os.path.join(images_dir, img) for img in sample.view_images],
            question=sample.question,
            view_angles=sample.view_angles,
        )

        # Add assistant response
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample.answer}],
        })

        return {
            "sample_id": sample.sample_id,
            "messages": messages,
            "view_images": sample.view_images,
            "view_angles": sample.view_angles,
            "metadata": sample.metadata,
        }

    def format_for_processor(
        self,
        images: List[Image.Image],
        question: str,
        view_angles: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Format for direct use with Qwen2.5-VL processor.

        This format is ready to be passed to processor.apply_chat_template().

        Args:
            images: List of PIL Images
            question: Question text
            view_angles: Optional list of view angles

        Returns:
            Messages in processor-compatible format
        """
        return self.format_multiview_prompt(
            images=images,
            question=question,
            view_angles=view_angles,
        )


class MultiViewDatasetBuilder:
    """Builds multi-view datasets from single-view training data.

    This class implements Approach 2: Multi-Image Training and Inference.
    It generates multiple views for each training sample and creates
    datasets where each sample contains multiple images.

    Example:
        config = MultiViewConfig(num_views=3, max_angle=15.0)
        builder = MultiViewDatasetBuilder(config)

        # Build dataset from existing QA pairs
        builder.build_dataset(
            input_qa_file="original_qa.json",
            input_images_dir="images/",
            output_dir="multiview_data/",
        )
    """

    def __init__(self, config: Optional[MultiViewConfig] = None):
        """Initialize the builder.

        Args:
            config: Multi-view configuration. Uses defaults if not provided.
        """
        self.config = config or MultiViewConfig()
        self._mvgen_generator = None
        self.formatter = MultiViewPromptFormatter()

    @property
    def mvgen_generator(self) -> MVGenMasterGenerator:
        """Lazy initialization of MVGenMaster generator."""
        if self._mvgen_generator is None:
            mvgen_config = self.config.get_mvgenmaster_config()
            self._mvgen_generator = MVGenMasterGenerator(mvgen_config)
        return self._mvgen_generator

    def build_sample(
        self,
        image_path: str,
        question: str,
        answer: str,
        sample_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MultiViewSample]:
        """Build a multi-view sample from a single image.

        Args:
            image_path: Path to source image
            question: Question text
            answer: Answer text
            sample_id: Sample identifier
            metadata: Optional additional metadata

        Returns:
            MultiViewSample or None if generation fails
        """
        view_angles = self.config.get_view_angles()

        # Generate views at specified angles
        try:
            views = self.mvgen_generator.generate_views_at_angles(
                image_path,
                angles=view_angles,
            )
        except Exception as e:
            logger.error(f"Failed to generate views for {image_path}: {e}")
            return None

        if not views:
            logger.warning(f"No views generated for {image_path}")
            return None

        # Save views and collect paths
        view_images = []
        final_angles = []

        for angle in view_angles:
            if angle not in views:
                continue

            view = views[angle]

            # Save view image
            view_filename = f"{sample_id}_view_{angle:+.1f}.{self.config.image_format}"
            view_path = os.path.join(self.config.output_dir, "images", view_filename)
            os.makedirs(os.path.dirname(view_path), exist_ok=True)

            view.save(view_path, quality=95 if self.config.image_format == "jpg" else None)

            view_images.append(view_filename)
            final_angles.append(angle)

        if len(view_images) < 2:
            logger.warning(f"Not enough valid views for {sample_id}")
            return None

        # Find primary view (closest to 0)
        primary_idx = min(range(len(final_angles)), key=lambda i: abs(final_angles[i]))

        return MultiViewSample(
            sample_id=sample_id,
            view_images=view_images,
            view_angles=final_angles,
            question=question,
            answer=answer,
            primary_view_idx=primary_idx,
            metadata=metadata or {},
        )

    def build_dataset(
        self,
        input_qa_file: str,
        input_images_dir: str,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[MultiViewSample]:
        """Build a complete multi-view dataset.

        Args:
            input_qa_file: Path to JSON file with original QA pairs
            input_images_dir: Directory containing source images
            output_dir: Output directory (uses config if not provided)
            show_progress: Whether to show progress bar

        Returns:
            List of MultiViewSample objects
        """
        if output_dir:
            self.config.output_dir = output_dir

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "images"), exist_ok=True)

        # Load original QA pairs
        with open(input_qa_file, "r") as f:
            qa_pairs = json.load(f)

        # Group by image
        image_qa_map: Dict[str, List[Dict]] = {}
        for qa in qa_pairs:
            image_file = qa.get("image_filename", "")
            if image_file not in image_qa_map:
                image_qa_map[image_file] = []
            image_qa_map[image_file].append(qa)

        # Build samples
        all_samples = []

        from tqdm import tqdm
        iterator = image_qa_map.items()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Building multi-view samples")

        for image_file, qa_list in iterator:
            image_path = os.path.join(input_images_dir, image_file)

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            for i, qa in enumerate(qa_list):
                sample_id = f"{Path(image_file).stem}_q{i}"
                question = qa.get("question", "")
                answer = qa.get("answer_cot", qa.get("answer", ""))

                sample = self.build_sample(
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    sample_id=sample_id,
                    metadata={
                        "source_image": image_file,
                        "original_qa": qa,
                    },
                )

                if sample is not None:
                    all_samples.append(sample)

        # Save dataset
        self._save_dataset(all_samples)

        logger.info(
            f"Built {len(all_samples)} multi-view samples from "
            f"{len(image_qa_map)} images"
        )

        return all_samples

    def _save_dataset(self, samples: List[MultiViewSample]) -> None:
        """Save the dataset to disk."""
        # Save samples as JSON
        samples_file = os.path.join(self.config.output_dir, "multiview_samples.json")
        samples_data = [s.to_dict() for s in samples]

        with open(samples_file, "w") as f:
            json.dump(samples_data, f, indent=2)

        # Save training format (for Qwen2.5-VL)
        training_file = os.path.join(self.config.output_dir, "training_data.json")
        training_data = [
            self.formatter.format_for_training(
                s,
                images_dir=os.path.join(self.config.output_dir, "images"),
            )
            for s in samples
        ]

        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=2)

        # Save metadata
        metadata = {
            "num_samples": len(samples),
            "num_views_per_sample": self.config.num_views,
            "view_angles": self.config.get_view_angles(),
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else {},
        }

        metadata_file = os.path.join(self.config.output_dir, "dataset_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset to {self.config.output_dir}")


def load_multiview_dataset(dataset_dir: str) -> List[MultiViewSample]:
    """Load a multi-view dataset from disk.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        List of MultiViewSample objects
    """
    samples_file = os.path.join(dataset_dir, "multiview_samples.json")

    with open(samples_file, "r") as f:
        samples_data = json.load(f)

    return [MultiViewSample.from_dict(s) for s in samples_data]


def create_multiview_collate_fn(
    processor,
    images_dir: str,
    max_length: int = 8192,
):
    """Create a collate function for multi-view training.

    This creates a collate function compatible with SFTTrainer for
    multi-view samples.

    Args:
        processor: Qwen2.5-VL processor
        images_dir: Directory containing view images
        max_length: Maximum sequence length

    Returns:
        Collate function for DataLoader
    """
    import torch
    formatter = MultiViewPromptFormatter()

    def collate_fn(examples):
        samples = []

        for example in examples:
            # Handle MultiViewSample format
            if "view_images" in example:
                # Load images
                images = []
                for img_path in example["view_images"]:
                    full_path = os.path.join(images_dir, img_path)
                    img = Image.open(full_path).convert("RGB")
                    images.append(img)

                # Format conversation
                messages = formatter.format_multiview_prompt(
                    images=images,
                    question=example["question"],
                    view_angles=example.get("view_angles"),
                )

                # Add assistant response
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["answer"]}],
                })

                samples.append(messages)
            else:
                # Fallback to single-image format
                # ... (handle standard format)
                pass

        # Apply chat template
        batch = processor.apply_chat_template(
            samples,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Handle labels
        labels = batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        # Mask image tokens
        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    return collate_fn
