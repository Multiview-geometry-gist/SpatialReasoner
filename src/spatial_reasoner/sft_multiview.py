# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-view SFT training for spatial reasoning.

This module extends the standard SFT training to support:
1. Single-image training with randomly selected views
2. Multi-image training with 2 views (original + rotated)
3. View-aware CoT augmentation

Architecture:
    - Modular view selection strategies
    - Configurable CoT augmentation
    - Qwen2.5-VL multi-image support
    - Minimal changes to base SFT code

Usage:
    # Single-view training with random view selection
    accelerate launch src/spatial_reasoner/sft_multiview.py \
        --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_mvgen_single.yaml

    # Multi-view training with two images per sample
    accelerate launch src/spatial_reasoner/sft_multiview.py \
        --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_mvgen_multi.yaml
"""

import logging
import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from PIL import Image
import torch
import datasets
import transformers

# Import system prompt for training/inference consistency
from spatial_reasoner.prompt import SYSTEM_PROMPT

# Fix PyTorch 2.6 weights_only issue for DeepSpeed checkpoint loading
import deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine as _ds_ckpt_engine

_original_load = _ds_ckpt_engine.TorchCheckpointEngine.load

def _patched_load(self, path, map_location=None):
    """Patched load that uses weights_only=False for PyTorch 2.6+ compatibility."""
    import torch
    partition = torch.load(path, map_location=map_location, weights_only=False)
    return partition

_ds_ckpt_engine.TorchCheckpointEngine.load = _patched_load

from datasets import load_dataset, Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed

# Disable torch compile to avoid NCCL timeout issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 1
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from spatial_reasoner.utils.callbacks import get_callbacks, EarlyStoppingCallback
from spatial_reasoner.configs import SFTConfig
from spatial_reasoner.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# Import CoT augmentation module
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_generation.cot_augmentation import CoTAugmentor, CoTAugmentationConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Extended Configuration
# ============================================================================

@dataclass
class MultiViewSFTConfig(SFTConfig):
    """Extended SFT configuration for multi-view training.

    Extends the base SFTConfig with MVGen-specific settings.
    """

    # MVGen augmentation settings
    mvgen_enabled: bool = field(
        default=False,
        metadata={"help": "Enable MVGen multi-view augmentation."},
    )
    mvgen_data_dir: str = field(
        default="./data/mvgen_augmented/",
        metadata={"help": "Directory containing augmented_dataset.json."},
    )
    mvgen_dataset_file: str = field(
        default="augmented_dataset.json",
        metadata={"help": "Dataset JSON filename in mvgen_data_dir."},
    )
    mvgen_mode: str = field(
        default="single",
        metadata={"help": "Mode: 'single' (one view) or 'multi' (two views)."},
    )
    view_selection: str = field(
        default="random",
        metadata={"help": "View selection: 'random', 'fixed_original', 'random_pair'."},
    )
    fixed_view_angle: Optional[float] = field(
        default=None,
        metadata={"help": "Fixed angle for view_selection='fixed_angle'."},
    )
    fixed_view_angles: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Fixed angles for multi-view fixed_pair mode."},
    )

    # Multi-view format settings
    multi_view_format: str = field(
        default="sequential",
        metadata={"help": "Format: 'sequential', 'comparison', 'interleaved'."},
    )
    primary_view_angle: float = field(
        default=0.0,
        metadata={"help": "Angle for primary view (usually original)."},
    )
    secondary_view_selection: str = field(
        default="diverse",
        metadata={"help": "Secondary view: 'diverse', 'random', 'max_rotation'."},
    )

    # CoT augmentation settings
    cot_augmentation_enabled: bool = field(
        default=True,
        metadata={"help": "Enable CoT augmentation with view context."},
    )
    cot_augmentation_strategy: str = field(
        default="prefix",
        metadata={"help": "Strategy: 'prefix', 'inline', 'suffix', 'full'."},
    )
    cot_include_angle: bool = field(
        default=True,
        metadata={"help": "Include angle value in CoT augmentation."},
    )
    cot_angle_precision: int = field(
        default=1,
        metadata={"help": "Decimal precision for angle values."},
    )
    cot_multi_view_format: bool = field(
        default=False,
        metadata={"help": "Use multi-view specific CoT templates."},
    )


# ============================================================================
# View Selection Strategies
# ============================================================================

class ViewSelector:
    """Handles view selection for training samples.

    This class encapsulates the logic for selecting which view(s)
    to use for each training sample based on the configuration.
    """

    def __init__(self, config: MultiViewSFTConfig):
        """Initialize the view selector.

        Args:
            config: Multi-view SFT configuration
        """
        self.config = config
        self.rng = random.Random(config.seed)

    def select_single_view(
        self,
        view_images: Dict[str, str],
    ) -> Tuple[str, float]:
        """Select a single view from available views.

        Args:
            view_images: Dictionary mapping angle_str -> image_path

        Returns:
            Tuple of (image_path, angle)
        """
        if self.config.view_selection == "fixed_original":
            # Always use original view
            angle_str = "0.0"
            if angle_str not in view_images:
                angle_str = list(view_images.keys())[0]

        elif self.config.view_selection == "fixed_angle":
            # Use specific fixed angle
            target = self.config.fixed_view_angle or 0.0
            angle_str = self._find_nearest_angle(view_images, target)

        else:  # random
            # Random selection from available views
            angle_str = self.rng.choice(list(view_images.keys()))

        return view_images[angle_str], float(angle_str)

    def select_view_pair(
        self,
        view_images: Dict[str, str],
    ) -> Tuple[Tuple[str, float], Tuple[str, float]]:
        """Select a pair of views for multi-view training.

        Args:
            view_images: Dictionary mapping angle_str -> image_path

        Returns:
            Tuple of ((primary_path, primary_angle), (secondary_path, secondary_angle))
        """
        # Handle fixed_pair mode with explicit angles
        if (self.config.view_selection == "fixed_pair" and
            self.config.fixed_view_angles is not None and
            len(self.config.fixed_view_angles) >= 2):
            primary_target = self.config.fixed_view_angles[0]
            secondary_target = self.config.fixed_view_angles[1]

            primary_angle_str = self._find_nearest_angle(view_images, primary_target)
            secondary_angle_str = self._find_nearest_angle(view_images, secondary_target)

            primary_path = view_images[primary_angle_str]
            secondary_path = view_images[secondary_angle_str]

            return (primary_path, float(primary_angle_str)), (secondary_path, float(secondary_angle_str))

        # Primary view
        primary_angle_str = self._find_nearest_angle(
            view_images, self.config.primary_view_angle
        )
        primary_path = view_images[primary_angle_str]
        primary_angle = float(primary_angle_str)

        # Secondary view
        remaining = {k: v for k, v in view_images.items() if k != primary_angle_str}

        if not remaining:
            # Only one view available, duplicate it
            return (primary_path, primary_angle), (primary_path, primary_angle)

        if self.config.secondary_view_selection == "diverse":
            # Select view that is most different from primary
            max_diff = 0
            secondary_angle_str = list(remaining.keys())[0]
            for angle_str in remaining:
                diff = abs(float(angle_str) - primary_angle)
                if diff > max_diff:
                    max_diff = diff
                    secondary_angle_str = angle_str

        elif self.config.secondary_view_selection == "max_rotation":
            # Select view with maximum rotation from center
            max_rot = 0
            secondary_angle_str = list(remaining.keys())[0]
            for angle_str in remaining:
                rot = abs(float(angle_str))
                if rot > max_rot:
                    max_rot = rot
                    secondary_angle_str = angle_str

        else:  # random
            secondary_angle_str = self.rng.choice(list(remaining.keys()))

        secondary_path = remaining[secondary_angle_str]
        secondary_angle = float(secondary_angle_str)

        return (primary_path, primary_angle), (secondary_path, secondary_angle)

    def _find_nearest_angle(
        self,
        view_images: Dict[str, str],
        target: float,
    ) -> str:
        """Find the nearest available angle to target.

        Args:
            view_images: Available views
            target: Target angle

        Returns:
            Angle string of nearest view
        """
        return min(
            view_images.keys(),
            key=lambda a: abs(float(a) - target)
        )


# ============================================================================
# Conversation Formatters
# ============================================================================

class ConversationFormatter:
    """Formats conversations for single and multi-view training.

    Handles the conversion of samples to Qwen2.5-VL conversation format,
    supporting both single-image and multi-image inputs.
    """

    def __init__(self, config: MultiViewSFTConfig):
        """Initialize the formatter.

        Args:
            config: Multi-view SFT configuration
        """
        self.config = config

        # Initialize CoT augmentor if enabled
        if config.cot_augmentation_enabled:
            cot_config = CoTAugmentationConfig(
                strategy=config.cot_augmentation_strategy,
                include_angle_value=config.cot_include_angle,
                angle_precision=config.cot_angle_precision,
                seed=config.seed,
            )
            self.cot_augmentor = CoTAugmentor(cot_config)
        else:
            self.cot_augmentor = None

    def format_single_view(
        self,
        question: str,
        options: Optional[Dict[str, str]],
        answer_cot: str,
        image: Image.Image,
        view_angle: float,
        answer: str = "",
    ) -> List[Dict[str, Any]]:
        """Format a single-view conversation.

        Args:
            question: Question text
            options: Answer options (A, B, C, D)
            answer_cot: Chain-of-thought answer
            image: PIL Image
            view_angle: Viewing angle in degrees
            answer: Correct answer letter (A, B, C, D)

        Returns:
            Conversation in Qwen2.5-VL format with system prompt
        """
        # Augment CoT if enabled
        if self.cot_augmentor:
            answer_cot = self.cot_augmentor.augment(answer_cot, view_angle)

        # Wrap answer_cot in <think><answer> tags for consistency with inference
        formatted_response = f"<think>{answer_cot}</think><answer>{answer}</answer>"

        # Format question with options
        question_text = self._format_question(question, options)

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": formatted_response},
                ],
            },
        ]

    def format_multi_view(
        self,
        question: str,
        options: Optional[Dict[str, str]],
        answer_cot: str,
        primary_image: Image.Image,
        primary_angle: float,
        secondary_image: Image.Image,
        secondary_angle: float,
        answer: str = "",
    ) -> List[Dict[str, Any]]:
        """Format a multi-view conversation with two images.

        Args:
            question: Question text
            options: Answer options
            answer_cot: Chain-of-thought answer
            primary_image: Primary view image
            primary_angle: Primary view angle
            secondary_image: Secondary view image
            secondary_angle: Secondary view angle
            answer: Correct answer letter (A, B, C, D)

        Returns:
            Conversation in Qwen2.5-VL format with system prompt
        """
        # Augment CoT for multi-view if enabled
        if self.cot_augmentor:
            # Use primary angle for augmentation
            # For multi-view, we could extend CoT augmentor to handle both angles
            answer_cot = self._augment_multiview_cot(
                answer_cot, primary_angle, secondary_angle
            )

        # Wrap answer_cot in <think><answer> tags for consistency with inference
        formatted_response = f"<think>{answer_cot}</think><answer>{answer}</answer>"

        # Format question with options
        question_text = self._format_question(question, options)

        # Build multi-view prompt based on format
        if self.config.multi_view_format == "comparison":
            prompt_text = (
                f"Compare these two views of the same scene:\n"
                f"View 1 ({primary_angle:.0f} degrees): [First image]\n"
                f"View 2 ({secondary_angle:.0f} degrees): [Second image]\n\n"
                f"{question_text}"
            )
        elif self.config.multi_view_format == "interleaved":
            prompt_text = (
                f"Looking at the scene from {primary_angle:.0f} degrees "
                f"[First image], then from {secondary_angle:.0f} degrees "
                f"[Second image].\n\n{question_text}"
            )
        else:  # sequential (default)
            prompt_text = (
                f"View 1 (rotation: {primary_angle:.0f} degrees):\n"
                f"View 2 (rotation: {secondary_angle:.0f} degrees):\n\n"
                f"{question_text}"
            )

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": primary_image},
                    {"type": "image", "image": secondary_image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": formatted_response},
                ],
            },
        ]

    def _format_question(
        self,
        question: str,
        options: Optional[Dict[str, str]],
    ) -> str:
        """Format question with options.

        Args:
            question: Question text
            options: Answer options dict

        Returns:
            Formatted question string
        """
        if options and options.get("A"):
            opts_text = "\n".join(
                f"{opt}. {options[opt]}"
                for opt in ["A", "B", "C", "D"]
                if options.get(opt)
            )
            return (
                f"Question: {question}\n"
                f"Options:\n{opts_text}\n"
                f"Please select the correct answer from the options above."
            )
        return question

    def _augment_multiview_cot(
        self,
        answer_cot: str,
        primary_angle: float,
        secondary_angle: float,
    ) -> str:
        """Augment CoT for multi-view context.

        Args:
            answer_cot: Original CoT
            primary_angle: Primary view angle
            secondary_angle: Secondary view angle

        Returns:
            Augmented CoT
        """
        if not self.cot_augmentor:
            return answer_cot

        # Add multi-view context
        angle_diff = abs(secondary_angle - primary_angle)
        context = (
            f"Analyzing the scene from two viewpoints "
            f"({primary_angle:.0f} and {secondary_angle:.0f} degrees, "
            f"a {angle_diff:.0f} degree difference): "
        )

        # Use base augmentor for the primary view
        augmented = self.cot_augmentor.augment(answer_cot, primary_angle)

        # Prepend multi-view context
        return context + augmented


# ============================================================================
# Dataset Loading
# ============================================================================

class MVGenDatasetLoader:
    """Loads and prepares MVGen-augmented datasets.

    Handles loading from:
    - Local augmented dataset JSON
    - HuggingFace datasets with MVGen augmentation
    """

    def __init__(self, config: MultiViewSFTConfig):
        """Initialize the loader.

        Args:
            config: Multi-view SFT configuration
        """
        self.config = config

    def load(self) -> Dataset:
        """Load the MVGen-augmented dataset.

        Returns:
            HuggingFace Dataset object
        """
        if self.config.mvgen_enabled:
            return self._load_mvgen_dataset()
        else:
            # Fall back to standard dataset loading
            return self._load_standard_dataset()

    def _load_mvgen_dataset(self) -> Dataset:
        """Load MVGen-augmented dataset from local directory.

        Returns:
            HuggingFace Dataset
        """
        augmented_json = Path(self.config.mvgen_data_dir) / self.config.mvgen_dataset_file

        if not augmented_json.exists():
            raise FileNotFoundError(
                f"Augmented dataset not found at {augmented_json}. "
                f"Run mvgen_bulk_augmentation.py first."
            )

        with open(augmented_json, "r") as f:
            samples = json.load(f)

        logger.info(f"Loaded {len(samples)} samples from {augmented_json}")

        return Dataset.from_list(samples)

    def _load_standard_dataset(self) -> Dataset:
        """Load standard HuggingFace dataset.

        Returns:
            HuggingFace Dataset
        """
        return load_dataset(
            self.config.dataset_name,
            name=self.config.dataset_config if hasattr(self.config, 'dataset_config') else None,
        )


# ============================================================================
# Collator
# ============================================================================

def create_multiview_collate_fn(
    config: MultiViewSFTConfig,
    processor: Qwen2_5_VLProcessor,
    view_selector: ViewSelector,
    formatter: ConversationFormatter,
):
    """Create a collate function for multi-view training.

    Args:
        config: Multi-view SFT configuration
        processor: Qwen2.5-VL processor
        view_selector: View selection handler
        formatter: Conversation formatter

    Returns:
        Collate function for DataLoader
    """

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function that handles view selection and formatting.

        Args:
            examples: List of sample dictionaries

        Returns:
            Batched tensors for training
        """
        samples = []

        for example in examples:
            try:
                if example.get('question'):
                    # Prepare options
                    options = {
                        opt: example.get(opt, "")
                        for opt in ["A", "B", "C", "D"]
                    }

                    # Get view images - support both expanded and standard datasets
                    # For expanded dataset: use image_path directly
                    # For standard dataset: use view_images dict
                    direct_image_path = example.get("image_path")
                    view_angle = example.get("view_angle", 0.0)

                    if direct_image_path:
                        # Expanded dataset with pre-selected single view
                        view_images = {str(view_angle): direct_image_path}
                    else:
                        view_images = example.get("view_images", {})
                        # Filter out None values from view_images
                        view_images = {k: v for k, v in view_images.items() if v is not None}

                    # Handle fallback to original image
                    if not view_images:
                        image_path = os.path.join(
                            config.data_dir, example.get("image_filename", "")
                        )
                        view_images = {"0.0": image_path}

                    if config.mvgen_mode == "multi":
                        # Multi-view mode: select two images
                        (primary_path, primary_angle), (secondary_path, secondary_angle) = \
                            view_selector.select_view_pair(view_images)

                        # Validate paths - skip if primary is None
                        if primary_path is None:
                            logger.warning(f"Skipping sample: primary_path is None, view_images={view_images}")
                            continue

                        # Fallback: if secondary_path is None, use primary
                        if secondary_path is None:
                            secondary_path = primary_path
                            secondary_angle = primary_angle
                            logger.debug(f"Fallback: secondary_path was None, using primary")

                        # Resolve paths
                        primary_full = _resolve_image_path(
                            primary_path, config.mvgen_data_dir, config.data_dir
                        )
                        secondary_full = _resolve_image_path(
                            secondary_path, config.mvgen_data_dir, config.data_dir
                        )

                        # Load images with error handling
                        try:
                            primary_image = Image.open(primary_full).convert("RGB")
                            secondary_image = Image.open(secondary_full).convert("RGB")
                        except (IOError, OSError) as e:
                            logger.warning(f"Failed to load images for multi-view: {e}")
                            continue

                        # Get augmented CoT for the primary view angle
                        augmented_cots = example.get("augmented_cots", {})
                        primary_angle_str = f"{primary_angle:.1f}".lstrip("+")
                        if augmented_cots and primary_angle_str in augmented_cots:
                            # Use pre-transformed CoT
                            answer_cot = augmented_cots[primary_angle_str]
                        else:
                            answer_cot = example.get("answer_cot", example.get("answer", ""))

                        conversation = formatter.format_multi_view(
                            question=example["question"],
                            options=options,
                            answer_cot=answer_cot,
                            primary_image=primary_image,
                            primary_angle=primary_angle,
                            secondary_image=secondary_image,
                            secondary_angle=secondary_angle,
                            answer=example.get("answer", ""),
                        )

                    else:
                        # Single-view mode
                        image_path, view_angle = view_selector.select_single_view(view_images)

                        full_path = _resolve_image_path(
                            image_path, config.mvgen_data_dir, config.data_dir
                        )

                        # Load image with error handling
                        try:
                            image = Image.open(full_path).convert("RGB")
                        except (IOError, OSError) as e:
                            logger.warning(f"Failed to load image {full_path}: {e}")
                            continue

                        # Get the augmented CoT for this specific view angle
                        # If augmented_cots exists, use the pre-transformed CoT
                        augmented_cots = example.get("augmented_cots", {})
                        angle_str = f"{view_angle:.1f}".lstrip("+")
                        if augmented_cots and angle_str in augmented_cots:
                            # Use pre-transformed CoT (coordinates already rotated)
                            answer_cot = augmented_cots[angle_str]
                        else:
                            # Fallback to original CoT
                            answer_cot = example.get("answer_cot", example.get("answer", ""))

                        conversation = formatter.format_single_view(
                            question=example["question"],
                            options=options,
                            answer_cot=answer_cot,
                            image=image,
                            view_angle=view_angle,
                            answer=example.get("answer", ""),
                        )

                    samples.append(conversation)

                else:
                    # Handle LLaVA-style conversations
                    conversation = _convert_llava_conversation(
                        example, config.llava_dir
                    )
                    samples.append(conversation)

            except Exception as e:
                import traceback
                logger.warning(f"Error processing example: {e}\n{traceback.format_exc()}")
                continue

        if not samples:
            # Return None to signal DataLoader to skip this batch
            # This prevents trainer crash from empty dict
            logger.warning("Empty batch - all samples failed processing")
            return None

        # Apply chat template and tokenize
        batch = processor.apply_chat_template(
            samples,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_length,
            truncation=True,
        )

        # Ensure consistent padding
        max_len = config.max_length
        pad_token_id = processor.tokenizer.pad_token_id

        for key in ["input_ids", "attention_mask"]:
            if key in batch:
                current_len = batch[key].shape[1]
                if current_len < max_len:
                    pad_value = pad_token_id if key == "input_ids" else 0
                    padding = torch.full(
                        (batch[key].shape[0], max_len - current_len),
                        pad_value,
                        dtype=batch[key].dtype,
                        device=batch[key].device,
                    )
                    batch[key] = torch.cat([batch[key], padding], dim=1)
                elif current_len > max_len:
                    batch[key] = batch[key][:, :max_len]

        # Create labels
        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100

        # Mask image tokens for Qwen2.5-VL
        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    return collate_fn


def _resolve_image_path(
    image_path: str,
    mvgen_data_dir: str,
    fallback_data_dir: str,
) -> str:
    """Resolve image path with security validation.

    Args:
        image_path: Relative or absolute image path
        mvgen_data_dir: MVGen augmented data directory
        fallback_data_dir: Fallback data directory

    Returns:
        Resolved absolute path

    Raises:
        FileNotFoundError: If image not found
        ValueError: If path traversal detected or path outside allowed directories
    """
    # Check for None path
    if image_path is None:
        raise ValueError("Image path is None")

    # Prevent path traversal attacks
    if ".." in image_path:
        raise ValueError(f"Path traversal detected: {image_path}")

    resolved = None

    if os.path.isabs(image_path) and os.path.exists(image_path):
        resolved = image_path
    else:
        # Try MVGen directory first
        mvgen_path = os.path.join(mvgen_data_dir, "images", image_path)
        if os.path.exists(mvgen_path):
            resolved = mvgen_path
        else:
            # Try direct join with MVGen dir
            mvgen_path = os.path.join(mvgen_data_dir, image_path)
            if os.path.exists(mvgen_path):
                resolved = mvgen_path
            else:
                # Fallback to original data directory
                fallback_path = os.path.join(fallback_data_dir, image_path)
                if os.path.exists(fallback_path):
                    resolved = fallback_path

    if resolved is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Validate path is within allowed directories
    resolved_abs = os.path.abspath(resolved)
    allowed_dirs = [
        os.path.abspath(mvgen_data_dir),
        os.path.abspath(fallback_data_dir),
    ]

    if not any(resolved_abs.startswith(allowed_dir) for allowed_dir in allowed_dirs):
        raise ValueError(f"Image path outside allowed directories: {image_path}")

    return resolved_abs


def _convert_llava_conversation(
    example: Dict[str, Any],
    llava_dir: str,
) -> List[Dict[str, Any]]:
    """Convert LLaVA-style conversation to Qwen2.5-VL format.

    Args:
        example: LLaVA sample dictionary
        llava_dir: LLaVA data directory

    Returns:
        Conversation in Qwen2.5-VL format
    """
    conversation = []

    for turn in example.get("conversations", []):
        role = "user" if turn["from"] == "human" else "assistant"

        if "<image>" in turn["value"]:
            image_path = os.path.join(llava_dir, example.get("image_path", ""))
            image = Image.open(image_path).convert("RGB")
            text = turn["value"].replace("<image>", "").strip()

            conversation.append({
                "role": role,
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            })
        else:
            conversation.append({
                "role": role,
                "content": [
                    {"type": "text", "text": turn["value"]},
                ],
            })

    return conversation


# ============================================================================
# Main Training Function
# ============================================================================

def main(script_args, training_args, model_args):
    """Main training function.

    Args:
        script_args: Script arguments
        training_args: Training configuration (MultiViewSFTConfig)
        model_args: Model configuration
    """
    # Set seed
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, "
        f"16-bits: {training_args.fp16}"
    )
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")

    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Initialize WandB
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load dataset
    logger.info("*** Loading dataset ***")
    if training_args.mvgen_enabled:
        loader = MVGenDatasetLoader(training_args)
        dataset = loader.load()
        # Wrap in dict format expected by trainer
        dataset = {"train": dataset}
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
        )

    # Initialize model
    logger.info("*** Initializing model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    processor.image_processor.max_pixels = training_args.max_pixels
    processor.image_processor.min_pixels = training_args.min_pixels

    # Initialize view selection and formatting
    view_selector = ViewSelector(training_args)
    formatter = ConversationFormatter(training_args)

    # Create collate function
    collate_fn = create_multiview_collate_fn(
        config=training_args,
        processor=processor,
        view_selector=view_selector,
        formatter=formatter,
    )

    # Initialize trainer
    logger.info("*** Initializing trainer ***")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset.get("val") if training_args.eval_strategy != "no" else None,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        processing_class=processor.tokenizer,
        callbacks=get_callbacks(training_args, model_args) + [
            EarlyStoppingCallback(stop_step=training_args.stop_steps)
        ],
    )

    # Train
    logger.info("*** Starting training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model
    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save on main process
    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": script_args.dataset_name,
        "tags": ["SpatialReasoner", "MVGen", "MultiView"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Push to hub
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, MultiViewSFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if model_args.use_peft:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    main(script_args, training_args, model_args)
