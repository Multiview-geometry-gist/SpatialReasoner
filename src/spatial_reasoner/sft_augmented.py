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

"""
SFT Training with Augmented HuggingFace Datasets
Supports CoT coordinate transformation for rotated view datasets.
"""

import logging
import os
import sys
import re
import math
import numpy as np
from PIL import Image
import torch
import datasets
import transformers

# Fix PyTorch 2.6 weights_only issue for DeepSpeed checkpoint loading
import deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine as _ds_ckpt_engine

_original_load = _ds_ckpt_engine.TorchCheckpointEngine.load

def _patched_load(self, path, map_location=None):
    """Patched load that uses weights_only=False for PyTorch 2.6+ compatibility."""
    import torch
    partition = torch.load(path, map_location=map_location, weights_only=False)
    return partition

_ds_ckpt_engine.TorchCheckpointEngine.load = _patched_load
from datasets import load_dataset, concatenate_datasets
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed

# Import system prompt for training/inference consistency
from spatial_reasoner.prompt import SYSTEM_PROMPT

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


logger = logging.getLogger(__name__)


# ============================================================================
# CoT Coordinate Transformation
# ============================================================================

def format_coord(value: float, precision: int = 1) -> str:
    """Format coordinate value with proper precision."""
    if abs(value) < 0.05:
        return "-0.0" if value < 0 else "0.0"
    return f"{value:.{precision}f}"


def format_coord_tuple(coords: list, precision: int = 1) -> str:
    """Format a list of coordinates as a tuple string."""
    formatted = [format_coord(c, precision) for c in coords]
    return f"({', '.join(formatted)})"


def compute_angle_degrees(vec1: list, vec2: list) -> float:
    """Compute angle between two vectors in degrees."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 90.0

    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2

    # Compute cosine similarity
    cos_sim = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle_rad = math.acos(cos_sim)

    return math.degrees(angle_rad)


def transform_cot_coordinates(
    cot: str,
    bounding_boxes: list,
    directions: list,
) -> str:
    """Transform CoT text to use rotated coordinates from bounding_box and direction fields.

    Handles ALL objects in multi-object scenarios, not just the first one.

    Args:
        cot: Original chain-of-thought text
        bounding_boxes: List of bounding box dicts with 'bbox_3d' and 'label'
        directions: List of direction dicts with 'front_dir', 'left_dir', 'label'

    Returns:
        Transformed CoT with updated coordinates for all objects
    """
    if not bounding_boxes:
        return cot

    transformed = cot

    # Helper for cosine similarity
    def cos_sim(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return np.dot(v1/n1, v2/n2)

    # Build direction lookup by label
    direction_by_label = {}
    for d in directions:
        label = d.get('label', '')
        if label:
            direction_by_label[label] = d

    # Process ALL bounding boxes (for multi-object categories)
    for bbox in bounding_boxes:
        label = bbox.get('label', '')
        bbox_3d = bbox.get('bbox_3d', [0, 0, 0])

        if not label:
            continue

        # Escape special regex characters in label
        escaped_label = re.escape(label)

        # Pattern: "The 3D location of X is (x, y, z)" or "The location of X is (x, y, z)"
        loc_pattern = rf"The (?:3D )?location of {escaped_label} is \([^)]+\)"
        loc_replacement = f"The 3D location of {label} is {format_coord_tuple(bbox_3d)}"
        transformed = re.sub(loc_pattern, loc_replacement, transformed)

        # Pattern: "vector from X to camera is hence (x, y, z)"
        camera_vec = [-bbox_3d[0], -bbox_3d[1], -bbox_3d[2]]
        vec_pattern = rf"(vector from {escaped_label} to camera is hence )\([^)]+\)"
        vec_replacement = f"\\1{format_coord_tuple(camera_vec)}"
        transformed = re.sub(vec_pattern, vec_replacement, transformed)

        # Pattern: "The vector from X to Y is (x, y, z)" - compute from positions
        # This handles multi-object distance/direction calculations
        for other_bbox in bounding_boxes:
            other_label = other_bbox.get('label', '')
            if other_label and other_label != label:
                other_coords = other_bbox.get('bbox_3d', [0, 0, 0])
                vec_to_other = [
                    other_coords[0] - bbox_3d[0],
                    other_coords[1] - bbox_3d[1],
                    other_coords[2] - bbox_3d[2]
                ]
                escaped_other = re.escape(other_label)
                vec_pattern = rf"(The vector from {escaped_label} to {escaped_other} is )\([^)]+\)"
                vec_replacement = f"\\1{format_coord_tuple(vec_to_other)}"
                transformed = re.sub(vec_pattern, vec_replacement, transformed)

        # Process direction if available for this object
        direction = direction_by_label.get(label, {})
        if direction:
            front_dir = direction.get('front_dir', [0, 0, -1])
            left_dir = direction.get('left_dir', [-1, 0, 0])

            # Pattern: "The left direction of X is (x, y, z)"
            left_pattern = rf"The left direction of {escaped_label} is \([^)]+\)"
            left_replacement = f"The left direction of {label} is {format_coord_tuple(left_dir)}"
            transformed = re.sub(left_pattern, left_replacement, transformed)

            # Pattern: "The front direction of X is (x, y, z)"
            front_pattern = rf"The front direction of {escaped_label} is \([^)]+\)"
            front_replacement = f"The front direction of {label} is {format_coord_tuple(front_dir)}"
            transformed = re.sub(front_pattern, front_replacement, transformed)

            # Compute angles for orientation categories
            angle_left = compute_angle_degrees(camera_vec, left_dir)
            angle_right = 180.0 - angle_left
            angle_front = compute_angle_degrees(camera_vec, front_dir)
            angle_back = 180.0 - angle_front

            cos_left_val = cos_sim(camera_vec, left_dir)
            cos_front_val = cos_sim(camera_vec, front_dir)

            # Update cosine similarities and angles (for single-object orientation categories)
            # Pattern: "cosine similarity between the vector pointing to camera and the left direction is X"
            cos_left_pattern = r"(cosine similarity between the vector pointing to camera and the left direction is )[-\d.]+"
            cos_left_replacement = f"\\g<1>{cos_left_val:.2f}"
            transformed = re.sub(cos_left_pattern, cos_left_replacement, transformed)

            cos_front_pattern = r"(cosine similarity between the vector pointing to camera and the front direction is )[-\d.]+"
            cos_front_replacement = f"\\g<1>{cos_front_val:.2f}"
            transformed = re.sub(cos_front_pattern, cos_front_replacement, transformed)

            # Update angle values
            angle_left_pattern = r"(corresponding to an angle of )[\d.]+ degrees(\. Thus the angle between the vector pointing to camera and the right direction)"
            angle_left_replacement = f"\\g<1>{angle_left:.2f} degrees\\2"
            transformed = re.sub(angle_left_pattern, angle_left_replacement, transformed)

            angle_right_pattern = r"(the angle between the vector pointing to camera and the right direction is )[\d.]+ degrees"
            angle_right_replacement = f"\\g<1>{angle_right:.2f} degrees"
            transformed = re.sub(angle_right_pattern, angle_right_replacement, transformed)

            angle_front_pattern = r"(The cosine similarity between the vector pointing to camera and the front direction is [-\d.]+, corresponding to an angle of )[\d.]+ degrees"
            angle_front_replacement = f"\\g<1>{angle_front:.2f} degrees"
            transformed = re.sub(angle_front_pattern, angle_front_replacement, transformed)

            angle_back_pattern = r"(the angle between the vector pointing to camera and the back direction is )[\d.]+ degrees"
            angle_back_replacement = f"\\g<1>{angle_back:.2f} degrees"
            transformed = re.sub(angle_back_pattern, angle_back_replacement, transformed)

            # Update smallest angle
            angles = {'left': angle_left, 'right': angle_right, 'front': angle_front, 'back': angle_back}
            min_dir = min(angles, key=angles.get)
            min_angle = angles[min_dir]

            smallest_pattern = r"(the smallest angle is \w+ direction, with an angle of )[\d.]+ degrees"
            smallest_replacement = f"\\g<1>{min_angle:.2f} degrees"
            transformed = re.sub(smallest_pattern, smallest_replacement, transformed)

    return transformed


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    # Support loading multiple HuggingFace datasets separated by '+'
    dataset_names = script_args.dataset_name.split('+')
    all_datasets = []

    for ds_name in dataset_names:
        ds_name = ds_name.strip()
        logger.info(f"Loading dataset: {ds_name}")
        ds = load_dataset(ds_name, split=script_args.dataset_train_split)
        # Add source dataset identifier for image path resolution
        ds = ds.add_column("_source_dataset", [ds_name] * len(ds))
        all_datasets.append(ds)

    # Concatenate all datasets
    if len(all_datasets) > 1:
        dataset = concatenate_datasets(all_datasets)
        logger.info(f"Concatenated {len(all_datasets)} datasets, total samples: {len(dataset)}")
    else:
        dataset = all_datasets[0]

    # Create a DatasetDict for compatibility
    dataset_dict = datasets.DatasetDict({'train': dataset})

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    processor.image_processor.max_pixels = training_args.max_pixels
    processor.image_processor.min_pixels = training_args.min_pixels

    # Check if CoT transformation is enabled
    augmented_cot_transform = getattr(training_args, 'augmented_cot_transform', True)

    # HF Augmented image directory mapping
    # Maps dataset name patterns to image subdirectories
    hf_augmented_base_dir = getattr(training_args, 'hf_augmented_data_dir',
                                     '/data/SpatialReasoner/data/hf_augmented')
    hf_augmented_mapping = {
        'p10_ccw': 'image_p10_ccw',
        'm10_cw': 'image_m10_cw',
    }

    def resolve_image_path(example):
        """Resolve image path considering HF augmented datasets."""
        image_filename = example.get("image_filename", "")
        if not image_filename:
            return None

        source_dataset = example.get("_source_dataset", "")

        # Check if this is from HF augmented dataset
        for pattern, subdir in hf_augmented_mapping.items():
            if pattern in source_dataset:
                # HF augmented images are PNG, but dataset references JPG
                base_name = os.path.splitext(image_filename)[0]
                png_filename = f"{base_name}.png"
                hf_image_path = os.path.join(hf_augmented_base_dir, subdir, png_filename)
                if os.path.exists(hf_image_path):
                    return hf_image_path
                # Also try with original extension
                hf_image_path_orig = os.path.join(hf_augmented_base_dir, subdir, image_filename)
                if os.path.exists(hf_image_path_orig):
                    return hf_image_path_orig

        # Fallback to original data_dir (for non-HF augmented datasets)
        original_path = os.path.join(training_args.data_dir, image_filename)
        if os.path.exists(original_path):
            return original_path

        return None

    def collate_fn(examples):
        samples = []
        for example in examples:
            if 'question' in example and example['question']:
                # Handle standard/augmented format
                if example.get("A"):
                    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example.get(opt)]
                    question_text = example["question"]
                    options_text = "\n".join(options)
                    question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."
                else:
                    question = example["question"]

                # Load image using the resolver
                user_content = []
                image_path = resolve_image_path(example)
                if image_path:
                    image = Image.open(image_path).convert("RGB")
                    user_content.append({"type": "image", "image": image})
                elif example.get("image_filename"):
                    logger.warning(f"Image not found for: {example.get('image_filename')} (source: {example.get('_source_dataset', 'unknown')})")

                # Add question text
                user_content.append({"type": "text", "text": question})

                # Get answer_cot and optionally transform coordinates
                answer_cot = example.get('answer_cot', '')

                if augmented_cot_transform and answer_cot:
                    bounding_boxes = example.get('bounding_box', [])
                    directions = example.get('direction', [])

                    if bounding_boxes and directions:
                        answer_cot = transform_cot_coordinates(
                            answer_cot,
                            bounding_boxes,
                            directions
                        )

                # Get answer letter and wrap CoT in <think><answer> tags
                answer_letter = example.get('answer', '')
                formatted_response = f"<think>{answer_cot}</think><answer>{answer_letter}</answer>"

                converted_sample = [
                    {"role": "system", "content": [
                        {"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": formatted_response}]},
                ]
                samples.append(converted_sample)
            else:
                # Fallback for conversation format
                converted_sample = []
                for turn in example.get('conversations', []):
                    role = 'user' if turn['from'] == 'human' else 'assistant'
                    if '<image>' in turn['value']:
                        image_path = os.path.join(training_args.data_dir, example.get("image_path", ""))
                        if os.path.exists(image_path):
                            image = Image.open(image_path).convert("RGB")
                            converted_sample.append({"role": role, "content": [
                                {"type": "image", 'image': image},
                                {"type": "text", "text": turn['value'].lstrip('<image>\n').rstrip('\n<image>')}
                            ]})
                    else:
                        converted_sample.append({"role": role, "content": [
                            {"type": "text", "text": turn['value']}
                        ]})
                if converted_sample:
                    samples.append(converted_sample)

        if not samples:
            return None

        # Use max_length padding to ensure consistent tensor shapes across all GPUs
        batch = processor.apply_chat_template(
            samples,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=training_args.max_length,
            truncation=True
        )

        # CRITICAL: Force explicit padding to max_length
        max_len = training_args.max_length
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
                        device=batch[key].device
                    )
                    batch[key] = torch.cat([batch[key], padding], dim=1)
                elif current_len > max_len:
                    batch[key] = batch[key][:, :max_len]

        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100

        # Ignore image token IDs in labels
        if isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=None,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        processing_class=processor.tokenizer,
        callbacks=get_callbacks(training_args, model_args)+[EarlyStoppingCallback(stop_step=training_args.stop_steps)],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset_dict['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": script_args.dataset_name,
        "tags": ["SpatialReasoner", "Augmented"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if model_args.use_peft:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    main(script_args, training_args, model_args)
