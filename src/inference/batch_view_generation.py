"""Batch view generation for 3DSRBench test images using MVGenMaster.

This script pre-generates all views for test images, enabling fast multiview inference.

Usage:
    # Generate views for all test images (parallelized across GPUs)
    python -m src.inference.batch_view_generation \
        --test_data ./data/benchmark/3dsrbench_v1_vlmevalkit_circular.tsv \
        --output_dir ./data/benchmark/generated_views \
        --angles "10,-10" \
        --num_gpus 4

    # Then run inference with pre-generated views
    python -m src.inference.mvgen_inference_precomputed \
        --model_path ./checkpoints/multiview \
        --views_dir ./data/benchmark/generated_views \
        --output_path ./results/multiview_eval.xlsx
"""

import os
import sys
import json
import base64
import hashlib
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MVGenConfig:
    """Configuration for MVGenMaster."""
    mvgenmaster_root: str = "/home/ubuntu/MVGenMaster"
    model_dir: str = "/home/ubuntu/MVGenMaster/check_points/pretrained_model"
    guidance_scale: float = 2.0
    num_frames: int = 14
    timeout: int = 300  # 5 minutes


def get_image_hash(image_data: str) -> str:
    """Generate hash for image data to use as unique identifier."""
    return hashlib.md5(image_data.encode()[:1000]).hexdigest()[:12]


def decode_image(image_string: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    image_data = base64.b64decode(image_string)
    return Image.open(BytesIO(image_data)).convert('RGB')


def generate_view_for_image(
    args: Tuple[str, str, float, str, MVGenConfig]
) -> Tuple[str, float, Optional[str]]:
    """Generate a single view for an image.

    Args:
        args: (image_hash, image_b64, angle, output_dir, config)

    Returns:
        (image_hash, angle, output_path or None if failed)
    """
    image_hash, image_b64, angle, output_dir, config = args

    # Output path for this view
    output_path = os.path.join(output_dir, f"{image_hash}_angle{angle:.0f}.jpg")

    # Skip if already exists
    if os.path.exists(output_path):
        return (image_hash, angle, output_path)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input image
            image = decode_image(image_b64)
            input_path = os.path.join(tmpdir, "input.jpg")
            image.save(input_path, "JPEG", quality=95)

            # Run MVGenMaster
            mvgen_output = os.path.join(tmpdir, "output")
            os.makedirs(mvgen_output, exist_ok=True)

            cmd = [
                "python", os.path.join(config.mvgenmaster_root, "run_mvgen.py"),
                "--input_path", input_path,
                "--model_dir", config.model_dir,
                "--output_path", mvgen_output,
                "--nframe", str(config.num_frames),
                "--val_cfg", str(config.guidance_scale),
                "--d_phi", str(angle),
                "--d_theta", "0",
                "--cam_traj", "free",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                cwd=config.mvgenmaster_root,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
            )

            if result.returncode != 0:
                logger.debug(f"MVGen failed for {image_hash} at {angle}°")
                return (image_hash, angle, None)

            # Find generated image
            images_dir = Path(mvgen_output) / "images"
            if not images_dir.exists():
                return (image_hash, angle, None)

            view_files = sorted(images_dir.glob("view*.png"))
            gen_files = [f for f in view_files if "_ref" not in f.name]

            if gen_files:
                # Save the last frame (most rotated)
                gen_image = Image.open(gen_files[-1]).convert("RGB")
                gen_image.save(output_path, "JPEG", quality=95)
                return (image_hash, angle, output_path)

    except Exception as e:
        logger.warning(f"Error generating view for {image_hash} at {angle}°: {e}")

    return (image_hash, angle, None)


def generate_views_on_gpu(
    tasks: List[Tuple],
    gpu_id: int,
    config: MVGenConfig,
    progress_dict: dict,
) -> List[Tuple[str, float, Optional[str]]]:
    """Generate views on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    results = []
    for i, task in enumerate(tasks):
        result = generate_view_for_image(task)
        results.append(result)
        progress_dict[gpu_id] = i + 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch generate views for test images")
    parser.add_argument(
        "--test_data",
        type=str,
        default="/data/SpatialReasoner/data/benchmark/3dsrbench_v1_vlmevalkit_circular.tsv",
        help="Path to test data TSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/SpatialReasoner/data/benchmark/generated_views",
        help="Directory to save generated views",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="10,-10",
        help="Comma-separated angles to generate (e.g., '10,-10')",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Specific GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for resuming",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (exclusive)",
    )

    args = parser.parse_args()

    # Parse angles
    angles = [float(a.strip()) for a in args.angles.split(",")]
    logger.info(f"Will generate views at angles: {angles}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = pd.read_csv(args.test_data, sep="\t")
    logger.info(f"Total samples: {len(test_data)}")

    # Extract unique images
    unique_images = {}
    for _, row in test_data.iterrows():
        img_hash = get_image_hash(row["image"])
        if img_hash not in unique_images:
            unique_images[img_hash] = row["image"]

    logger.info(f"Unique images: {len(unique_images)}")

    # Apply start/end indices
    image_items = list(unique_images.items())
    if args.end_idx:
        image_items = image_items[args.start_idx:args.end_idx]
    else:
        image_items = image_items[args.start_idx:]

    logger.info(f"Processing images {args.start_idx} to {args.start_idx + len(image_items)}")

    # Create tasks for all images and angles
    config = MVGenConfig()
    tasks = []
    for img_hash, img_b64 in image_items:
        for angle in angles:
            tasks.append((img_hash, img_b64, angle, args.output_dir, config))

    logger.info(f"Total view generation tasks: {len(tasks)}")

    # Determine GPUs to use
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))

    logger.info(f"Using GPUs: {gpu_ids}")

    # Split tasks across GPUs
    if len(gpu_ids) > 1:
        tasks_per_gpu = len(tasks) // len(gpu_ids)
        gpu_tasks = []
        for i, gpu_id in enumerate(gpu_ids):
            start = i * tasks_per_gpu
            end = start + tasks_per_gpu if i < len(gpu_ids) - 1 else len(tasks)
            gpu_tasks.append((tasks[start:end], gpu_id))

        # Use multiprocessing for parallel GPU execution
        manager = Manager()
        progress_dict = manager.dict()

        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for gpu_task, gpu_id in gpu_tasks:
                future = executor.submit(
                    generate_views_on_gpu,
                    gpu_task,
                    gpu_id,
                    config,
                    progress_dict,
                )
                futures.append(future)

            # Wait for all to complete with progress
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        # Single GPU execution with progress bar
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        results = []
        for task in tqdm(tasks, desc="Generating views"):
            result = generate_view_for_image(task)
            results.append(result)

    # Save mapping file
    mapping = {}
    success_count = 0
    for img_hash, angle, path in results:
        if path:
            if img_hash not in mapping:
                mapping[img_hash] = {}
            mapping[img_hash][angle] = path
            success_count += 1

    mapping_path = os.path.join(args.output_dir, "view_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Generated {success_count}/{len(tasks)} views successfully")
    logger.info(f"Mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
