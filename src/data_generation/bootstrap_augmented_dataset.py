"""Bootstrap augmented dataset from HuggingFace SpatialReasonerTrain-SFT.

Creates an initial augmented_dataset.json with original images as 0-degree views.
This allows training to start immediately while more views are generated.

Usage:
    python -m src.data_generation.bootstrap_augmented_dataset \
        --output_dir /data/SpatialReasoner/data/mvgen_augmented \
        --data_dir /data/SpatialReasoner/data/openimages
"""

import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)


def bootstrap_from_hf(
    output_dir: str,
    data_dir: str,
    split: str = "train",
) -> int:
    """Bootstrap augmented dataset from HuggingFace dataset.

    Args:
        output_dir: Output directory for augmented_dataset.json
        data_dir: Directory containing original images
        split: Dataset split to use

    Returns:
        Number of samples processed
    """
    logger.info("Loading HuggingFace dataset ccvl/SpatialReasonerTrain-SFT...")
    dataset = load_dataset("ccvl/SpatialReasonerTrain-SFT", split=split)
    logger.info(f"Loaded {len(dataset)} samples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    samples = []
    skipped = 0

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Processing"):
        # Skip LLaVA samples (no spatial reasoning question)
        if not sample.get("question"):
            skipped += 1
            continue

        image_filename = sample.get("image_filename", "")
        if not image_filename:
            skipped += 1
            continue

        # Build full path to original image
        original_path = os.path.join(data_dir, image_filename)

        # Check if image exists
        if not os.path.exists(original_path):
            skipped += 1
            continue

        # Create augmented sample with just original view
        augmented_sample = {
            "sample_id": f"sample_{idx:06d}",
            "original_image": image_filename,
            "question": sample["question"],
            "answer": sample.get("answer", ""),
            "answer_cot": sample.get("answer_cot", sample.get("answer", "")),
            "A": sample.get("A", ""),
            "B": sample.get("B", ""),
            "C": sample.get("C", ""),
            "D": sample.get("D", ""),
            # View images: just the original (0-degree) view
            "view_images": {
                "0.0": original_path,
            },
            # Augmented CoTs: just the original CoT for 0 degrees
            "augmented_cots": {
                "0.0": sample.get("answer_cot", sample.get("answer", "")),
            },
        }

        samples.append(augmented_sample)

    # Save augmented dataset
    output_path = os.path.join(output_dir, "augmented_dataset.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Saved {len(samples)} samples to {output_path}")
    logger.info(f"Skipped {skipped} samples (no question or missing image)")

    # Save metadata
    metadata = {
        "total_samples": len(samples),
        "skipped_samples": skipped,
        "source": "ccvl/SpatialReasonerTrain-SFT",
        "data_dir": data_dir,
        "views_generated": ["0.0"],
        "note": "Bootstrap dataset with original images only. Run mvgen_bulk_augmentation.py to add more views.",
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap augmented dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/SpatialReasoner/data/mvgen_augmented",
        help="Output directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/SpatialReasoner/data/openimages",
        help="Directory containing original images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    count = bootstrap_from_hf(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        split=args.split,
    )

    logger.info(f"Bootstrap complete: {count} samples ready for training")


if __name__ == "__main__":
    main()
