"""MVGenMaster-based inference for 3DSRBench evaluation.

This module provides inference-time view synthesis using MVGenMaster,
enabling the model to reason from multiple viewpoints during evaluation.

Usage:
    python -m src.inference.mvgen_inference \
        --model_path ./checkpoints/mvgen_multi \
        --test_data ./data/3dsrbench_v1_vlmevalkit_circular.tsv \
        --output_path ./results/mvgen_inference.xlsx \
        --num_views 2 \
        --gpu_ids 0,1
"""

import os
import sys
import json
import base64
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# System prompt for CoT reasoning format
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

logger = logging.getLogger(__name__)


@dataclass
class MVGenConfig:
    """Configuration for MVGenMaster inference."""
    mvgenmaster_root: str = "/home/ubuntu/MVGenMaster"
    model_dir: str = "/home/ubuntu/MVGenMaster/check_points/pretrained_model"
    num_views: int = 2  # Number of views to generate (including original)
    azimuth_angles: List[float] = None  # Specific angles to generate
    guidance_scale: float = 2.0
    num_frames: int = 14  # Reduced for inference speed
    timeout: int = 300  # 5 minutes timeout

    def __post_init__(self):
        if self.azimuth_angles is None:
            # Default: original + 30 degree rotation
            self.azimuth_angles = [0.0, 30.0]


class MVGenMasterInference:
    """MVGenMaster wrapper for inference-time view synthesis."""

    def __init__(self, config: MVGenConfig):
        """Initialize MVGenMaster inference.

        Args:
            config: MVGen configuration
        """
        self.config = config
        self._root = Path(config.mvgenmaster_root)
        self._model_dir = Path(config.model_dir)
        self._validate_installation()

    def _validate_installation(self):
        """Verify MVGenMaster is properly installed."""
        if not self._root.exists():
            raise RuntimeError(f"MVGenMaster not found at {self._root}")
        if not (self._root / "run_mvgen.py").exists():
            raise RuntimeError(f"run_mvgen.py not found")
        if not self._model_dir.exists():
            raise RuntimeError(f"Model checkpoints not found at {self._model_dir}")

    def generate_views(
        self,
        image: Image.Image,
        angles: List[float] = None,
    ) -> Dict[float, Image.Image]:
        """Generate novel views from a single image.

        Args:
            image: Input PIL Image
            angles: List of azimuth angles to generate (default from config)

        Returns:
            Dictionary mapping angle -> generated image
        """
        angles = angles or self.config.azimuth_angles
        views = {0.0: image}  # Original image at 0 degrees

        if len(angles) <= 1:
            return views

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input image
            input_path = os.path.join(tmpdir, "input.jpg")
            image.convert("RGB").save(input_path, "JPEG", quality=95)

            # Generate views for non-zero angles
            for angle in angles:
                if abs(angle) < 0.5:
                    continue  # Skip original

                output_dir = os.path.join(tmpdir, f"output_{angle:.0f}")
                os.makedirs(output_dir, exist_ok=True)

                try:
                    generated = self._run_mvgen(input_path, output_dir, angle)
                    if generated:
                        views[angle] = generated
                except Exception as e:
                    logger.warning(f"Failed to generate view at {angle}°: {e}")

        return views

    def _run_mvgen(
        self,
        input_path: str,
        output_dir: str,
        d_phi: float,
    ) -> Optional[Image.Image]:
        """Run MVGenMaster to generate a single view.

        Args:
            input_path: Path to input image
            output_dir: Output directory
            d_phi: Azimuth rotation angle

        Returns:
            Generated PIL Image or None
        """
        cmd = [
            "python", str(self._root / "run_mvgen.py"),
            "--input_path", input_path,
            "--model_dir", str(self._model_dir),
            "--output_path", output_dir,
            "--nframe", str(self.config.num_frames),
            "--val_cfg", str(self.config.guidance_scale),
            "--d_phi", str(d_phi),
            "--d_theta", "0",
            "--cam_traj", "free",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(self._root),
            )

            if result.returncode != 0:
                logger.debug(f"MVGenMaster stderr: {result.stderr[:500]}")
                return None

            # Find the last generated frame
            images_dir = Path(output_dir) / "images"
            if not images_dir.exists():
                return None

            view_files = sorted(images_dir.glob("view*.png"))
            gen_files = [f for f in view_files if "_ref" not in f.name]

            if gen_files:
                # Return the last frame (most rotated)
                return Image.open(gen_files[-1]).convert("RGB")

        except subprocess.TimeoutExpired:
            logger.warning(f"MVGenMaster timed out for angle {d_phi}")
        except Exception as e:
            logger.warning(f"MVGenMaster error: {e}")

        return None


class MultiViewInferencer:
    """Main class for multi-view inference on 3DSRBench."""

    def __init__(
        self,
        model_path: str,
        mvgen_config: MVGenConfig = None,
        device: str = "cuda:0",
    ):
        """Initialize the multi-view inferencer.

        Args:
            model_path: Path to trained model
            mvgen_config: MVGenMaster configuration
            device: CUDA device
        """
        self.device = device
        self.mvgen_config = mvgen_config or MVGenConfig()
        self.mvgen = MVGenMasterInference(self.mvgen_config)

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=True,
        )
        logger.info("Model loaded")

    def infer_single(
        self,
        image: Image.Image,
        question: str,
        options: Dict[str, str],
        use_multiview: bool = True,
    ) -> str:
        """Run inference on a single sample.

        Args:
            image: Input image
            question: Question text
            options: Answer options (A, B, C, D)
            use_multiview: Whether to use multi-view synthesis

        Returns:
            Model prediction
        """
        # Generate views if multi-view enabled
        if use_multiview:
            views = self.mvgen.generate_views(image)
        else:
            views = {0.0: image}

        # Build conversation
        images = list(views.values())
        angles = list(views.keys())

        # Format question
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items() if v)
        question_text = (
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n"
            f"Please select the correct answer from the options above."
        )

        # Build message with multiple images
        if len(images) > 1:
            view_intro = f"You are shown {len(images)} views of the same scene "
            view_intro += f"at angles {angles}. "
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": view_intro + question_text})
        else:
            content = [
                {"type": "image", "image": images[0]},
                {"type": "text", "text": question_text},
            ]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}
        ]

        # Process and generate
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.01,
            )

        # Decode output
        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return output_text

    def infer_batch(
        self,
        test_data: pd.DataFrame,
        use_multiview: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run inference on entire test dataset.

        Args:
            test_data: DataFrame with test samples
            use_multiview: Whether to use multi-view synthesis
            show_progress: Show progress bar

        Returns:
            DataFrame with predictions added
        """
        predictions = []
        iterator = test_data.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(test_data), desc="Inference")

        for idx, row in iterator:
            try:
                # Decode image
                image_data = base64.b64decode(row["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")

                # Get options
                options = {
                    k: row[k] for k in ["A", "B", "C", "D"]
                    if k in row and pd.notna(row[k])
                }

                # Run inference
                prediction = self.infer_single(
                    image=image,
                    question=row["question"],
                    options=options,
                    use_multiview=use_multiview,
                )
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error on sample {idx}: {e}")
                predictions.append("")

        result = test_data.copy()
        result["prediction"] = predictions

        return result


def main():
    """Main entry point for MVGen inference."""
    parser = argparse.ArgumentParser(
        description="Multi-view inference on 3DSRBench"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/3dsrbench_v1_vlmevalkit_circular.tsv",
        help="Path to test data TSV",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save results",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=2,
        help="Number of views to use (1=single, 2+=multi)",
    )
    parser.add_argument(
        "--view_angles",
        type=str,
        default="0,30",
        help="Comma-separated view angles (e.g., '0,30,-30')",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse view angles
    angles = [float(a.strip()) for a in args.view_angles.split(",")]

    # Create config
    mvgen_config = MVGenConfig(
        num_views=args.num_views,
        azimuth_angles=angles,
    )

    # Initialize inferencer
    device = f"cuda:{args.gpu_id}"
    inferencer = MultiViewInferencer(
        model_path=args.model_path,
        mvgen_config=mvgen_config,
        device=device,
    )

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = pd.read_csv(args.test_data, sep="\t")
    if args.limit:
        test_data = test_data.head(args.limit)
    logger.info(f"Loaded {len(test_data)} samples")

    # Run inference
    use_multiview = args.num_views > 1
    results = inferencer.infer_batch(test_data, use_multiview=use_multiview)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".xlsx"):
        results.to_excel(output_path, index=False)
    else:
        results.to_csv(output_path, sep="\t", index=False)

    logger.info(f"Results saved to {output_path}")

    # Calculate accuracy
    def extract_answer(pred):
        if pd.isna(pred) or not pred:
            return ""
        for c in str(pred):
            if c.upper() in "ABCD":
                return c.upper()
        return ""

    results["pred_answer"] = results["prediction"].apply(extract_answer)
    results["correct"] = results["pred_answer"] == results["answer"].str.upper()
    accuracy = results["correct"].mean() * 100

    logger.info(f"Overall accuracy: {accuracy:.2f}%")

    # Per-category accuracy
    if "category" in results.columns:
        logger.info("Per-category accuracy:")
        for cat in results["category"].unique():
            cat_acc = results[results["category"] == cat]["correct"].mean() * 100
            logger.info(f"  {cat}: {cat_acc:.2f}%")


if __name__ == "__main__":
    main()
