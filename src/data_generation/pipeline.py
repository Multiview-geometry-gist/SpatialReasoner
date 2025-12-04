"""Full data generation pipeline orchestration.

Combines all components:
1. Depth estimation
2. Segmentation
3. View synthesis
4. Pose estimation
5. QA generation
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import logging
import random

from .config import DataGenerationConfig
from .depth_estimation import DepthEstimator
from .segmentation import ObjectSegmenter, SegmentationResult
from .view_synthesis import ViewSynthesizer, SynthesizedView
from .pose_estimation import PoseEstimator, ObjectPose
from .qa_generation import QAGenerator, QAPair


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerationPipeline:
    """Full pipeline for generating multi-view spatial reasoning data."""

    def __init__(self, config: DataGenerationConfig):
        self.config = config

        logger.info("Initializing pipeline components...")
        self.depth_estimator = DepthEstimator(config.depth)
        self.segmenter = ObjectSegmenter(config.segmentation)

        if config.view_synthesis.enabled:
            self.view_synthesizer = ViewSynthesizer(config.view_synthesis)
        else:
            self.view_synthesizer = None

        self.pose_estimator = PoseEstimator(config.pose_estimation)
        self.qa_generator = QAGenerator(config.qa_generation)

    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image through the full pipeline.

        Returns:
            Dictionary containing all generated data, or None if failed
        """
        image_id = Path(image_path).stem

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            W, H = image.size
            if W < self.config.min_image_size[0] or H < self.config.min_image_size[1]:
                logger.warning(f"Image {image_id} too small, skipping")
                return None

            # 1. Depth estimation
            logger.debug(f"Estimating depth for {image_id}")
            depth_map = self.depth_estimator.estimate_metric(image_np)

            # 2. Segmentation
            logger.debug(f"Segmenting {image_id}")
            seg_result = self.segmenter.segment(image_np)
            seg_result = seg_result.filter_by_area(
                min_area=self.config.segmentation.min_mask_region_area
            )

            if len(seg_result) < self.config.qa_generation.min_objects_per_image:
                logger.warning(f"Not enough objects in {image_id}, skipping")
                return None

            # 3. Estimate camera intrinsics
            K = self.depth_estimator.estimate_intrinsics(W, H)

            # 4. View synthesis (optional)
            views = {}
            if self.view_synthesizer:
                logger.debug(f"Synthesizing views for {image_id}")
                views = self.view_synthesizer.synthesize(image_np, depth_map, K)

                valid_views = {k: v for k, v in views.items() if v.is_valid}
                if len(valid_views) < 2:
                    logger.warning(f"Not enough valid views for {image_id}, using original only")
                    views = {0.0: SynthesizedView(
                        image=image_np,
                        angle=0.0,
                        mask=np.ones((H, W), dtype=bool),
                        hole_ratio=0.0
                    )}
            else:
                views = {0.0: SynthesizedView(
                    image=image_np,
                    angle=0.0,
                    mask=np.ones((H, W), dtype=bool),
                    hole_ratio=0.0
                )}

            # 5. Pose estimation
            logger.debug(f"Estimating poses for {image_id}")
            poses = self.pose_estimator.estimate_poses(
                image_np,
                depth_map,
                seg_result.masks,
                seg_result.boxes,
                K,
                seg_result.labels,
            )

            if len(poses) < self.config.qa_generation.min_objects_per_image:
                logger.warning(f"Not enough valid poses in {image_id}, skipping")
                return None

            # 6. QA generation
            logger.debug(f"Generating QA pairs for {image_id}")
            qa_pairs = self.qa_generator.generate(image_id, poses)

            if len(qa_pairs) == 0:
                logger.warning(f"No QA pairs generated for {image_id}, skipping")
                return None

            # 7. Save outputs
            result = self._save_outputs(
                image_id=image_id,
                output_dir=output_dir,
                original_image=image_np,
                depth_map=depth_map,
                views=views,
                poses=poses,
                qa_pairs=qa_pairs,
                seg_result=seg_result,
            )

            return result

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_outputs(
        self,
        image_id: str,
        output_dir: str,
        original_image: np.ndarray,
        depth_map: np.ndarray,
        views: Dict[float, SynthesizedView],
        poses: List[ObjectPose],
        qa_pairs: List[QAPair],
        seg_result: SegmentationResult,
    ) -> Dict[str, Any]:
        """Save all generated data to disk."""

        # Create directories
        images_dir = os.path.join(output_dir, "images")
        depth_dir = os.path.join(output_dir, "depth")
        poses_dir = os.path.join(output_dir, "poses")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)

        # Save views
        view_paths = {}
        for angle, view in views.items():
            view_filename = f"{image_id}_view_{angle:+.0f}.jpg"
            view_path = os.path.join(images_dir, view_filename)
            Image.fromarray(view.image).save(view_path, quality=95)
            view_paths[angle] = view_filename

        # Save depth
        depth_filename = f"{image_id}_depth.npy"
        depth_path = os.path.join(depth_dir, depth_filename)
        np.save(depth_path, depth_map)

        # Save poses
        poses_data = [pose.to_dict() for pose in poses]
        poses_filename = f"{image_id}_poses.json"
        poses_path = os.path.join(poses_dir, poses_filename)
        with open(poses_path, "w") as f:
            json.dump(poses_data, f, indent=2)

        # Prepare QA data
        qa_data = []
        for i, qa in enumerate(qa_pairs):
            qa_dict = self.qa_generator.format_for_training(
                qa,
                image_filename=f"{image_id}_view_+0.jpg",  # Use original view
                question_index=f"{image_id}_{i}",
            )
            qa_dict["view_images"] = {str(k): v for k, v in view_paths.items()}
            qa_data.append(qa_dict)

        return {
            "image_id": image_id,
            "view_paths": view_paths,
            "depth_path": depth_filename,
            "poses_path": poses_filename,
            "qa_pairs": qa_data,
            "num_objects": len(poses),
            "num_qa": len(qa_pairs),
        }

    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        image_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process entire dataset.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output data
            image_list: Optional list of image filenames to process

        Returns:
            List of result dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get image list
        if image_list is None:
            image_extensions = [".jpg", ".jpeg", ".png"]
            image_list = [
                f for f in os.listdir(input_dir)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]

        # Limit to configured number
        if len(image_list) > self.config.num_images:
            random.seed(self.config.seed)
            image_list = random.sample(image_list, self.config.num_images)

        logger.info(f"Processing {len(image_list)} images...")

        # Process images
        all_results = []
        all_qa_pairs = []

        for image_file in tqdm(image_list, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            result = self.process_single_image(image_path, output_dir)

            if result is not None:
                all_results.append(result)
                all_qa_pairs.extend(result["qa_pairs"])

        # Save combined QA file
        qa_file = os.path.join(output_dir, "qa_pairs.json")
        with open(qa_file, "w") as f:
            json.dump(all_qa_pairs, f, indent=2)

        # Save metadata
        metadata = {
            "num_images": len(all_results),
            "num_qa_pairs": len(all_qa_pairs),
            "config": self.config.to_dict(),
            "query_type_distribution": self._compute_query_distribution(all_qa_pairs),
        }

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Done! Generated {len(all_qa_pairs)} QA pairs from {len(all_results)} images")
        logger.info(f"Output saved to {output_dir}")

        return all_results

    def _compute_query_distribution(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """Compute distribution of query types."""
        distribution = {}
        for qa in qa_pairs:
            subtype = qa.get("query_subtype", qa.get("category", "unknown"))
            distribution[subtype] = distribution.get(subtype, 0) + 1
        return distribution


def main():
    """CLI entry point for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-view spatial reasoning data")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        config = DataGenerationConfig.from_yaml(args.config)
    else:
        config = DataGenerationConfig()

    # Override from command line
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.seed = args.seed
    if args.num_images:
        config.num_images = args.num_images

    # Run pipeline
    pipeline = DataGenerationPipeline(config)
    pipeline.process_dataset(config.input_dir, config.output_dir)


if __name__ == "__main__":
    main()
