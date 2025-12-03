"""Full data generation pipeline."""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import logging

from .config import DataGenerationConfig
from .depth import DepthEstimator
from .segment import Segmenter
from .view_synth import ViewSynthesizer
from .pose import PoseEstimator
from .qa_gen import QAGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerationPipeline:
    """Full pipeline for multi-view spatial reasoning data."""

    def __init__(self, config: DataGenerationConfig):
        self.config = config
        logger.info("Initializing pipeline components...")
        self.depth_estimator = DepthEstimator(config.depth)
        self.segmenter = Segmenter(config.segmentation)
        self.view_synthesizer = ViewSynthesizer(config.view_synthesis)
        self.pose_estimator = PoseEstimator(config.pose_estimation)
        self.qa_generator = QAGenerator(config.qa_generation)

    def process_single_image(self, image_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
        image_id = Path(image_path).stem

        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            W, H = image.size

            # Size check
            if W < self.config.min_image_size[0] or H < self.config.min_image_size[1]:
                logger.debug(f"Image {image_id} too small")
                return None

            # 1. Depth
            depth_map = self.depth_estimator.estimate_metric(image_np)

            # 2. Segmentation
            seg = self.segmenter.segment(image_np)
            seg = seg.filter_by_area(self.config.segmentation.min_mask_region_area)

            if len(seg) < self.config.qa_generation.min_objects_per_image:
                logger.debug(f"Not enough objects in {image_id}")
                return None

            # 3. Camera intrinsics
            K = self.depth_estimator.estimate_intrinsics(W, H)

            # 4. View synthesis
            views = self.view_synthesizer.synthesize(image_np, depth_map, K)
            valid_views = {k: v for k, v in views.items() if v.is_valid}

            if len(valid_views) < 2:
                logger.debug(f"Not enough valid views for {image_id}")
                return None

            # 5. Pose estimation
            poses = self.pose_estimator.estimate_poses(
                depth_map, seg.masks, seg.boxes, K, seg.labels
            )

            if len(poses) < self.config.qa_generation.min_objects_per_image:
                logger.debug(f"Not enough valid poses in {image_id}")
                return None

            # 6. QA generation
            qa_pairs = self.qa_generator.generate(image_id, poses)

            if not qa_pairs:
                logger.debug(f"No QA pairs for {image_id}")
                return None

            # 7. Save
            return self._save_outputs(
                image_id, output_dir, image_np, depth_map,
                valid_views, poses, qa_pairs
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def _save_outputs(
        self, image_id: str, output_dir: str, image: np.ndarray,
        depth_map: np.ndarray, views: Dict, poses: List, qa_pairs: List
    ) -> Dict[str, Any]:
        images_dir = os.path.join(output_dir, "images")
        depth_dir = os.path.join(output_dir, "depth")
        poses_dir = os.path.join(output_dir, "poses")

        for d in [images_dir, depth_dir, poses_dir]:
            os.makedirs(d, exist_ok=True)

        # Save views
        view_paths = {}
        for angle, view in views.items():
            fname = f"{image_id}_view_{angle:+.0f}.jpg"
            Image.fromarray(view.image).save(os.path.join(images_dir, fname), quality=95)
            view_paths[angle] = fname

        # Save depth
        depth_fname = f"{image_id}_depth.npy"
        np.save(os.path.join(depth_dir, depth_fname), depth_map)

        # Save poses
        poses_data = [{
            "object_id": p.object_id,
            "label": p.label,
            "position": p.position.tolist(),
            "scale": p.scale.tolist(),
            "quaternion": p.quaternion.tolist(),
            "bbox_2d": p.bbox_2d.tolist(),
        } for p in poses]

        poses_fname = f"{image_id}_poses.json"
        with open(os.path.join(poses_dir, poses_fname), "w") as f:
            json.dump(poses_data, f, indent=2)

        # Prepare QA data
        qa_data = [{
            "image_id": image_id,
            "question": qa.question,
            "answer": qa.answer,
            "answer_name": qa.answer_name,
            "query_type": qa.query_type,
            "query_subtype": qa.query_subtype,
            "objects_involved": qa.objects_involved,
            "options": qa.options,
            "view_images": {str(k): v for k, v in view_paths.items()},
        } for qa in qa_pairs]

        return {
            "image_id": image_id,
            "view_paths": view_paths,
            "depth_path": depth_fname,
            "poses_path": poses_fname,
            "qa_pairs": qa_data,
            "num_objects": len(poses),
            "num_qa": len(qa_pairs),
        }

    def process_dataset(
        self, input_dir: str, output_dir: str, image_list: Optional[List[str]] = None
    ):
        os.makedirs(output_dir, exist_ok=True)

        if image_list is None:
            exts = [".jpg", ".jpeg", ".png"]
            image_list = [f for f in os.listdir(input_dir)
                          if any(f.lower().endswith(e) for e in exts)]

        # Limit images
        if len(image_list) > self.config.num_images:
            import random
            random.seed(self.config.seed)
            image_list = random.sample(image_list, self.config.num_images)

        logger.info(f"Processing {len(image_list)} images...")

        all_results = []
        all_qa = []

        for image_file in tqdm(image_list, desc="Processing"):
            result = self.process_single_image(
                os.path.join(input_dir, image_file), output_dir
            )
            if result:
                all_results.append(result)
                all_qa.extend(result["qa_pairs"])

        # Save combined QA
        with open(os.path.join(output_dir, "qa_pairs.json"), "w") as f:
            json.dump(all_qa, f, indent=2)

        # Save metadata
        metadata = {
            "num_images": len(all_results),
            "num_qa_pairs": len(all_qa),
            "query_distribution": self._query_dist(all_qa),
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Done! {len(all_qa)} QA pairs from {len(all_results)} images")
        return all_results

    def _query_dist(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        dist = {}
        for qa in qa_pairs:
            st = qa["query_subtype"]
            dist[st] = dist.get(st, 0) + 1
        return dist
