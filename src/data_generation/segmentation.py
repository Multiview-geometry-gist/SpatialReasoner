"""Instance segmentation using SAM2.

Provides automatic mask generation and object detection.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    masks: np.ndarray  # (N, H, W) boolean masks
    boxes: np.ndarray  # (N, 4) bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # (N,) confidence scores
    labels: Optional[List[str]] = None  # Object labels if available

    def __len__(self) -> int:
        return len(self.masks)

    def filter_by_area(self, min_area: int, max_area: Optional[int] = None) -> "SegmentationResult":
        """Filter segments by area."""
        areas = self.masks.sum(axis=(1, 2))
        valid = areas >= min_area
        if max_area is not None:
            valid &= areas <= max_area

        return SegmentationResult(
            masks=self.masks[valid],
            boxes=self.boxes[valid],
            scores=self.scores[valid],
            labels=[self.labels[i] for i, v in enumerate(valid) if v] if self.labels else None
        )

    def filter_by_score(self, min_score: float) -> "SegmentationResult":
        """Filter segments by confidence score."""
        valid = self.scores >= min_score
        return SegmentationResult(
            masks=self.masks[valid],
            boxes=self.boxes[valid],
            scores=self.scores[valid],
            labels=[self.labels[i] for i, v in enumerate(valid) if v] if self.labels else None
        )


class ObjectSegmenter:
    """Wrapper for SAM2 automatic mask generation."""

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.processor = None
        self._initialized = False
        self._use_transformers = False

    def _lazy_init(self):
        """Lazy initialization."""
        if self._initialized:
            return

        try:
            # Try SAM2 first
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            self.model = build_sam2(
                config_file="sam2_hiera_l.yaml",
                ckpt_path=self.config.model_name,
                device=self.device,
            )
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                min_mask_region_area=self.config.min_mask_region_area,
            )
            self._use_transformers = False
        except ImportError:
            # Fallback to transformers SAM
            try:
                from transformers import SamModel, SamProcessor
                import torch

                self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
                self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
                self.model.to(self.device)
                self.model.eval()
                self._use_transformers = True
            except ImportError:
                raise ImportError(
                    "Either sam2 or transformers with SAM is required. "
                    "Install sam2: pip install sam2 or use transformers: pip install transformers"
                )

        self._initialized = True

    def segment(
        self,
        image: Union[Image.Image, np.ndarray, str],
    ) -> SegmentationResult:
        """
        Perform automatic instance segmentation.

        Args:
            image: Input image

        Returns:
            SegmentationResult containing masks, boxes, and scores
        """
        self._lazy_init()

        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        H, W = image_np.shape[:2]

        if self._use_transformers:
            return self._segment_transformers(image_np)

        # Use SAM2
        masks_data = self.mask_generator.generate(image_np)

        if len(masks_data) == 0:
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                boxes=np.zeros((0, 4)),
                scores=np.zeros((0,)),
            )

        # Extract results
        masks = np.stack([m["segmentation"] for m in masks_data])
        boxes = np.stack([m["bbox"] for m in masks_data])  # [x, y, w, h] format

        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0]
        boxes_xyxy[:, 1] = boxes[:, 1]
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]

        scores = np.array([m["predicted_iou"] for m in masks_data])

        return SegmentationResult(
            masks=masks,
            boxes=boxes_xyxy,
            scores=scores,
        )

    def _segment_transformers(self, image_np: np.ndarray) -> SegmentationResult:
        """Fallback segmentation using transformers SAM."""
        import torch

        H, W = image_np.shape[:2]

        # Generate grid of points
        points_per_side = min(self.config.points_per_side, 16)  # Limit for efficiency
        x = np.linspace(0, W, points_per_side + 2)[1:-1]
        y = np.linspace(0, H, points_per_side + 2)[1:-1]
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)

        # Process in batches
        all_masks = []
        all_scores = []
        batch_size = 16

        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                inputs = self.processor(
                    Image.fromarray(image_np),
                    input_points=[[p.tolist() for p in batch_points]],
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # Get masks
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )[0]

                masks = masks.cpu().numpy().squeeze()
                scores = outputs.iou_scores.cpu().numpy().squeeze()

                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]
                    scores = np.array([scores])

                all_masks.append(masks)
                all_scores.append(scores)

        if not all_masks:
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                boxes=np.zeros((0, 4)),
                scores=np.zeros((0,)),
            )

        masks = np.concatenate(all_masks, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # Filter by score
        valid = scores > self.config.pred_iou_thresh
        masks = masks[valid]
        scores = scores[valid]

        # Compute boxes from masks
        boxes = self._masks_to_boxes(masks)

        return SegmentationResult(
            masks=masks.astype(bool),
            boxes=boxes,
            scores=scores,
        )

    @staticmethod
    def _masks_to_boxes(masks: np.ndarray) -> np.ndarray:
        """Convert binary masks to bounding boxes."""
        boxes = []
        for mask in masks:
            if mask.sum() == 0:
                boxes.append([0, 0, 0, 0])
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            if len(y_indices) == 0 or len(x_indices) == 0:
                boxes.append([0, 0, 0, 0])
                continue
            y1, y2 = y_indices[[0, -1]]
            x1, x2 = x_indices[[0, -1]]
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32)

    def get_object_depth_stats(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray
    ) -> Dict[str, float]:
        """
        Get depth statistics for a masked object.

        Returns:
            Dict with 'mean', 'median', 'min', 'max', 'std'
        """
        object_depths = depth_map[mask]

        if len(object_depths) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        return {
            "mean": float(np.mean(object_depths)),
            "median": float(np.median(object_depths)),
            "min": float(np.min(object_depths)),
            "max": float(np.max(object_depths)),
            "std": float(np.std(object_depths)),
        }
