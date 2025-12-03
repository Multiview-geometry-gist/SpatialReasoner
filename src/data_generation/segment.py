"""Instance segmentation using SAM2 or transformers SAM."""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from dataclasses import dataclass

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    masks: np.ndarray  # (N, H, W) boolean
    boxes: np.ndarray  # (N, 4) [x1, y1, x2, y2]
    scores: np.ndarray  # (N,)
    labels: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.masks)

    def filter_by_area(self, min_area: int, max_area: Optional[int] = None) -> "SegmentationResult":
        areas = self.masks.sum(axis=(1, 2))
        valid = areas >= min_area
        if max_area:
            valid &= areas <= max_area
        return SegmentationResult(
            masks=self.masks[valid],
            boxes=self.boxes[valid],
            scores=self.scores[valid],
            labels=[self.labels[i] for i, v in enumerate(valid) if v] if self.labels else None
        )


class Segmenter:
    """SAM-based segmentation."""

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._init_model()

    def _init_model(self):
        try:
            # Try SAM2 first
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self._use_sam2 = True
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
        except ImportError:
            # Fallback to transformers SAM
            from transformers import SamModel, SamProcessor
            self._use_sam2 = False
            self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()
    def segment(self, image: Union[Image.Image, np.ndarray, str]) -> SegmentationResult:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        if self._use_sam2:
            return self._segment_sam2(image_np)
        return self._segment_transformers(image_np)

    def _segment_sam2(self, image_np: np.ndarray) -> SegmentationResult:
        masks_data = self.mask_generator.generate(image_np)
        if not masks_data:
            H, W = image_np.shape[:2]
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                boxes=np.zeros((0, 4)),
                scores=np.zeros((0,)),
            )

        masks = np.stack([m["segmentation"] for m in masks_data])
        boxes = np.stack([m["bbox"] for m in masks_data])  # [x, y, w, h]
        # Convert to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0]
        boxes_xyxy[:, 1] = boxes[:, 1]
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]

        scores = np.array([m["predicted_iou"] for m in masks_data])
        return SegmentationResult(masks=masks, boxes=boxes_xyxy, scores=scores)

    def _segment_transformers(self, image_np: np.ndarray) -> SegmentationResult:
        H, W = image_np.shape[:2]
        # Grid points
        n = self.config.points_per_side
        x = np.linspace(0, W, n + 2)[1:-1]
        y = np.linspace(0, H, n + 2)[1:-1]
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)

        inputs = self.processor(
            Image.fromarray(image_np),
            input_points=[points.tolist()],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0].cpu().numpy().squeeze()

        scores = outputs.iou_scores.cpu().numpy().squeeze()
        boxes = self._masks_to_boxes(masks)

        return SegmentationResult(masks=masks, boxes=boxes, scores=scores)

    @staticmethod
    def _masks_to_boxes(masks: np.ndarray) -> np.ndarray:
        boxes = []
        for mask in masks:
            if mask.sum() == 0:
                boxes.append([0, 0, 0, 0])
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes)
