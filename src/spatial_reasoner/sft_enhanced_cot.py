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
Enhanced CoT SFT Training with Category-Specific Reasoning Templates.

This module extends HF-Augmented training with:
1. Category-specific CoT generation (not just coordinate replacement)
2. Explicit geometric reasoning explanations
3. View-aware reasoning that explains WHY coordinates change
4. Multi-angle support with progressive training

Key Innovation:
    Instead of just replacing coordinates in existing CoT, we generate
    enhanced CoT that explicitly explains the spatial reasoning process
    for each question category.

Usage:
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
        src/spatial_reasoner/sft_enhanced_cot.py \
        --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_enhanced_cot.yaml
"""

import logging
import os
import sys
import re
import math
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
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
# Category-Specific CoT Templates
# ============================================================================

class EnhancedCoTGenerator:
    """Generates category-specific Chain-of-Thought reasoning.

    This class produces detailed, mathematically grounded CoT explanations
    tailored to each spatial reasoning category.
    """

    # Category templates with explicit reasoning steps
    CATEGORY_TEMPLATES = {
        "orientation_viewpoint": """To determine which side of {object} faces the camera, I'll analyze the 3D geometry:

Step 1: Identify positions
- Camera position: (0, 0, 0) - at the origin
- {object} position: {position}

Step 2: Calculate the camera-to-object vector
- Vector from {object} to camera: {to_camera_vec}

Step 3: Get object orientation vectors
- {object}'s front direction: {front_dir}
- {object}'s left direction: {left_dir}
- Right direction (opposite of left): {right_dir}
- Back direction (opposite of front): {back_dir}

Step 4: Calculate angles to each side
- Angle between camera vector and front: {angle_front:.1f}°
- Angle between camera vector and back: {angle_back:.1f}°
- Angle between camera vector and left: {angle_left:.1f}°
- Angle between camera vector and right: {angle_right:.1f}°

Step 5: Determine facing direction
The side with the smallest angle to the camera vector is facing the camera.
Smallest angle: {min_angle:.1f}° (to the {min_direction})

Therefore, the {min_direction} side of {object} is facing the camera.
The answer is {answer}.""",

        "orientation_on_the_left": """To determine if {object} is on the left side of the image:

Step 1: Get {object}'s 3D position
- Position: {position}

Step 2: Analyze the X-coordinate
- X-coordinate: {x_coord:.2f}
- In this coordinate system, negative X = left, positive X = right

Step 3: Determine side
- Since X = {x_coord:.2f}, which is {x_analysis}

Therefore, {object} is {conclusion}.
The answer is {answer}.""",

        "orientation_in_front_of": """To determine if {object1} is in front of {object2}:

Step 1: Get 3D positions
- {object1} position: {pos1}
- {object2} position: {pos2}

Step 2: Compare Z-coordinates (depth)
- {object1} Z-coordinate: {z1:.2f}
- {object2} Z-coordinate: {z2:.2f}
- In this system, smaller Z = closer to camera = more "in front"

Step 3: Calculate depth difference
- Difference: {z1:.2f} - {z2:.2f} = {z_diff:.2f}
- {depth_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "location_closer_to_camera": """To determine which object is closer to the camera:

Step 1: Get 3D positions
- {object1} position: {pos1}
- {object2} position: {pos2}
- Camera position: (0, 0, 0)

Step 2: Calculate distances from camera
- Distance to {object1}: √({x1:.2f}² + {y1:.2f}² + {z1:.2f}²) = {dist1:.2f}
- Distance to {object2}: √({x2:.2f}² + {y2:.2f}² + {z2:.2f}²) = {dist2:.2f}

Step 3: Compare distances
- {dist1:.2f} {'<' if dist1 < dist2 else '>'} {dist2:.2f}
- {comparison}

Therefore, {conclusion} is closer to the camera.
The answer is {answer}.""",

        "location_above": """To determine if {object1} is above {object2}:

Step 1: Get 3D positions
- {object1} position: {pos1}
- {object2} position: {pos2}

Step 2: Compare Y-coordinates (height)
- {object1} Y-coordinate: {y1:.2f}
- {object2} Y-coordinate: {y2:.2f}
- In this system, larger Y = higher position

Step 3: Calculate height difference
- {object1} Y - {object2} Y = {y_diff:.2f}
- {height_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "location_next_to": """To determine if {object1} is next to {object2}:

Step 1: Get 3D positions
- {object1} position: {pos1}
- {object2} position: {pos2}

Step 2: Calculate horizontal distance (X-Z plane)
- ΔX = {dx:.2f}, ΔZ = {dz:.2f}
- Horizontal distance: √({dx:.2f}² + {dz:.2f}²) = {h_dist:.2f}

Step 3: Calculate vertical distance
- ΔY = {dy:.2f}

Step 4: Assess proximity
- {proximity_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "height_higher": """To determine which object is higher:

Step 1: Get 3D positions
- {object1} position: {pos1}
- {object2} position: {pos2}

Step 2: Compare Y-coordinates
- {object1} Y: {y1:.2f}
- {object2} Y: {y2:.2f}

Step 3: Determine which is higher
- Difference: {y1:.2f} - {y2:.2f} = {y_diff:.2f}
- {height_comparison}

Therefore, {conclusion} is higher.
The answer is {answer}.""",

        "multi_object_facing": """To determine if {object1} and {object2} are facing each other:

Step 1: Get positions and orientations
- {object1} at {pos1}, front direction: {front1}
- {object2} at {pos2}, front direction: {front2}

Step 2: Calculate vector between objects
- Vector from {object1} to {object2}: {vec_1to2}
- Vector from {object2} to {object1}: {vec_2to1}

Step 3: Check if fronts point toward each other
- Angle between {object1}'s front and vector to {object2}: {angle1:.1f}°
- Angle between {object2}'s front and vector to {object1}: {angle2:.1f}°

Step 4: Determine facing relationship
- If both angles < 90°, objects face each other
- {facing_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "multi_object_closer_to": """To determine which of {object1} or {object2} is closer to {object3}:

Step 1: Get all positions
- {object1} position: {pos1}
- {object2} position: {pos2}
- {object3} position: {pos3}

Step 2: Calculate distances to {object3}
- Distance from {object1} to {object3}: {dist1:.2f}
- Distance from {object2} to {object3}: {dist2:.2f}

Step 3: Compare
- {dist1:.2f} {'<' if dist1 < dist2 else '>'} {dist2:.2f}

Therefore, {conclusion} is closer to {object3}.
The answer is {answer}.""",

        "multi_object_parallel": """To determine if {object1} and {object2} are facing parallel directions:

Step 1: Get front directions
- {object1} front direction: {front1}
- {object2} front direction: {front2}

Step 2: Calculate angle between front directions
- Dot product: {dot_product:.3f}
- Angle: {angle:.1f}°

Step 3: Assess parallelism
- Parallel: angle ≈ 0° or 180°
- Perpendicular: angle ≈ 90°
- {parallel_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "multi_object_same_direction": """To determine if {object1} and {object2} face the same direction:

Step 1: Get front directions
- {object1} front direction: {front1}
- {object2} front direction: {front2}

Step 2: Calculate angle between directions
- Cosine similarity: {cos_sim:.3f}
- Angle: {angle:.1f}°

Step 3: Assess similarity
- Same direction: angle < 45°
- Opposite direction: angle > 135°
- Different directions: 45° < angle < 135°
- {direction_analysis}

Therefore, {conclusion}.
The answer is {answer}.""",

        "multi_object_viewpoint_towards_object": """To determine if the viewpoint direction points more toward {object1} or {object2}:

Step 1: Get positions
- Camera/viewpoint at: (0, 0, 0)
- Looking direction: (0, 0, -1) (negative Z)
- {object1} position: {pos1}
- {object2} position: {pos2}

Step 2: Calculate angles from viewing direction
- Vector to {object1}: {vec1}
- Angle to {object1}: {angle1:.1f}°
- Vector to {object2}: {vec2}
- Angle to {object2}: {angle2:.1f}°

Step 3: Compare angles
- Smaller angle = more aligned with view direction
- {angle_comparison}

Therefore, the viewpoint is more toward {conclusion}.
The answer is {answer}.""",
    }

    # Default template for uncategorized questions
    DEFAULT_TEMPLATE = """Analyzing the spatial relationship:

{original_reasoning}

Based on the 3D positions and orientations of the objects, {conclusion}.
The answer is {answer}."""

    def __init__(self, include_angle: bool = True, angle_precision: int = 1):
        """Initialize the generator.

        Args:
            include_angle: Whether to include view angle in output
            angle_precision: Decimal precision for angle values
        """
        self.include_angle = include_angle
        self.angle_precision = angle_precision

    @staticmethod
    def _format_vec(vec: List[float], precision: int = 1) -> str:
        """Format a vector as a tuple string."""
        formatted = [f"{v:.{precision}f}" for v in vec]
        return f"({', '.join(formatted)})"

    @staticmethod
    def _compute_angle(v1: List[float], v2: List[float]) -> float:
        """Compute angle between two vectors in degrees."""
        v1, v2 = np.array(v1), np.array(v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 90.0
        cos_sim = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
        return math.degrees(math.acos(cos_sim))

    @staticmethod
    def _compute_distance(pos: List[float]) -> float:
        """Compute distance from origin."""
        return math.sqrt(sum(x**2 for x in pos))

    @staticmethod
    def _vec_between(p1: List[float], p2: List[float]) -> List[float]:
        """Compute vector from p1 to p2."""
        return [p2[i] - p1[i] for i in range(len(p1))]

    def generate_cot(
        self,
        category: str,
        bounding_boxes: List[Dict],
        directions: List[Dict],
        answer: str,
        answer_name: str,
        original_cot: str = "",
    ) -> str:
        """Generate enhanced CoT for the given category.

        Args:
            category: Question category (e.g., 'orientation_viewpoint')
            bounding_boxes: List of object bounding boxes with 'bbox_3d' and 'label'
            directions: List of object directions with 'front_dir', 'left_dir', 'label'
            answer: Answer letter (A, B, C, D)
            answer_name: Full answer text (e.g., 'A. front')
            original_cot: Original CoT to use as fallback

        Returns:
            Generated enhanced CoT string
        """
        # Build lookup dictionaries
        bbox_by_label = {b['label']: b['bbox_3d'] for b in bounding_boxes if 'label' in b}
        dir_by_label = {d['label']: d for d in directions if 'label' in d}

        # Get labels
        labels = list(bbox_by_label.keys())

        try:
            if category == "orientation_viewpoint" and len(labels) >= 1:
                return self._gen_orientation_viewpoint(
                    labels[0], bbox_by_label, dir_by_label, answer, answer_name
                )
            elif category == "orientation_on_the_left" and len(labels) >= 1:
                return self._gen_orientation_left(
                    labels[0], bbox_by_label, answer, answer_name
                )
            elif category == "orientation_in_front_of" and len(labels) >= 2:
                return self._gen_orientation_in_front(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            elif category == "location_closer_to_camera" and len(labels) >= 2:
                return self._gen_location_closer(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            elif category == "location_above" and len(labels) >= 2:
                return self._gen_location_above(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            elif category == "location_next_to" and len(labels) >= 2:
                return self._gen_location_next_to(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            elif category == "height_higher" and len(labels) >= 2:
                return self._gen_height_higher(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            elif category == "multi_object_facing" and len(labels) >= 2:
                return self._gen_multi_facing(
                    labels[0], labels[1], bbox_by_label, dir_by_label, answer, answer_name
                )
            elif category == "multi_object_closer_to" and len(labels) >= 3:
                return self._gen_multi_closer_to(
                    labels[0], labels[1], labels[2], bbox_by_label, answer, answer_name
                )
            elif category == "multi_object_parallel" and len(labels) >= 2:
                return self._gen_multi_parallel(
                    labels[0], labels[1], dir_by_label, answer, answer_name
                )
            elif category == "multi_object_same_direction" and len(labels) >= 2:
                return self._gen_multi_same_direction(
                    labels[0], labels[1], dir_by_label, answer, answer_name
                )
            elif category == "multi_object_viewpoint_towards_object" and len(labels) >= 2:
                return self._gen_viewpoint_towards(
                    labels[0], labels[1], bbox_by_label, answer, answer_name
                )
            else:
                # Use original CoT if category not matched
                return original_cot if original_cot else f"The answer is {answer}."
        except Exception as e:
            logger.warning(f"Error generating CoT for {category}: {e}")
            return original_cot if original_cot else f"The answer is {answer}."

    def _gen_orientation_viewpoint(self, obj, bboxes, dirs, answer, answer_name):
        pos = bboxes[obj]
        d = dirs.get(obj, {})
        front = d.get('front_dir', [0, 0, -1])
        left = d.get('left_dir', [-1, 0, 0])

        # Camera-to-object vector
        to_cam = [-pos[0], -pos[1], -pos[2]]
        right = [-left[0], -left[1], -left[2]]
        back = [-front[0], -front[1], -front[2]]

        # Calculate angles
        angle_front = self._compute_angle(to_cam, front)
        angle_back = self._compute_angle(to_cam, back)
        angle_left = self._compute_angle(to_cam, left)
        angle_right = self._compute_angle(to_cam, right)

        angles = {'front': angle_front, 'back': angle_back, 'left': angle_left, 'right': angle_right}
        min_dir = min(angles, key=angles.get)
        min_angle = angles[min_dir]

        return self.CATEGORY_TEMPLATES["orientation_viewpoint"].format(
            object=obj, position=self._format_vec(pos),
            to_camera_vec=self._format_vec(to_cam),
            front_dir=self._format_vec(front), left_dir=self._format_vec(left),
            right_dir=self._format_vec(right), back_dir=self._format_vec(back),
            angle_front=angle_front, angle_back=angle_back,
            angle_left=angle_left, angle_right=angle_right,
            min_angle=min_angle, min_direction=min_dir, answer=answer
        )

    def _gen_orientation_left(self, obj, bboxes, answer, answer_name):
        pos = bboxes[obj]
        x = pos[0]

        if x < -0.5:
            x_analysis = "clearly negative (left of center)"
            conclusion = "on the left side of the image"
        elif x > 0.5:
            x_analysis = "clearly positive (right of center)"
            conclusion = "on the right side of the image"
        else:
            x_analysis = "near zero (close to center)"
            conclusion = "near the center of the image"

        return self.CATEGORY_TEMPLATES["orientation_on_the_left"].format(
            object=obj, position=self._format_vec(pos),
            x_coord=x, x_analysis=x_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_orientation_in_front(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        z1, z2 = pos1[2], pos2[2]
        z_diff = z1 - z2

        if z_diff < -0.5:
            depth_analysis = f"{obj1} is significantly closer (smaller Z)"
            conclusion = f"{obj1} is in front of {obj2}"
        elif z_diff > 0.5:
            depth_analysis = f"{obj2} is significantly closer (smaller Z)"
            conclusion = f"{obj2} is in front of {obj1}"
        else:
            depth_analysis = "Objects are at similar depths"
            conclusion = "neither is clearly in front of the other"

        return self.CATEGORY_TEMPLATES["orientation_in_front_of"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            z1=z1, z2=z2, z_diff=z_diff,
            depth_analysis=depth_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_location_closer(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        dist1 = self._compute_distance(pos1)
        dist2 = self._compute_distance(pos2)

        if dist1 < dist2:
            comparison = f"{obj1} is closer by {dist2 - dist1:.2f} units"
            conclusion = obj1
        else:
            comparison = f"{obj2} is closer by {dist1 - dist2:.2f} units"
            conclusion = obj2

        return self.CATEGORY_TEMPLATES["location_closer_to_camera"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            x1=pos1[0], y1=pos1[1], z1=pos1[2],
            x2=pos2[0], y2=pos2[1], z2=pos2[2],
            dist1=dist1, dist2=dist2,
            comparison=comparison, conclusion=conclusion, answer=answer
        )

    def _gen_location_above(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        y1, y2 = pos1[1], pos2[1]
        y_diff = y1 - y2

        if y_diff > 0.1:
            height_analysis = f"{obj1} is {y_diff:.2f} units higher"
            conclusion = f"{obj1} is above {obj2}"
        elif y_diff < -0.1:
            height_analysis = f"{obj2} is {-y_diff:.2f} units higher"
            conclusion = f"{obj2} is above {obj1}"
        else:
            height_analysis = "Objects are at similar heights"
            conclusion = "they are at roughly the same height"

        return self.CATEGORY_TEMPLATES["location_above"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            y1=y1, y2=y2, y_diff=y_diff,
            height_analysis=height_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_location_next_to(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        h_dist = math.sqrt(dx**2 + dz**2)

        if h_dist < 2.0 and abs(dy) < 1.0:
            proximity_analysis = f"Horizontal distance {h_dist:.2f} is small, objects are close"
            conclusion = f"{obj1} and {obj2} are next to each other"
        else:
            proximity_analysis = f"Distance is significant ({h_dist:.2f} horizontal, {abs(dy):.2f} vertical)"
            conclusion = f"{obj1} and {obj2} are not next to each other"

        return self.CATEGORY_TEMPLATES["location_next_to"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            dx=dx, dy=dy, dz=dz, h_dist=h_dist,
            proximity_analysis=proximity_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_height_higher(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        y1, y2 = pos1[1], pos2[1]
        y_diff = y1 - y2

        if y_diff > 0:
            height_comparison = f"{obj1} has a higher Y-coordinate"
            conclusion = obj1
        else:
            height_comparison = f"{obj2} has a higher Y-coordinate"
            conclusion = obj2

        return self.CATEGORY_TEMPLATES["height_higher"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            y1=y1, y2=y2, y_diff=y_diff,
            height_comparison=height_comparison, conclusion=conclusion, answer=answer
        )

    def _gen_multi_facing(self, obj1, obj2, bboxes, dirs, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        d1 = dirs.get(obj1, {'front_dir': [0, 0, -1]})
        d2 = dirs.get(obj2, {'front_dir': [0, 0, -1]})
        front1 = d1.get('front_dir', [0, 0, -1])
        front2 = d2.get('front_dir', [0, 0, -1])

        vec_1to2 = self._vec_between(pos1, pos2)
        vec_2to1 = self._vec_between(pos2, pos1)

        angle1 = self._compute_angle(front1, vec_1to2)
        angle2 = self._compute_angle(front2, vec_2to1)

        if angle1 < 90 and angle2 < 90:
            facing_analysis = f"Both angles ({angle1:.1f}°, {angle2:.1f}°) < 90°, so they face each other"
            conclusion = f"{obj1} and {obj2} are facing each other"
        else:
            facing_analysis = f"At least one angle > 90°, so they don't face each other"
            conclusion = f"{obj1} and {obj2} are not facing each other"

        return self.CATEGORY_TEMPLATES["multi_object_facing"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            front1=self._format_vec(front1), front2=self._format_vec(front2),
            vec_1to2=self._format_vec(vec_1to2), vec_2to1=self._format_vec(vec_2to1),
            angle1=angle1, angle2=angle2,
            facing_analysis=facing_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_multi_closer_to(self, obj1, obj2, obj3, bboxes, answer, answer_name):
        pos1, pos2, pos3 = bboxes[obj1], bboxes[obj2], bboxes[obj3]

        vec1 = self._vec_between(pos1, pos3)
        vec2 = self._vec_between(pos2, pos3)
        dist1 = math.sqrt(sum(x**2 for x in vec1))
        dist2 = math.sqrt(sum(x**2 for x in vec2))

        conclusion = obj1 if dist1 < dist2 else obj2

        return self.CATEGORY_TEMPLATES["multi_object_closer_to"].format(
            object1=obj1, object2=obj2, object3=obj3,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2), pos3=self._format_vec(pos3),
            dist1=dist1, dist2=dist2, conclusion=conclusion, answer=answer
        )

    def _gen_multi_parallel(self, obj1, obj2, dirs, answer, answer_name):
        d1 = dirs.get(obj1, {'front_dir': [0, 0, -1]})
        d2 = dirs.get(obj2, {'front_dir': [0, 0, -1]})
        front1 = np.array(d1.get('front_dir', [0, 0, -1]))
        front2 = np.array(d2.get('front_dir', [0, 0, -1]))

        dot = np.dot(front1/np.linalg.norm(front1), front2/np.linalg.norm(front2))
        angle = math.degrees(math.acos(np.clip(dot, -1, 1)))

        if angle < 30 or angle > 150:
            parallel_analysis = f"Angle {angle:.1f}° is close to 0° or 180°"
            conclusion = f"{obj1} and {obj2} are facing parallel directions"
        elif 60 < angle < 120:
            parallel_analysis = f"Angle {angle:.1f}° is close to 90°"
            conclusion = f"{obj1} and {obj2} are facing perpendicular directions"
        else:
            parallel_analysis = f"Angle {angle:.1f}° is neither parallel nor perpendicular"
            conclusion = f"{obj1} and {obj2} are facing different directions"

        return self.CATEGORY_TEMPLATES["multi_object_parallel"].format(
            object1=obj1, object2=obj2,
            front1=self._format_vec(front1.tolist()), front2=self._format_vec(front2.tolist()),
            dot_product=dot, angle=angle,
            parallel_analysis=parallel_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_multi_same_direction(self, obj1, obj2, dirs, answer, answer_name):
        d1 = dirs.get(obj1, {'front_dir': [0, 0, -1]})
        d2 = dirs.get(obj2, {'front_dir': [0, 0, -1]})
        front1 = np.array(d1.get('front_dir', [0, 0, -1]))
        front2 = np.array(d2.get('front_dir', [0, 0, -1]))

        cos_sim = np.dot(front1/np.linalg.norm(front1), front2/np.linalg.norm(front2))
        angle = math.degrees(math.acos(np.clip(cos_sim, -1, 1)))

        if angle < 45:
            direction_analysis = f"Angle {angle:.1f}° < 45°, very similar directions"
            conclusion = f"{obj1} and {obj2} face the same direction"
        elif angle > 135:
            direction_analysis = f"Angle {angle:.1f}° > 135°, opposite directions"
            conclusion = f"{obj1} and {obj2} face opposite directions"
        else:
            direction_analysis = f"Angle {angle:.1f}° is between 45° and 135°"
            conclusion = f"{obj1} and {obj2} face different directions"

        return self.CATEGORY_TEMPLATES["multi_object_same_direction"].format(
            object1=obj1, object2=obj2,
            front1=self._format_vec(front1.tolist()), front2=self._format_vec(front2.tolist()),
            cos_sim=cos_sim, angle=angle,
            direction_analysis=direction_analysis, conclusion=conclusion, answer=answer
        )

    def _gen_viewpoint_towards(self, obj1, obj2, bboxes, answer, answer_name):
        pos1, pos2 = bboxes[obj1], bboxes[obj2]
        view_dir = [0, 0, -1]  # Camera looks along -Z

        vec1 = pos1  # Vector from origin to obj1
        vec2 = pos2  # Vector from origin to obj2

        angle1 = self._compute_angle(view_dir, vec1)
        angle2 = self._compute_angle(view_dir, vec2)

        if angle1 < angle2:
            angle_comparison = f"{obj1} is more aligned with view ({angle1:.1f}° vs {angle2:.1f}°)"
            conclusion = obj1
        else:
            angle_comparison = f"{obj2} is more aligned with view ({angle2:.1f}° vs {angle1:.1f}°)"
            conclusion = obj2

        return self.CATEGORY_TEMPLATES["multi_object_viewpoint_towards_object"].format(
            object1=obj1, object2=obj2,
            pos1=self._format_vec(pos1), pos2=self._format_vec(pos2),
            vec1=self._format_vec(vec1), vec2=self._format_vec(vec2),
            angle1=angle1, angle2=angle2,
            angle_comparison=angle_comparison, conclusion=conclusion, answer=answer
        )


# ============================================================================
# Main Training Logic
# ============================================================================

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
    dataset_names = script_args.dataset_name.split('+')
    all_datasets = []

    for ds_name in dataset_names:
        ds_name = ds_name.strip()
        logger.info(f"Loading dataset: {ds_name}")
        ds = load_dataset(ds_name, split=script_args.dataset_train_split)
        ds = ds.add_column("_source_dataset", [ds_name] * len(ds))
        all_datasets.append(ds)

    if len(all_datasets) > 1:
        dataset = concatenate_datasets(all_datasets)
        logger.info(f"Concatenated {len(all_datasets)} datasets, total samples: {len(dataset)}")
    else:
        dataset = all_datasets[0]

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

    # Initialize enhanced CoT generator
    cot_generator = EnhancedCoTGenerator()

    # Get enhanced CoT mode setting
    use_enhanced_cot = getattr(training_args, 'use_enhanced_cot', True)

    # HF Augmented image directory mapping
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

        for pattern, subdir in hf_augmented_mapping.items():
            if pattern in source_dataset:
                base_name = os.path.splitext(image_filename)[0]
                png_filename = f"{base_name}.png"
                hf_image_path = os.path.join(hf_augmented_base_dir, subdir, png_filename)
                if os.path.exists(hf_image_path):
                    return hf_image_path
                hf_image_path_orig = os.path.join(hf_augmented_base_dir, subdir, image_filename)
                if os.path.exists(hf_image_path_orig):
                    return hf_image_path_orig

        original_path = os.path.join(training_args.data_dir, image_filename)
        if os.path.exists(original_path):
            return original_path

        return None

    def collate_fn(examples):
        samples = []
        for example in examples:
            if 'question' in example and example['question']:
                if example.get("A"):
                    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example.get(opt)]
                    question_text = example["question"]
                    options_text = "\n".join(options)
                    question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."
                else:
                    question = example["question"]

                user_content = []
                image_path = resolve_image_path(example)
                if image_path:
                    image = Image.open(image_path).convert("RGB")
                    user_content.append({"type": "image", "image": image})
                elif example.get("image_filename"):
                    logger.warning(f"Image not found for: {example.get('image_filename')}")

                user_content.append({"type": "text", "text": question})

                # Generate enhanced CoT or use original
                if use_enhanced_cot:
                    category = example.get('category', '')
                    bounding_boxes = example.get('bounding_box', [])
                    directions = example.get('direction', [])
                    answer = example.get('answer', '')
                    answer_name = example.get('answer_name', '')
                    original_cot = example.get('answer_cot', '')

                    answer_cot = cot_generator.generate_cot(
                        category=category,
                        bounding_boxes=bounding_boxes,
                        directions=directions,
                        answer=answer,
                        answer_name=answer_name,
                        original_cot=original_cot,
                    )
                else:
                    answer_cot = example.get('answer_cot', '')

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

        if not samples:
            return None

        batch = processor.apply_chat_template(
            samples,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=training_args.max_length,
            truncation=True
        )

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

    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": script_args.dataset_name,
        "tags": ["SpatialReasoner", "EnhancedCoT"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if model_args.use_peft:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    main(script_args, training_args, model_args)
