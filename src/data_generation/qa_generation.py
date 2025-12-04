"""Spatial reasoning QA pair generation.

Generates QA pairs for:
- Distance queries: "What is the distance between [A] and [B]?"
- Directional queries: "Is [A] to the left of [B]?"
- Rotation queries: "Is [A] facing toward [B]?", "Are [A] and [B] facing same direction?"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import random

from .config import QAGenerationConfig
from .pose_estimation import ObjectPose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_reasoner.utils import quaternion as quat
from spatial_reasoner.utils import geometry as geom


@dataclass
class QAPair:
    """A single question-answer pair with full pose data for CoT generation."""
    question: str
    answer: str
    answer_name: str  # A, B, C, D for MCQ
    query_type: str   # distance, directional, rotation
    query_subtype: str  # More specific category
    objects_involved: List[int]
    computation: str  # How answer was computed
    options: Dict[str, str] = field(default_factory=dict)
    gt_value: float = None
    gt_quaternion: np.ndarray = None

    # Pose data for CoT generation (matching REQUIRED_PHRASES patterns)
    name_a: str = None
    name_b: str = None
    name_c: str = None  # For multi_object queries
    position_a: np.ndarray = None
    position_b: np.ndarray = None
    position_c: np.ndarray = None
    forward_a: np.ndarray = None
    forward_b: np.ndarray = None
    left_a: np.ndarray = None
    left_b: np.ndarray = None
    distance_value: float = None
    cosine_value: float = None
    angle_value: float = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "question": self.question,
            "answer": self.answer,
            "answer_name": self.answer_name,
            "query_type": self.query_type,
            "query_subtype": self.query_subtype,
            "objects_involved": self.objects_involved,
            "computation": self.computation,
        }
        if self.options:
            result["A"] = self.options.get("A", "")
            result["B"] = self.options.get("B", "")
            result["C"] = self.options.get("C", "")
            result["D"] = self.options.get("D", "")
        if self.gt_value is not None:
            result["gt_value"] = self.gt_value
        if self.gt_quaternion is not None:
            result["gt_quaternion"] = self.gt_quaternion.tolist()
        return result


class QAGenerator:
    """Generate spatial reasoning QA pairs from object poses."""

    # All supported categories (matching REQUIRED_PHRASES in reward.py)
    CATEGORIES = [
        # Location
        'height_higher',
        'location_above',
        'location_closer_to_camera',
        'location_next_to',
        # Orientation
        'orientation_in_front_of',
        'orientation_on_the_left',
        'orientation_viewpoint',
        # Multi-object
        'multi_object_closer_to',
        'multi_object_parallel',
        'multi_object_same_direction',
        'multi_object_facing',
        'multi_object_viewpoint_towards_object',
        # Rotation (NEW)
        'rotation_angle_difference',
        'rotation_facing_toward',
    ]

    def __init__(self, config: QAGenerationConfig):
        self.config = config

        self.type_weights = {
            "distance": config.distance_ratio,
            "directional": config.directional_ratio,
            "rotation": config.rotation_ratio,
        }

    def generate(
        self,
        image_id: str,
        poses: List[ObjectPose],
        num_pairs: Optional[int] = None,
    ) -> List[QAPair]:
        """
        Generate QA pairs for an image.

        Args:
            image_id: Unique identifier for the image
            poses: List of object poses
            num_pairs: Number of QA pairs (default: config.num_qa_per_image)

        Returns:
            List of QAPair objects
        """
        if len(poses) < self.config.min_objects_per_image:
            return []

        num_pairs = num_pairs or self.config.num_qa_per_image

        # Calculate number of each type
        num_distance = int(num_pairs * self.type_weights["distance"])
        num_directional = int(num_pairs * self.type_weights["directional"])
        num_rotation = num_pairs - num_distance - num_directional

        qa_pairs = []
        qa_pairs.extend(self._generate_distance_queries(poses, num_distance))
        qa_pairs.extend(self._generate_directional_queries(poses, num_directional))
        qa_pairs.extend(self._generate_rotation_queries(poses, num_rotation))

        random.shuffle(qa_pairs)
        return qa_pairs[:num_pairs]

    def _generate_distance_queries(
        self,
        poses: List[ObjectPose],
        num_queries: int,
    ) -> List[QAPair]:
        """Generate distance-related queries."""
        pairs = []

        for _ in range(num_queries):
            if len(poses) < 2:
                break

            obj_a, obj_b = random.sample(poses, 2)
            distance = geom.euclidean_distance(obj_a.position, obj_b.position)

            correct_answer = f"{distance:.2f} meters"
            distractors = self._generate_distance_distractors(distance)
            options, answer_key = self._create_mcq_options(correct_answer, distractors)

            name_a = obj_a.label or f"object {obj_a.object_id}"
            name_b = obj_b.label or f"object {obj_b.object_id}"

            pairs.append(QAPair(
                question=f"What is the distance between the {name_a} and the {name_b}?",
                answer=correct_answer,
                answer_name=answer_key,
                query_type="distance",
                query_subtype="location_next_to",
                objects_involved=[obj_a.object_id, obj_b.object_id],
                computation=f"||t_A - t_B|| = {distance:.4f}",
                options=options,
                gt_value=distance,
                # Pose data for CoT generation
                name_a=name_a,
                name_b=name_b,
                position_a=obj_a.position,
                position_b=obj_b.position,
                distance_value=distance,
            ))

        return pairs

    def _generate_directional_queries(
        self,
        poses: List[ObjectPose],
        num_queries: int,
    ) -> List[QAPair]:
        """Generate directional relationship queries."""
        pairs = []

        query_templates = [
            {
                "template": "Is the {name_a} above the {name_b}?",
                "subtype": "location_above",
                "check": lambda a, b: geom.is_above(a.position, b.position),
            },
            {
                "template": "Is the {name_a} to the left of the {name_b}?",
                "subtype": "orientation_on_the_left",
                "check": lambda a, b: geom.is_left_of(a.position, b.position),
            },
            {
                "template": "Is the {name_a} closer to the camera than the {name_b}?",
                "subtype": "location_closer_to_camera",
                "check": lambda a, b: geom.is_closer_to_camera(a.position, b.position),
            },
            {
                "template": "Is the {name_a} in front of the {name_b}?",
                "subtype": "orientation_in_front_of",
                "check": lambda a, b: geom.is_in_front_of(a.position, b.position),
            },
            {
                "template": "Is the {name_a} higher than the {name_b}?",
                "subtype": "height_higher",
                "check": lambda a, b: geom.is_above(a.position, b.position),
            },
        ]

        for _ in range(num_queries):
            if len(poses) < 2:
                break

            obj_a, obj_b = random.sample(poses, 2)
            template_info = random.choice(query_templates)

            is_true = template_info["check"](obj_a, obj_b)

            correct_answer = "Yes" if is_true else "No"
            options = {"A": "Yes", "B": "No"}
            answer_key = "A" if is_true else "B"

            name_a = obj_a.label or f"object {obj_a.object_id}"
            name_b = obj_b.label or f"object {obj_b.object_id}"

            # Compute additional values for CoT
            direction_to_b = geom.direction_vector(obj_a.position, obj_b.position)
            distance = geom.euclidean_distance(obj_a.position, obj_b.position)
            dist_a = np.linalg.norm(obj_a.position)
            dist_b = np.linalg.norm(obj_b.position)

            # Compute cosine/angle for orientation queries
            if template_info["subtype"] in ["orientation_in_front_of", "orientation_on_the_left"]:
                cos_sim = geom.cosine_similarity(obj_a.forward_direction, direction_to_b)
                angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            else:
                cos_sim = None
                angle = None

            pairs.append(QAPair(
                question=template_info["template"].format(name_a=name_a, name_b=name_b),
                answer=correct_answer,
                answer_name=answer_key,
                query_type="directional",
                query_subtype=template_info["subtype"],
                objects_involved=[obj_a.object_id, obj_b.object_id],
                computation=f"Check: {template_info['subtype']} = {is_true}",
                options=options,
                # Pose data for CoT generation
                name_a=name_a,
                name_b=name_b,
                position_a=obj_a.position,
                position_b=obj_b.position,
                forward_a=obj_a.forward_direction,
                forward_b=obj_b.forward_direction,
                left_a=obj_a.left_direction if hasattr(obj_a, 'left_direction') else None,
                distance_value=distance,
                cosine_value=cos_sim,
                angle_value=angle,
            ))

        return pairs

    def _generate_rotation_queries(
        self,
        poses: List[ObjectPose],
        num_queries: int,
    ) -> List[QAPair]:
        """Generate rotation-aware queries."""
        pairs = []

        query_types = ["facing_toward", "same_direction", "parallel"]

        for _ in range(num_queries):
            if len(poses) < 2:
                break

            query_type = random.choice(query_types)

            if query_type == "facing_toward":
                qa = self._generate_facing_toward_query(poses)
            elif query_type == "same_direction":
                qa = self._generate_same_direction_query(poses)
            else:
                qa = self._generate_parallel_query(poses)

            if qa:
                pairs.extend(qa)

        return pairs[:num_queries]

    def _generate_facing_toward_query(self, poses: List[ObjectPose]) -> List[QAPair]:
        """Generate 'Is [A] facing toward [B]?' query."""
        if len(poses) < 2:
            return []

        obj_a, obj_b = random.sample(poses, 2)

        # Check if A is facing toward B
        v_fwd = obj_a.forward_direction
        direction_to_b = geom.direction_vector(obj_a.position, obj_b.position)
        cos_sim = geom.cosine_similarity(v_fwd, direction_to_b)
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

        is_facing = cos_sim > self.config.rotation_threshold

        correct_answer = "Yes" if is_facing else "No"
        options = {"A": "Yes", "B": "No"}
        answer_key = "A" if is_facing else "B"

        name_a = obj_a.label or f"object {obj_a.object_id}"
        name_b = obj_b.label or f"object {obj_b.object_id}"

        return [QAPair(
            question=f"Is the {name_a} facing toward the {name_b}?",
            answer=correct_answer,
            answer_name=answer_key,
            query_type="rotation",
            query_subtype="rotation_facing_toward",
            objects_involved=[obj_a.object_id, obj_b.object_id],
            computation=f"v_fwd . direction = {cos_sim:.4f} (threshold: {self.config.rotation_threshold})",
            options=options,
            gt_value=cos_sim,
            gt_quaternion=obj_a.quaternion,
            # Pose data for CoT generation
            name_a=name_a,
            name_b=name_b,
            position_a=obj_a.position,
            position_b=obj_b.position,
            forward_a=v_fwd,
            cosine_value=cos_sim,
            angle_value=angle,
        )]

    def _generate_same_direction_query(self, poses: List[ObjectPose]) -> List[QAPair]:
        """Generate 'Are [A] and [B] facing same direction?' query."""
        if len(poses) < 2:
            return []

        obj_a, obj_b = random.sample(poses, 2)

        v_fwd_a = obj_a.forward_direction
        v_fwd_b = obj_b.forward_direction
        cos_sim = abs(geom.cosine_similarity(v_fwd_a, v_fwd_b))
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

        is_same = cos_sim > self.config.rotation_threshold

        correct_answer = "Yes" if is_same else "No"
        options = {"A": "Yes", "B": "No"}
        answer_key = "A" if is_same else "B"

        name_a = obj_a.label or f"object {obj_a.object_id}"
        name_b = obj_b.label or f"object {obj_b.object_id}"

        return [QAPair(
            question=f"Are the {name_a} and the {name_b} facing the same direction?",
            answer=correct_answer,
            answer_name=answer_key,
            query_type="rotation",
            query_subtype="multi_object_same_direction",
            objects_involved=[obj_a.object_id, obj_b.object_id],
            computation=f"|v_fwd^A . v_fwd^B| = {cos_sim:.4f}",
            options=options,
            gt_value=cos_sim,
            # Pose data for CoT generation
            name_a=name_a,
            name_b=name_b,
            forward_a=v_fwd_a,
            forward_b=v_fwd_b,
            cosine_value=cos_sim,
            angle_value=angle,
        )]

    def _generate_parallel_query(self, poses: List[ObjectPose]) -> List[QAPair]:
        """Generate 'Are [A] and [B] parallel?' query."""
        if len(poses) < 2:
            return []

        obj_a, obj_b = random.sample(poses, 2)

        v_fwd_a = obj_a.forward_direction
        v_fwd_b = obj_b.forward_direction
        cos_sim = abs(geom.cosine_similarity(v_fwd_a, v_fwd_b))
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

        is_parallel = cos_sim > 0.9

        correct_answer = "Yes" if is_parallel else "No"
        options = {"A": "Yes", "B": "No"}
        answer_key = "A" if is_parallel else "B"

        name_a = obj_a.label or f"object {obj_a.object_id}"
        name_b = obj_b.label or f"object {obj_b.object_id}"

        return [QAPair(
            question=f"Are the {name_a} and the {name_b} parallel to each other?",
            answer=correct_answer,
            answer_name=answer_key,
            query_type="rotation",
            query_subtype="multi_object_parallel",
            objects_involved=[obj_a.object_id, obj_b.object_id],
            computation=f"|v_fwd^A . v_fwd^B| = {cos_sim:.4f} (threshold: 0.9)",
            options=options,
            gt_value=cos_sim,
            # Pose data for CoT generation
            name_a=name_a,
            name_b=name_b,
            forward_a=v_fwd_a,
            forward_b=v_fwd_b,
            cosine_value=cos_sim,
            angle_value=angle,
        )]

    def _generate_distance_distractors(
        self,
        correct_distance: float,
        num_distractors: int = 3,
    ) -> List[str]:
        """Generate plausible wrong answers for distance queries."""
        distractors = []

        scales = [0.5, 1.5, 2.0, 0.7, 1.3]
        random.shuffle(scales)

        for scale in scales[:num_distractors]:
            wrong_distance = correct_distance * scale
            wrong_distance += random.uniform(-0.1, 0.1) * correct_distance
            wrong_distance = max(0.01, wrong_distance)
            distractors.append(f"{wrong_distance:.2f} meters")

        return distractors

    def _create_mcq_options(
        self,
        correct_answer: str,
        distractors: List[str],
    ) -> Tuple[Dict[str, str], str]:
        """Create shuffled MCQ options."""
        all_options = [correct_answer] + distractors
        random.shuffle(all_options)

        keys = ["A", "B", "C", "D"][:len(all_options)]
        options = {k: v for k, v in zip(keys, all_options)}

        correct_key = [k for k, v in options.items() if v == correct_answer][0]
        return options, correct_key

    def format_for_training(
        self,
        qa: QAPair,
        image_filename: str,
        question_index: str,
    ) -> Dict:
        """
        Format QA for SpatialReasoner training format.

        Matches the expected format in sft.py and grpo.py.
        """
        result = qa.to_dict()
        result["image_filename"] = image_filename
        result["question_index"] = question_index
        result["category"] = qa.query_subtype

        # Generate chain-of-thought answer for SFT
        result["answer_cot"] = self._generate_cot(qa)

        return result

    def _format_pos(self, pos: np.ndarray) -> str:
        """Format position as (x.xx, y.yy, z.zz)."""
        if pos is None:
            return "(0.00, 0.00, 0.00)"
        return f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

    def _format_vec(self, vec: np.ndarray) -> str:
        """Format vector as (x.xx, y.yy, z.zz)."""
        if vec is None:
            return "(0.00, 0.00, 0.00)"
        return f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"

    def _generate_cot(self, qa: QAPair) -> str:
        """
        Generate chain-of-thought answer matching REQUIRED_PHRASES patterns.

        Pattern requirements from reward.py:
        - location: "3D location of ... (x, y, z)"
        - vector: "vector from ... to ... (x, y, z)"
        - dist_from: "distance from ... to the camera ... N.NN"
        - dist_between: "distance between ... N.NN"
        - cosine: "cosine similarity between ... N.NN"
        - angle: "angle between ... N"
        - front: "front direction of ... (x, y, z)"
        - left: "left direction of ... (x, y, z)"
        - rotation_angle: "rotation angle ... N degrees"
        """
        subtype = qa.query_subtype
        lines = ["Let me analyze this spatial reasoning question step by step."]

        # Generate patterns based on category
        if subtype == "location_next_to":
            # Requires: 2x location, 1x dist_between
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            lines.append(f"The distance between the {qa.name_a} and the {qa.name_b} is {qa.distance_value:.2f}.")

        elif subtype in ["height_higher", "location_above"]:
            # Requires: 2x location
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            if qa.position_a is not None and qa.position_b is not None:
                lines.append(f"Comparing the y-coordinates: {qa.position_a[1]:.2f} vs {qa.position_b[1]:.2f}.")

        elif subtype == "location_closer_to_camera":
            # Requires: 2x location, 2x dist_from
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            if qa.position_a is not None:
                dist_a = np.linalg.norm(qa.position_a)
                lines.append(f"The distance from the {qa.name_a} to the camera is {dist_a:.2f}.")
            if qa.position_b is not None:
                dist_b = np.linalg.norm(qa.position_b)
                lines.append(f"The distance from the {qa.name_b} to the camera is {dist_b:.2f}.")

        elif subtype == "orientation_in_front_of":
            # Requires: 2x location, vector, front, (cosine or angle)
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            if qa.position_a is not None and qa.position_b is not None:
                direction = qa.position_b - qa.position_a
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                lines.append(f"The vector from the {qa.name_a} to the {qa.name_b} is {self._format_vec(direction)}.")
            lines.append(f"The front direction of the {qa.name_a} is {self._format_vec(qa.forward_a)}.")
            if qa.cosine_value is not None:
                lines.append(f"The cosine similarity between these vectors is {qa.cosine_value:.2f}.")
            if qa.angle_value is not None:
                lines.append(f"The angle between these vectors is {int(qa.angle_value)} degrees.")

        elif subtype == "orientation_on_the_left":
            # Requires: 2x location, vector, left, (cosine or angle)
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            if qa.position_a is not None and qa.position_b is not None:
                direction = qa.position_b - qa.position_a
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                lines.append(f"The vector from the {qa.name_a} to the {qa.name_b} is {self._format_vec(direction)}.")
            if qa.left_a is not None:
                lines.append(f"The left direction of the {qa.name_a} is {self._format_vec(qa.left_a)}.")
            if qa.cosine_value is not None:
                lines.append(f"The cosine similarity between these vectors is {qa.cosine_value:.2f}.")
            if qa.angle_value is not None:
                lines.append(f"The angle between these vectors is {int(qa.angle_value)} degrees.")

        elif subtype in ["multi_object_parallel", "multi_object_same_direction"]:
            # Requires: 2x front, (cosine or angle)
            lines.append(f"The front direction of the {qa.name_a} is {self._format_vec(qa.forward_a)}.")
            lines.append(f"The front direction of the {qa.name_b} is {self._format_vec(qa.forward_b)}.")
            if qa.cosine_value is not None:
                lines.append(f"The cosine similarity between their forward vectors is {qa.cosine_value:.2f}.")
            if qa.angle_value is not None:
                lines.append(f"The angle between their forward vectors is {int(qa.angle_value)} degrees.")

        elif subtype == "rotation_facing_toward":
            # Requires: 2x location, front, vector, (cosine or angle)
            lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            lines.append(f"The front direction of the {qa.name_a} is {self._format_vec(qa.forward_a)}.")
            if qa.position_a is not None and qa.position_b is not None:
                direction = qa.position_b - qa.position_a
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                lines.append(f"The vector from the {qa.name_a} to the {qa.name_b} is {self._format_vec(direction)}.")
            if qa.cosine_value is not None:
                lines.append(f"The cosine similarity between these vectors is {qa.cosine_value:.2f}.")
            if qa.angle_value is not None:
                lines.append(f"The angle between these vectors is {int(qa.angle_value)} degrees.")

        elif subtype == "rotation_angle_difference":
            # Requires: 2x front, (rotation_angle or angle)
            lines.append(f"The front direction of the {qa.name_a} is {self._format_vec(qa.forward_a)}.")
            lines.append(f"The front direction of the {qa.name_b} is {self._format_vec(qa.forward_b)}.")
            if qa.angle_value is not None:
                lines.append(f"The rotation angle difference is {int(qa.angle_value)} degrees.")

        else:
            # Fallback for other categories
            if qa.position_a is not None:
                lines.append(f"The 3D location of the {qa.name_a} is {self._format_pos(qa.position_a)}.")
            if qa.position_b is not None:
                lines.append(f"The 3D location of the {qa.name_b} is {self._format_pos(qa.position_b)}.")
            if qa.distance_value is not None:
                lines.append(f"The distance between them is {qa.distance_value:.2f}.")

        # Add conclusion
        lines.append(f"Based on this analysis, the answer is {qa.answer}.")

        think_content = "\n".join(lines)
        return f"<think>{think_content}</think><answer>{qa.answer_name}. {qa.answer}</answer>"
