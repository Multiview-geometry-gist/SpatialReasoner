"""Spatial reasoning QA pair generation."""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import QAGenerationConfig
from .pose import ObjectPose


@dataclass
class QAPair:
    question: str
    answer: str
    answer_name: str  # "A", "B", etc.
    query_type: str  # "distance", "directional", "rotation"
    query_subtype: str
    objects_involved: List[int]
    options: Dict[str, str] = None
    gt_value: float = None
    gt_quaternion: np.ndarray = None


class QAGenerator:
    """Generate spatial reasoning QA pairs from object poses."""

    def __init__(self, config: QAGenerationConfig):
        self.config = config

    def generate(
        self, image_id: str, poses: List[ObjectPose], num_pairs: Optional[int] = None
    ) -> List[QAPair]:
        if len(poses) < self.config.min_objects_per_image:
            return []

        num_pairs = num_pairs or self.config.num_qa_per_image
        num_dist = int(num_pairs * self.config.distance_ratio)
        num_dir = int(num_pairs * self.config.directional_ratio)
        num_rot = num_pairs - num_dist - num_dir

        pairs = []
        pairs.extend(self._gen_distance(poses, num_dist))
        pairs.extend(self._gen_directional(poses, num_dir))
        pairs.extend(self._gen_rotation(poses, num_rot))

        random.shuffle(pairs)
        return pairs

    def _gen_distance(self, poses: List[ObjectPose], n: int) -> List[QAPair]:
        pairs = []
        for _ in range(n):
            if len(poses) < 2:
                break
            a, b = random.sample(poses, 2)
            dist = np.linalg.norm(a.position - b.position)
            correct = f"{dist:.2f} meters"
            distractors = [f"{dist * s:.2f} meters" for s in [0.5, 1.5, 2.0]]
            options, key = self._make_mcq(correct, distractors)

            name_a = a.label or f"object {a.object_id}"
            name_b = b.label or f"object {b.object_id}"

            pairs.append(QAPair(
                question=f"What is the distance between the {name_a} and the {name_b}?",
                answer=correct,
                answer_name=key,
                query_type="distance",
                query_subtype="distance_between",
                objects_involved=[a.object_id, b.object_id],
                options=options,
                gt_value=dist,
            ))
        return pairs

    def _gen_directional(self, poses: List[ObjectPose], n: int) -> List[QAPair]:
        templates = [
            ("Is the {a} to the left of the {b}?", "location_left", np.array([1, 0, 0]), "less"),
            ("Is the {a} to the right of the {b}?", "location_right", np.array([1, 0, 0]), "greater"),
            ("Is the {a} above the {b}?", "location_above", np.array([0, 1, 0]), "greater"),
            ("Is the {a} closer to the camera than the {b}?", "location_closer", np.array([0, 0, 1]), "less"),
        ]

        pairs = []
        for _ in range(n):
            if len(poses) < 2:
                break
            a, b = random.sample(poses, 2)
            tmpl, subtype, axis, cond = random.choice(templates)

            proj = np.dot(a.position - b.position, axis)
            is_true = (proj < 0) if cond == "less" else (proj > 0)

            name_a = a.label or f"object {a.object_id}"
            name_b = b.label or f"object {b.object_id}"

            pairs.append(QAPair(
                question=tmpl.format(a=name_a, b=name_b),
                answer="Yes" if is_true else "No",
                answer_name="A" if is_true else "B",
                query_type="directional",
                query_subtype=subtype,
                objects_involved=[a.object_id, b.object_id],
                options={"A": "Yes", "B": "No"},
                gt_value=proj,
            ))
        return pairs

    def _gen_rotation(self, poses: List[ObjectPose], n: int) -> List[QAPair]:
        pairs = []
        for _ in range(n):
            if len(poses) < 2:
                break
            if random.random() < 0.5:
                pairs.extend(self._gen_facing_toward(poses))
            else:
                pairs.extend(self._gen_same_direction(poses))
        return pairs[:n]

    def _gen_facing_toward(self, poses: List[ObjectPose]) -> List[QAPair]:
        if len(poses) < 2:
            return []
        a, b = random.sample(poses, 2)

        v_fwd = a.forward_direction
        dir_to_b = b.position - a.position
        dist = np.linalg.norm(dir_to_b)
        if dist < 1e-6:
            return []
        dir_to_b = dir_to_b / dist

        cos_sim = np.dot(v_fwd, dir_to_b)
        is_facing = cos_sim > self.config.rotation_threshold

        name_a = a.label or f"object {a.object_id}"
        name_b = b.label or f"object {b.object_id}"

        return [QAPair(
            question=f"Is the {name_a} facing toward the {name_b}?",
            answer="Yes" if is_facing else "No",
            answer_name="A" if is_facing else "B",
            query_type="rotation",
            query_subtype="orientation_facing_toward",
            objects_involved=[a.object_id, b.object_id],
            options={"A": "Yes", "B": "No"},
            gt_value=cos_sim,
            gt_quaternion=a.quaternion,
        )]

    def _gen_same_direction(self, poses: List[ObjectPose]) -> List[QAPair]:
        if len(poses) < 2:
            return []
        a, b = random.sample(poses, 2)

        cos_sim = np.abs(np.dot(a.forward_direction, b.forward_direction))
        is_same = cos_sim > self.config.rotation_threshold

        name_a = a.label or f"object {a.object_id}"
        name_b = b.label or f"object {b.object_id}"

        return [QAPair(
            question=f"Are the {name_a} and the {name_b} facing the same direction?",
            answer="Yes" if is_same else "No",
            answer_name="A" if is_same else "B",
            query_type="rotation",
            query_subtype="multi_object_same_direction",
            objects_involved=[a.object_id, b.object_id],
            options={"A": "Yes", "B": "No"},
            gt_value=cos_sim,
        )]

    def _make_mcq(self, correct: str, distractors: List[str]) -> Tuple[Dict[str, str], str]:
        opts = [correct] + distractors
        random.shuffle(opts)
        keys = ["A", "B", "C", "D"][:len(opts)]
        options = dict(zip(keys, opts))
        key = next(k for k, v in options.items() if v == correct)
        return options, key
