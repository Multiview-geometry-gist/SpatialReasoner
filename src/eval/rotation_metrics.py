"""Rotation-specific evaluation metrics.

Evaluates:
- Binary rotation query accuracy
- Geodesic error distribution (if quaternion outputs available)
- Per-category breakdown
"""

import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Import quaternion utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from spatial_reasoner.utils import quaternion as quat
except ImportError:
    quat = None


@dataclass
class RotationEvalResult:
    """Results for rotation evaluation."""
    overall_accuracy: float
    accuracy_by_subtype: Dict[str, float]
    geodesic_errors: List[float]
    mean_geodesic_error: float
    median_geodesic_error: float
    acc_at_5deg: float
    acc_at_15deg: float
    acc_at_30deg: float
    num_samples: int


def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response."""
    # Try <answer> tags first
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_quaternion_from_response(response: str) -> Optional[np.ndarray]:
    """Extract quaternion from model response."""
    patterns = [
        r"quaternion[:\s]*[\(\[]?\s*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\s*[\)\]]?",
        r"orientation[:\s]*[\(\[]?\s*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\s*[\)\]]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                values = [float(match.group(i)) for i in range(1, 5)]
                q = np.array(values)
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    return q / norm
            except:
                pass

    return None


def evaluate_rotation_queries(
    predictions: List[str],
    ground_truths: List[str],
    query_subtypes: List[str],
    quaternion_gt: Optional[List[np.ndarray]] = None,
) -> RotationEvalResult:
    """
    Evaluate rotation-based spatial reasoning.

    Args:
        predictions: Model predictions (full response strings)
        ground_truths: Ground truth answers
        query_subtypes: Category for each query
        quaternion_gt: Optional quaternion ground truths

    Returns:
        RotationEvalResult with detailed metrics
    """
    # Rotation-related subtypes
    rotation_subtypes = {
        "rotation_facing_toward",
        "rotation_angle_difference",
        "multi_object_same_direction",
        "multi_object_parallel",
        "multi_object_facing",
        "orientation_in_front_of",
        "orientation_on_the_left",
        "orientation_viewpoint",
    }

    correct_by_subtype = {s: [] for s in rotation_subtypes}
    geodesic_errors = []

    for i, (pred, gt, subtype) in enumerate(zip(predictions, ground_truths, query_subtypes)):
        if subtype not in rotation_subtypes:
            continue

        # Extract answer
        pred_answer = extract_answer_from_response(pred)
        gt_answer = gt.strip() if isinstance(gt, str) else str(gt)

        # Binary comparison for Yes/No questions
        pred_lower = pred_answer.lower()
        gt_lower = gt_answer.lower()

        # Check for correct answer
        is_correct = False

        # Direct match
        if pred_lower == gt_lower:
            is_correct = True
        # Yes/No matching
        elif (("yes" in pred_lower or pred_lower.startswith("a")) and
              ("yes" in gt_lower or gt_lower.startswith("a"))):
            is_correct = True
        elif (("no" in pred_lower or pred_lower.startswith("b")) and
              ("no" in gt_lower or gt_lower.startswith("b"))):
            is_correct = True
        # Option matching (A/B/C/D)
        elif len(pred_lower) >= 1 and len(gt_lower) >= 1:
            if pred_lower[0] == gt_lower[0] and pred_lower[0] in 'abcd':
                is_correct = True

        correct_by_subtype[subtype].append(int(is_correct))

        # Geodesic error computation
        if quaternion_gt is not None and i < len(quaternion_gt) and quaternion_gt[i] is not None:
            q_pred = extract_quaternion_from_response(pred)
            if q_pred is not None and quat is not None:
                try:
                    q_gt = np.array(quaternion_gt[i])
                    error_rad = quat.geodesic_distance(q_pred, q_gt)
                    error_deg = np.degrees(error_rad)
                    geodesic_errors.append(error_deg)
                except:
                    pass

    # Compute metrics
    accuracy_by_subtype = {}
    all_correct = []

    for subtype, correct_list in correct_by_subtype.items():
        if len(correct_list) > 0:
            accuracy_by_subtype[subtype] = np.mean(correct_list)
            all_correct.extend(correct_list)
        else:
            accuracy_by_subtype[subtype] = 0.0

    overall_accuracy = np.mean(all_correct) if all_correct else 0.0

    # Geodesic metrics
    if geodesic_errors:
        mean_geo = np.mean(geodesic_errors)
        median_geo = np.median(geodesic_errors)
        acc_5 = np.mean([e <= 5 for e in geodesic_errors])
        acc_15 = np.mean([e <= 15 for e in geodesic_errors])
        acc_30 = np.mean([e <= 30 for e in geodesic_errors])
    else:
        mean_geo = median_geo = 0.0
        acc_5 = acc_15 = acc_30 = 0.0

    return RotationEvalResult(
        overall_accuracy=overall_accuracy,
        accuracy_by_subtype=accuracy_by_subtype,
        geodesic_errors=geodesic_errors,
        mean_geodesic_error=mean_geo,
        median_geodesic_error=median_geo,
        acc_at_5deg=acc_5,
        acc_at_15deg=acc_15,
        acc_at_30deg=acc_30,
        num_samples=len(all_correct),
    )


def print_rotation_eval_report(result: RotationEvalResult):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("ROTATION EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal samples evaluated: {result.num_samples}")
    print(f"Overall Rotation Accuracy: {result.overall_accuracy:.2%}")

    print("\nAccuracy by Query Subtype:")
    print("-" * 40)
    for subtype, acc in sorted(result.accuracy_by_subtype.items()):
        if acc > 0:  # Only show subtypes with data
            print(f"  {subtype:40s}: {acc:.2%}")

    if result.geodesic_errors:
        print("\nGeodesic Error Metrics:")
        print("-" * 40)
        print(f"  Num samples with quaternion: {len(result.geodesic_errors)}")
        print(f"  Mean Error:    {result.mean_geodesic_error:.2f} deg")
        print(f"  Median Error:  {result.median_geodesic_error:.2f} deg")
        print(f"  Acc@5deg:      {result.acc_at_5deg:.2%}")
        print(f"  Acc@15deg:     {result.acc_at_15deg:.2%}")
        print(f"  Acc@30deg:     {result.acc_at_30deg:.2%}")

    print("=" * 60 + "\n")


def evaluate_3dsrbench_with_rotation(
    results_df,
    include_rotation: bool = True
) -> Dict[str, float]:
    """
    Extended 3DSRBench evaluation including rotation categories.

    Args:
        results_df: DataFrame with columns: 'prediction', 'ground_truth', 'category'
        include_rotation: Whether to include rotation-specific metrics

    Returns:
        Dictionary of metrics
    """
    # Standard 3DSRBench categories
    mapping = {
        'location': ['location_above', 'location_closer_to_camera', 'location_next_to'],
        'height': ['height_higher'],
        'orientation': ['orientation_in_front_of', 'orientation_on_the_left', 'orientation_viewpoint'],
        'multi_object': ['multi_object_closer_to', 'multi_object_facing',
                         'multi_object_viewpoint_towards_object', 'multi_object_parallel',
                         'multi_object_same_direction'],
    }

    if include_rotation:
        mapping['rotation'] = [
            'rotation_facing_toward',
            'rotation_angle_difference',
        ]

    metrics = {}

    # Compute per-category metrics
    for cat_name, subtypes in mapping.items():
        cat_mask = results_df['category'].isin(subtypes)
        if cat_mask.sum() > 0:
            cat_df = results_df[cat_mask]
            accuracy = (cat_df['prediction'] == cat_df['ground_truth']).mean()
            metrics[f'accuracy_{cat_name}'] = accuracy

    # Overall accuracy
    metrics['accuracy_overall'] = (results_df['prediction'] == results_df['ground_truth']).mean()

    return metrics


if __name__ == "__main__":
    # Example usage
    predictions = [
        "<think>Analysis...</think><answer>A. Yes</answer>",
        "<think>Analysis...</think><answer>B. No</answer>",
    ]
    ground_truths = ["A", "B"]
    subtypes = ["rotation_facing_toward", "multi_object_same_direction"]

    result = evaluate_rotation_queries(predictions, ground_truths, subtypes)
    print_rotation_eval_report(result)
