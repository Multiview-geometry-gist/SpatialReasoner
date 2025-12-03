"""Rotation-specific evaluation metrics."""

import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


ROTATION_SUBTYPES = [
    "orientation_facing_toward",
    "multi_object_same_direction",
    "orientation_in_front_of",
    "orientation_on_the_left",
    "orientation_viewpoint",
    "multi_object_facing",
    "multi_object_parallel",
]


@dataclass
class RotationEvalResult:
    """Results for rotation evaluation."""
    overall_accuracy: float
    accuracy_by_subtype: Dict[str, float] = field(default_factory=dict)
    geodesic_errors: List[float] = field(default_factory=list)
    mean_geodesic_error: float = 0.0
    median_geodesic_error: float = 0.0
    acc_at_5deg: float = 0.0
    acc_at_15deg: float = 0.0
    acc_at_30deg: float = 0.0


def evaluate_rotation_queries(
    predictions: List[str],
    ground_truths: List[str],
    query_subtypes: List[str],
    quaternion_gt: Optional[List[np.ndarray]] = None,
) -> RotationEvalResult:
    """
    Evaluate rotation-based spatial reasoning.

    Args:
        predictions: Model predictions (answer strings)
        ground_truths: Ground truth answers
        query_subtypes: Category for each query
        quaternion_gt: Optional quaternion ground truths

    Returns:
        RotationEvalResult with detailed metrics
    """
    correct_by_subtype = {s: [] for s in ROTATION_SUBTYPES}

    for pred, gt, subtype in zip(predictions, ground_truths, query_subtypes):
        if subtype not in ROTATION_SUBTYPES:
            continue

        pred_norm = pred.strip().lower()
        gt_norm = gt.strip().lower()

        # Binary comparison
        pred_yes = pred_norm in ["yes", "a", "true"]
        gt_yes = gt_norm in ["yes", "a", "true"]
        is_correct = (pred_yes == gt_yes) or (pred_norm == gt_norm)

        correct_by_subtype[subtype].append(int(is_correct))

    # Compute accuracy by subtype
    accuracy_by_subtype = {}
    for subtype, correct_list in correct_by_subtype.items():
        if correct_list:
            accuracy_by_subtype[subtype] = np.mean(correct_list)
        else:
            accuracy_by_subtype[subtype] = 0.0

    # Overall accuracy
    all_correct = sum(correct_by_subtype.values(), [])
    overall_accuracy = np.mean(all_correct) if all_correct else 0.0

    # Geodesic error computation
    geodesic_errors = []
    if quaternion_gt:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from spatial_reasoner.quaternion import parse_quaternion_from_text, geodesic_distance_np
        except ImportError:
            parse_quaternion_from_text = None

        if parse_quaternion_from_text:
            for pred, q_gt in zip(predictions, quaternion_gt):
                if q_gt is None:
                    continue
                q_pred = parse_quaternion_from_text(pred)
                if q_pred is not None:
                    error = geodesic_distance_np(q_pred, q_gt)
                    geodesic_errors.append(error)

    # Geodesic metrics
    if geodesic_errors:
        geo_deg = np.rad2deg(geodesic_errors)
        mean_geo = np.mean(geo_deg)
        median_geo = np.median(geo_deg)
        acc_5 = np.mean(geo_deg < 5)
        acc_15 = np.mean(geo_deg < 15)
        acc_30 = np.mean(geo_deg < 30)
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
    )


def print_rotation_report(result: RotationEvalResult):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("ROTATION EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOverall Rotation Accuracy: {result.overall_accuracy:.2%}")

    print("\nAccuracy by Query Subtype:")
    print("-" * 40)
    for subtype, acc in sorted(result.accuracy_by_subtype.items()):
        print(f"  {subtype:35s}: {acc:.2%}")

    if result.geodesic_errors:
        print("\nGeodesic Error Metrics:")
        print("-" * 40)
        print(f"  Mean Error:    {result.mean_geodesic_error:.2f} deg")
        print(f"  Median Error:  {result.median_geodesic_error:.2f} deg")
        print(f"  Acc@5deg:      {result.acc_at_5deg:.2%}")
        print(f"  Acc@15deg:     {result.acc_at_15deg:.2%}")
        print(f"  Acc@30deg:     {result.acc_at_30deg:.2%}")

    print("=" * 60 + "\n")


def load_and_evaluate(results_file: str, qa_file: str) -> RotationEvalResult:
    """Load results and QA file, then evaluate."""
    with open(results_file, "r") as f:
        results = json.load(f)

    with open(qa_file, "r") as f:
        qa_pairs = json.load(f)

    # Build lookup
    qa_lookup = {qa["image_id"]: qa for qa in qa_pairs}

    predictions = []
    ground_truths = []
    query_subtypes = []

    for result in results:
        image_id = result.get("image_id")
        if image_id and image_id in qa_lookup:
            qa = qa_lookup[image_id]
            predictions.append(result.get("prediction", ""))
            ground_truths.append(qa.get("answer", ""))
            query_subtypes.append(qa.get("query_subtype", ""))

    return evaluate_rotation_queries(predictions, ground_truths, query_subtypes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate rotation queries")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--qa", required=True, help="Path to QA pairs JSON")

    args = parser.parse_args()

    result = load_and_evaluate(args.results, args.qa)
    print_rotation_report(result)
