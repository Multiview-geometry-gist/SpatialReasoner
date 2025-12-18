"""Update augmented_dataset.json with generated multi-view images.

This script scans the MVGenMaster output directories and updates the
augmented_dataset.json to include:
1. Multiple view image paths in view_images
2. Transformed CoT coordinates for each view angle in augmented_cots

The CoT transformation uses Y-axis rotation to match the MVGenMaster camera azimuth rotation.

Usage:
    python -m src.data_generation.update_multiview_dataset \
        --dataset_path /data/SpatialReasoner/data/mvgen_augmented/augmented_dataset.json \
        --images_dir /data/SpatialReasoner/data/mvgen_augmented/images \
        --output_path /data/SpatialReasoner/data/mvgen_augmented/augmented_dataset_multiview.json
"""

import os
import re
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from src.spatial_reasoner.utils.geometry import rotation_matrix_y

logger = logging.getLogger(__name__)


# ============================================================================
# Y-axis Rotation for Coordinate Transformation
# ============================================================================

def rotate_y_axis(coords: List[float], angle_deg: float) -> List[float]:
    """Rotate 3D coordinates around Y-axis (azimuth/yaw).

    This matches the MVGenMaster camera rotation where azimuth is Y-axis rotation.
    When camera rotates by +phi (azimuth), object coordinates in camera frame
    rotate by -phi (inverse transformation).

    Y-axis rotation keeps the Y coordinate (height) unchanged while rotating X and Z.
    This is the correct behavior for horizontal camera movement around a subject.

    Args:
        coords: [x, y, z] coordinates
        angle_deg: Rotation angle in degrees (positive = counterclockwise when viewed from +Y)

    Returns:
        Rotated [x', y', z'] coordinates with Y unchanged
    """
    # Negative angle because camera moves, not object
    rot_matrix = rotation_matrix_y(-angle_deg, degrees=True)
    coords_array = np.array(coords)
    rotated = rot_matrix @ coords_array
    return rotated.tolist()


def format_coord(value: float, precision: int = 1) -> str:
    """Format coordinate value with proper precision."""
    if abs(value) < 0.05:
        return "0.0" if value >= 0 else "-0.0"
    return f"{value:.{precision}f}"


def format_coord_tuple(coords: List[float], precision: int = 1) -> str:
    """Format coordinates as tuple string."""
    formatted = [format_coord(c, precision) for c in coords]
    return f"({', '.join(formatted)})"


def compute_angle_degrees(vec1: List[float], vec2: List[float]) -> float:
    """Compute angle between two vectors in degrees."""
    v1, v2 = np.array(vec1), np.array(vec2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:
        return 90.0

    cos_sim = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_sim))


# ============================================================================
# CoT Coordinate Transformation
# ============================================================================

def transform_cot_for_angle(
    original_cot: str,
    angle_deg: float,
    original_bbox: Optional[List[float]] = None,
    original_front_dir: Optional[List[float]] = None,
    original_left_dir: Optional[List[float]] = None,
    label: str = "the object",
) -> str:
    """Transform CoT coordinates for a specific view angle.

    Applies Y-axis rotation to all coordinate patterns in the CoT text.
    This matches MVGenMaster's azimuth rotation which is around the Y-axis.

    Args:
        original_cot: Original chain-of-thought text
        angle_deg: View angle in degrees
        original_bbox: Original bounding box center [x, y, z]
        original_front_dir: Original front direction vector
        original_left_dir: Original left direction vector
        label: Object label for replacement

    Returns:
        Transformed CoT with rotated coordinates
    """
    if abs(angle_deg) < 0.5:
        # No rotation needed for 0 degrees
        return original_cot

    # If we have original geometry data, use specific transformations
    if original_bbox and original_front_dir and original_left_dir:
        return _transform_cot_with_geometry(
            original_cot, angle_deg,
            original_bbox, original_front_dir, original_left_dir, label
        )

    # Otherwise, use regex-based coordinate transformation
    return _transform_cot_regex(original_cot, angle_deg)


def _transform_cot_with_geometry(
    cot: str,
    angle_deg: float,
    bbox: List[float],
    front_dir: List[float],
    left_dir: List[float],
    label: str,
) -> str:
    """Transform CoT using known geometry data."""
    # Rotate all vectors around Y-axis (azimuth rotation)
    new_bbox = rotate_y_axis(bbox, angle_deg)
    new_front_dir = rotate_y_axis(front_dir, angle_deg)
    new_left_dir = rotate_y_axis(left_dir, angle_deg)

    # Compute derived values
    camera_vec = [-new_bbox[0], -new_bbox[1], -new_bbox[2]]
    right_dir = [-new_left_dir[0], -new_left_dir[1], -new_left_dir[2]]
    back_dir = [-new_front_dir[0], -new_front_dir[1], -new_front_dir[2]]

    # Compute angles
    angle_left = compute_angle_degrees(camera_vec, new_left_dir)
    angle_right = 180.0 - angle_left if angle_left <= 180 else angle_left - 180
    angle_front = compute_angle_degrees(camera_vec, new_front_dir)
    angle_back = 180.0 - angle_front if angle_front <= 180 else angle_front - 180

    # Cosine similarities
    def cos_sim(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1/n1, v2/n2))

    cos_left = cos_sim(camera_vec, new_left_dir)
    cos_front = cos_sim(camera_vec, new_front_dir)

    # Apply transformations
    transformed = cot

    # Pattern 1: Location coordinates
    loc_pattern = r"The location of [^(]+ is \([-\d.]+,\s*[-\d.]+,\s*[-\d.]+\)"
    loc_replacement = f"The location of {label} is {format_coord_tuple(new_bbox)}"
    transformed = re.sub(loc_pattern, loc_replacement, transformed)

    # Pattern 2: Vector to camera
    cam_pattern = r"(vector from [^(]+ to camera is hence )\([-\d.]+,\s*[-\d.]+,\s*[-\d.]+\)"
    cam_replacement = f"\\1{format_coord_tuple(camera_vec)}"
    transformed = re.sub(cam_pattern, cam_replacement, transformed)

    # Pattern 3: Left direction
    left_pattern = r"The left direction of [^(]+ is \([-\d.]+,\s*[-\d.]+,\s*[-\d.]+\)"
    left_replacement = f"The left direction of {label} is {format_coord_tuple(new_left_dir)}"
    transformed = re.sub(left_pattern, left_replacement, transformed)

    # Pattern 4: Front direction
    front_pattern = r"The front direction of [^(]+ is \([-\d.]+,\s*[-\d.]+,\s*[-\d.]+\)"
    front_replacement = f"The front direction of {label} is {format_coord_tuple(new_front_dir)}"
    transformed = re.sub(front_pattern, front_replacement, transformed)

    # Pattern 5: Cosine similarity for left
    cos_left_pattern = r"(cosine similarity between the vector pointing to camera and the left direction is )[-\d.]+"
    cos_left_replacement = f"\\g<1>{cos_left:.2f}"
    transformed = re.sub(cos_left_pattern, cos_left_replacement, transformed)

    # Pattern 6: Angle for left
    angle_left_pattern = r"(corresponding to an angle of )[\d.]+ degrees(\. Thus the angle between the vector pointing to camera and the right direction)"
    angle_left_replacement = f"\\g<1>{angle_left:.2f} degrees\\2"
    transformed = re.sub(angle_left_pattern, angle_left_replacement, transformed)

    # Pattern 7: Right angle
    angle_right_pattern = r"(the angle between the vector pointing to camera and the right direction is )[\d.]+ degrees"
    angle_right_replacement = f"\\g<1>{angle_right:.2f} degrees"
    transformed = re.sub(angle_right_pattern, angle_right_replacement, transformed)

    # Pattern 8: Cosine similarity for front
    cos_front_pattern = r"(cosine similarity between the vector pointing to camera and the front direction is )[-\d.]+"
    cos_front_replacement = f"\\g<1>{cos_front:.2f}"
    transformed = re.sub(cos_front_pattern, cos_front_replacement, transformed)

    # Pattern 9: Angle for front
    angle_front_pattern = r"(The cosine similarity between the vector pointing to camera and the front direction is [-\d.]+, corresponding to an angle of )[\d.]+ degrees"
    angle_front_replacement = f"\\g<1>{angle_front:.2f} degrees"
    transformed = re.sub(angle_front_pattern, angle_front_replacement, transformed)

    # Pattern 10: Back angle
    angle_back_pattern = r"(the angle between the vector pointing to camera and the back direction is )[\d.]+ degrees"
    angle_back_replacement = f"\\g<1>{angle_back:.2f} degrees"
    transformed = re.sub(angle_back_pattern, angle_back_replacement, transformed)

    # Pattern 11: Smallest angle
    angles = {'left': angle_left, 'right': angle_right, 'front': angle_front, 'back': angle_back}
    min_dir = min(angles, key=angles.get)
    min_angle = angles[min_dir]

    smallest_pattern = r"(the smallest angle is )(\w+)( direction, with an angle of )[\d.]+ degrees"
    smallest_replacement = f"\\g<1>{min_dir}\\g<3>{min_angle:.2f} degrees"
    transformed = re.sub(smallest_pattern, smallest_replacement, transformed)

    return transformed


def _transform_cot_regex(cot: str, angle_deg: float) -> str:
    """Transform CoT using regex-based coordinate rotation.

    This is a fallback when original geometry data is not available.
    Rotates all (x, y, z) coordinate tuples around Y-axis.
    """
    # Regex to match coordinate tuples (x, y, z)
    coord_pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"

    def rotate_match(match):
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))

            rotated = rotate_y_axis([x, y, z], angle_deg)
            return format_coord_tuple(rotated)
        except ValueError:
            return match.group(0)

    return re.sub(coord_pattern, rotate_match, cot)


# ============================================================================
# Security Utilities
# ============================================================================

def sanitize_path_component(component: str) -> str:
    """Sanitize a path component to prevent directory traversal attacks.

    Removes or replaces dangerous characters and patterns that could be used
    for path traversal (e.g., '..', '/', '\\').

    Args:
        component: Raw path component (e.g., image ID extracted from filename)

    Returns:
        Sanitized path component safe for use in file paths

    Raises:
        ValueError: If the component is empty after sanitization
    """
    if not component:
        raise ValueError("Path component cannot be empty")

    # Remove any directory separators and parent directory references
    sanitized = component.replace('/', '').replace('\\', '')
    sanitized = sanitized.replace('..', '')

    # Remove any null bytes (potential security issue)
    sanitized = sanitized.replace('\x00', '')

    # Only allow alphanumeric characters, underscores, hyphens, and dots
    # but not leading dots (hidden files) or consecutive dots
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '', sanitized)
    sanitized = sanitized.lstrip('.')
    sanitized = re.sub(r'\.{2,}', '.', sanitized)

    if not sanitized:
        raise ValueError(f"Path component '{component}' is invalid after sanitization")

    return sanitized


# ============================================================================
# View File Mapping
# ============================================================================

def get_view_angles(num_frames: int = 10, d_phi: float = 30.0) -> List[float]:
    """Get view angles for MVGenMaster output.

    MVGenMaster generates num_frames views from 0 to d_phi degrees.

    Args:
        num_frames: Number of frames generated
        d_phi: Total azimuth rotation

    Returns:
        List of angles for each frame
    """
    if num_frames <= 1:
        return [0.0]

    step = d_phi / (num_frames - 1)
    return [i * step for i in range(num_frames)]


def map_view_files_to_angles(
    view_dir: str,
    num_frames: int = 10,
    d_phi: float = 30.0,
) -> Dict[str, str]:
    """Map view files to their corresponding angles.

    Args:
        view_dir: Directory containing view*.png files
        num_frames: Number of frames
        d_phi: Total azimuth rotation

    Returns:
        Dictionary mapping angle_str -> absolute_image_path
    """
    if not os.path.exists(view_dir):
        return {}

    angles = get_view_angles(num_frames, d_phi)
    view_images = {}

    files = sorted(os.listdir(view_dir))

    for f in files:
        if not f.endswith('.png'):
            continue

        # Parse view index
        if f == 'view000_ref.png' or f == 'view000.png':
            idx = 0
        else:
            match = re.match(r'view(\d+)\.png', f)
            if match:
                idx = int(match.group(1))
            else:
                continue

        if idx < len(angles):
            angle = angles[idx]
            angle_str = f"{angle:.1f}"
            view_images[angle_str] = os.path.join(view_dir, f)

    return view_images


# ============================================================================
# Dataset Update
# ============================================================================

def extract_geometry_from_cot(cot: str) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]], str]:
    """Extract original geometry data from CoT text.

    Args:
        cot: Chain-of-thought text

    Returns:
        Tuple of (bbox, front_dir, left_dir, label)
    """
    bbox = None
    front_dir = None
    left_dir = None
    label = "the object"

    # Extract location
    loc_match = re.search(
        r"The location of ([^(]+) is \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
        cot
    )
    if loc_match:
        label = loc_match.group(1).strip()
        bbox = [float(loc_match.group(2)), float(loc_match.group(3)), float(loc_match.group(4))]

    # Extract front direction
    front_match = re.search(
        r"The front direction of [^(]+ is \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
        cot
    )
    if front_match:
        front_dir = [float(front_match.group(1)), float(front_match.group(2)), float(front_match.group(3))]

    # Extract left direction
    left_match = re.search(
        r"The left direction of [^(]+ is \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
        cot
    )
    if left_match:
        left_dir = [float(left_match.group(1)), float(left_match.group(2)), float(left_match.group(3))]

    return bbox, front_dir, left_dir, label


def update_sample_with_views(
    sample: Dict[str, Any],
    images_base_dir: str,
    num_frames: int = 10,
    d_phi: float = 30.0,
) -> Dict[str, Any]:
    """Update a single sample with multi-view data.

    Args:
        sample: Original sample dictionary
        images_base_dir: Base directory for generated images
        num_frames: Number of frames per sample
        d_phi: Total azimuth rotation

    Returns:
        Updated sample with multi-view data
    """
    updated = deepcopy(sample)

    # Get image ID with path sanitization to prevent directory traversal
    original_image = sample.get('original_image', '')
    raw_img_id = os.path.splitext(os.path.basename(original_image))[0]

    try:
        img_id = sanitize_path_component(raw_img_id)
    except ValueError as e:
        logger.warning(f"Invalid image ID '{raw_img_id}': {e}")
        return updated

    # Check for generated views
    view_dir = os.path.join(images_base_dir, img_id, 'images')

    if not os.path.exists(view_dir):
        # No views generated yet
        return updated

    # Map view files to angles
    view_images = map_view_files_to_angles(view_dir, num_frames, d_phi)

    if len(view_images) <= 1:
        # Only original view
        return updated

    # Extract original geometry from CoT
    original_cot = sample.get('answer_cot', '')
    bbox, front_dir, left_dir, label = extract_geometry_from_cot(original_cot)

    # Transform CoT for each view angle
    augmented_cots = {}
    angles = get_view_angles(num_frames, d_phi)

    for angle in angles:
        angle_str = f"{angle:.1f}"

        if angle_str in view_images:
            transformed_cot = transform_cot_for_angle(
                original_cot,
                angle,
                bbox,
                front_dir,
                left_dir,
                label,
            )
            augmented_cots[angle_str] = transformed_cot

    # Update sample
    updated['view_images'] = view_images
    updated['augmented_cots'] = augmented_cots

    return updated


def update_dataset(
    dataset_path: str,
    images_dir: str,
    output_path: str,
    num_frames: int = 10,
    d_phi: float = 30.0,
    min_views: int = 2,
) -> Dict[str, int]:
    """Update entire dataset with multi-view data.

    Args:
        dataset_path: Path to augmented_dataset.json
        images_dir: Directory containing generated view images
        output_path: Output path for updated dataset
        num_frames: Number of frames per sample
        d_phi: Total azimuth rotation
        min_views: Minimum views required to include sample

    Returns:
        Statistics dictionary

    Raises:
        FileNotFoundError: If dataset_path does not exist
        json.JSONDecodeError: If dataset file contains invalid JSON
        PermissionError: If file cannot be read or written due to permissions
        IOError: If file write operation fails
    """
    logger.info(f"Loading dataset from {dataset_path}")

    try:
        with open(dataset_path, 'r') as f:
            samples = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset file {dataset_path}: {e}")
        raise
    except PermissionError:
        logger.error(f"Permission denied reading dataset file: {dataset_path}")
        raise

    logger.info(f"Loaded {len(samples)} samples")

    # Update samples
    updated_samples = []
    stats = {
        'total': len(samples),
        'updated': 0,
        'skipped_no_views': 0,
        'skipped_few_views': 0,
    }

    for sample in tqdm(samples, desc="Updating samples"):
        updated = update_sample_with_views(
            sample, images_dir, num_frames, d_phi
        )

        num_views = len(updated.get('view_images', {}))

        if num_views >= min_views:
            updated_samples.append(updated)
            if num_views > 1:
                stats['updated'] += 1
        elif num_views == 1:
            # Keep single-view samples but mark as not updated
            updated_samples.append(updated)
            stats['skipped_few_views'] += 1
        else:
            stats['skipped_no_views'] += 1

    # Save updated dataset
    logger.info(f"Saving {len(updated_samples)} samples to {output_path}")

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied creating output directory: {output_dir}")
        raise
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    try:
        with open(output_path, 'w') as f:
            json.dump(updated_samples, f, indent=2)
    except PermissionError:
        logger.error(f"Permission denied writing to output file: {output_path}")
        raise
    except IOError as e:
        logger.error(f"Failed to write output file {output_path}: {e}")
        raise

    logger.info(f"Statistics: {stats}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Update augmented dataset with multi-view data")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/SpatialReasoner/data/mvgen_augmented/augmented_dataset.json",
        help="Path to input augmented_dataset.json",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/data/SpatialReasoner/data/mvgen_augmented/images",
        help="Directory containing generated view images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/SpatialReasoner/data/mvgen_augmented/augmented_dataset_multiview.json",
        help="Output path for updated dataset",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames generated by MVGenMaster",
    )
    parser.add_argument(
        "--d_phi",
        type=float,
        default=30.0,
        help="Total azimuth rotation in degrees",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=1,
        help="Minimum views required to include sample (1=include all)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    stats = update_dataset(
        dataset_path=args.dataset_path,
        images_dir=args.images_dir,
        output_path=args.output_path,
        num_frames=args.num_frames,
        d_phi=args.d_phi,
        min_views=args.min_views,
    )

    print("\n=== Update Complete ===")
    print(f"Total samples: {stats['total']}")
    print(f"Updated with multi-view: {stats['updated']}")
    print(f"Single-view only: {stats['skipped_few_views']}")
    print(f"Skipped (no views): {stats['skipped_no_views']}")


if __name__ == "__main__":
    main()
