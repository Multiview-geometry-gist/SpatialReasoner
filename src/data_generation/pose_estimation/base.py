"""Abstract base class for pose estimation backends.

Defines the interface that all pose estimation implementations must follow.
This enables plug-and-play backend switching without modifying pipeline code.

Pose estimation converts 2D detections + depth into 6-DOF poses:
    - Position (3D): Object centroid in camera coordinates
    - Scale (3D): Bounding box dimensions [width, height, depth]
    - Orientation (quaternion): Unit quaternion [w, x, y, z]

Architecture:
    PoseEstimatorBase (ABC)
        |
        +-- DepthBasedEstimator (PCA-based, requires depth)
        +-- OrientAnythingEstimator (neural network, image-based)
        +-- HybridEstimator (combines both approaches)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Import quaternion utilities from spatial_reasoner
import sys
from pathlib import Path
_SPATIAL_REASONER_PATH = Path(__file__).parent.parent.parent
if str(_SPATIAL_REASONER_PATH) not in sys.path:
    sys.path.insert(0, str(_SPATIAL_REASONER_PATH))

from spatial_reasoner.utils import quaternion as quat


@dataclass
class ObjectPose:
    """6-DOF pose representation for a single object.

    Implements the 10D object representation: s_i = [t_i, d_i, q_i]
    where:
        - t_i: 3D position (translation)
        - d_i: 3D scale (bounding box dimensions)
        - q_i: 4D quaternion (orientation)

    Attributes:
        position: 3D position (t_i) in camera coordinates
        scale: 3D bounding box dimensions (d_i) [width, height, depth]
        quaternion: Unit quaternion (q_i) [w, x, y, z]
        rotation_matrix: 3x3 rotation matrix (derived from quaternion)
        mask: Binary segmentation mask
        bbox_2d: 2D bounding box [x1, y1, x2, y2]
        confidence: Pose estimation confidence (0.0 to 1.0)
        label: Optional object label/class name
        object_id: Unique identifier within the image
        euler_angles: Optional Euler angles [azimuth, polar, roll] in degrees
        metadata: Additional backend-specific metadata
    """
    position: np.ndarray
    scale: np.ndarray
    quaternion: np.ndarray
    rotation_matrix: np.ndarray
    mask: np.ndarray
    bbox_2d: np.ndarray
    confidence: float = 1.0
    label: Optional[str] = None
    object_id: int = 0
    euler_angles: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_10d_vector(self) -> np.ndarray:
        """Convert to 10D representation: s_i = [t_i, d_i, q_i]."""
        return np.concatenate([
            self.position,   # 3D
            self.scale,      # 3D
            self.quaternion  # 4D
        ])

    @property
    def forward_direction(self) -> np.ndarray:
        """Compute forward direction vector from quaternion."""
        return quat.forward_direction(self.quaternion)

    @property
    def left_direction(self) -> np.ndarray:
        """Compute left direction vector from quaternion."""
        return quat.left_direction(self.quaternion)

    @property
    def up_direction(self) -> np.ndarray:
        """Compute up direction vector from quaternion."""
        return quat.up_direction(self.quaternion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "object_id": self.object_id,
            "label": self.label,
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "quaternion": self.quaternion.tolist(),
            "bbox_2d": self.bbox_2d.tolist(),
            "confidence": float(self.confidence),
        }
        if self.euler_angles is not None:
            result["euler_angles"] = self.euler_angles.tolist()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], mask: Optional[np.ndarray] = None) -> "ObjectPose":
        """Create ObjectPose from dictionary.

        Args:
            data: Dictionary with pose data
            mask: Optional segmentation mask (not serialized)

        Returns:
            ObjectPose instance
        """
        quaternion = np.array(data["quaternion"])
        return cls(
            position=np.array(data["position"]),
            scale=np.array(data["scale"]),
            quaternion=quaternion,
            rotation_matrix=quat.to_rotation_matrix(quaternion),
            mask=mask if mask is not None else np.zeros((1, 1), dtype=bool),
            bbox_2d=np.array(data["bbox_2d"]),
            confidence=data.get("confidence", 1.0),
            label=data.get("label"),
            object_id=data.get("object_id", 0),
            euler_angles=np.array(data["euler_angles"]) if "euler_angles" in data else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PoseEstimationConfig:
    """Configuration for pose estimation backends.

    This configuration supports multiple backends with graceful fallback.

    Backends:
        - depth_based: Uses depth map + PCA for orientation (default)
        - orient_anything: Uses Orient-Anything neural network
        - hybrid: Combines depth-based position with neural orientation

    Attributes:
        backend: Backend selection ("depth_based", "orient_anything", "hybrid")
        method: Legacy compatibility - maps to backend
        use_pca_orientation: For depth_based, use PCA for orientation
        orient_anything_model_path: Path to Orient-Anything checkpoint
        orient_anything_device: Device for Orient-Anything model
        min_mask_pixels: Minimum pixels for valid object detection
        confidence_threshold: Minimum confidence for valid pose
        fallback_to_depth: If orient_anything fails, fall back to depth_based
    """
    backend: str = "depth_based"
    method: str = "depth_based"  # Legacy compatibility
    use_pca_orientation: bool = True

    # Orient-Anything settings
    orient_anything_model_path: Optional[str] = None
    orient_anything_device: str = "cuda"
    use_background_removal: bool = False
    use_inference_augmentation: bool = False

    # Quality settings
    min_mask_pixels: int = 10
    confidence_threshold: float = 0.0
    fallback_to_depth: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for compatibility."""
        return getattr(self, key, default)


class PoseEstimatorBase(ABC):
    """Abstract base class for all pose estimation backends.

    All pose estimation implementations must inherit from this class and
    implement the required abstract methods. This ensures consistent
    interface across different backends.

    Example:
        @PoseEstimatorRegistry.register("my_backend")
        class MyPoseEstimator(PoseEstimatorBase):
            backend_name = "my_backend"

            def estimate_single(self, image, depth_map, mask, box, K):
                # Custom implementation
                ...
    """

    # Class attribute - subclasses must override
    backend_name: str = "base"

    def __init__(self, config: PoseEstimationConfig):
        """Initialize the estimator with configuration.

        Args:
            config: PoseEstimationConfig or compatible configuration object
        """
        self.config = config
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization hook - called before first estimation.

        Subclasses can override to load models on demand.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Perform actual initialization. Override in subclasses."""
        pass

    def estimate_poses(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        K: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> List[ObjectPose]:
        """Estimate poses for all detected objects.

        This is the main entry point for pose estimation. It handles:
        - Input validation
        - Iterating over detected objects
        - Calling estimate_single for each object
        - Filtering by confidence threshold

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            masks: Segmentation masks (N, H, W)
            boxes: 2D bounding boxes (N, 4) as [x1, y1, x2, y2]
            K: Camera intrinsics (3, 3)
            labels: Optional object labels (N,)

        Returns:
            List of ObjectPose for each detected object
        """
        self._ensure_initialized()

        poses = []
        min_pixels = getattr(self.config, 'min_mask_pixels', 10)
        confidence_threshold = getattr(self.config, 'confidence_threshold', 0.0)

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Skip if mask too small
            if mask.sum() < min_pixels:
                continue

            label = labels[i] if labels else None

            try:
                pose = self.estimate_single(
                    image=image,
                    depth_map=depth_map,
                    mask=mask,
                    box=box,
                    K=K,
                    label=label,
                    object_id=i,
                )

                # Filter by confidence
                if pose is not None and pose.confidence >= confidence_threshold:
                    poses.append(pose)

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to estimate pose for object {i}: {e}"
                )
                continue

        return poses

    @abstractmethod
    def estimate_single(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray,
        K: np.ndarray,
        label: Optional[str] = None,
        object_id: int = 0,
    ) -> Optional[ObjectPose]:
        """Estimate pose for a single object.

        This is the core method that subclasses must implement.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            mask: Segmentation mask (H, W), binary
            box: 2D bounding box [x1, y1, x2, y2]
            K: Camera intrinsics (3, 3)
            label: Optional object label
            object_id: Unique identifier for this object

        Returns:
            ObjectPose or None if estimation failed
        """
        raise NotImplementedError

    def get_capabilities(self) -> Dict[str, bool]:
        """Return backend capabilities for feature discovery.

        Returns:
            Dictionary of capability flags
        """
        return {
            "requires_depth": True,
            "requires_mask": True,
            "estimates_orientation": True,
            "gpu_required": False,
            "provides_confidence": False,
        }

    @staticmethod
    def compute_relative_pose(pose1: ObjectPose, pose2: ObjectPose) -> Dict[str, Any]:
        """Compute relative pose between two objects.

        Args:
            pose1: Reference object pose
            pose2: Target object pose

        Returns:
            Dictionary with relative pose information
        """
        # Relative position
        rel_position = pose2.position - pose1.position
        distance = np.linalg.norm(rel_position)

        # Direction from pose1 to pose2
        if distance > 1e-8:
            direction = rel_position / distance
        else:
            direction = np.array([0, 0, 1])

        # Relative rotation (q_rel = q2 * q1^-1)
        q1_inv = quat.inverse(pose1.quaternion)
        rel_quaternion = quat.multiply(pose2.quaternion, q1_inv)

        # Cosine similarity between forward directions
        fwd1 = pose1.forward_direction
        fwd2 = pose2.forward_direction
        fwd_similarity = np.dot(fwd1, fwd2)

        return {
            "relative_position": rel_position.tolist(),
            "distance": float(distance),
            "direction": direction.tolist(),
            "relative_quaternion": rel_quaternion.tolist(),
            "forward_similarity": float(fwd_similarity),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend_name})"


def euler_to_quaternion(
    azimuth: float,
    polar: float,
    roll: float,
    degrees: bool = True
) -> np.ndarray:
    """Convert Euler angles to quaternion.

    Uses ZYX convention (yaw-pitch-roll) which matches Orient-Anything output:
        - Azimuth: rotation around Z axis (yaw, 0-360 degrees)
        - Polar: rotation around Y axis (pitch, -90 to 90 degrees)
        - Roll: rotation around X axis (roll, -180 to 180 degrees)

    Args:
        azimuth: Azimuth angle (Z-axis rotation)
        polar: Polar/elevation angle (Y-axis rotation)
        roll: Roll angle (X-axis rotation)
        degrees: If True, angles are in degrees; otherwise radians

    Returns:
        Unit quaternion [w, x, y, z]
    """
    if degrees:
        azimuth = np.radians(azimuth)
        polar = np.radians(polar)
        roll = np.radians(roll)

    # Half angles
    cy = np.cos(azimuth * 0.5)
    sy = np.sin(azimuth * 0.5)
    cp = np.cos(polar * 0.5)
    sp = np.sin(polar * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # ZYX convention quaternion
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler(quaternion: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles.

    Uses ZYX convention to match Orient-Anything format.

    Args:
        quaternion: Unit quaternion [w, x, y, z]
        degrees: If True, return angles in degrees

    Returns:
        Tuple of (azimuth, polar, roll)
    """
    w, x, y, z = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Polar (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        polar = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        polar = np.arcsin(sinp)

    # Azimuth (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    azimuth = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        azimuth = np.degrees(azimuth)
        polar = np.degrees(polar)
        roll = np.degrees(roll)

        # Normalize azimuth to 0-360
        if azimuth < 0:
            azimuth += 360

    return azimuth, polar, roll
