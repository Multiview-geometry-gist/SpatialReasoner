"""Orient-Anything pose estimation backend.

Wraps the Orient-Anything model for neural network-based orientation estimation.
Orient-Anything predicts Euler angles (azimuth, polar, roll) for object orientation.

This backend combines:
- Position/scale from depth (same as depth_based)
- Orientation from Orient-Anything neural network

Reference:
    https://github.com/Viglong/Orient-Anything
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from .base import (
    PoseEstimatorBase,
    ObjectPose,
    PoseEstimationConfig,
    euler_to_quaternion,
)
from .registry import PoseEstimatorRegistry

logger = logging.getLogger(__name__)

# Default paths for Orient-Anything
_ORIENT_ANYTHING_ROOT = Path("/home/ubuntu/Orient-Anything")
_DEFAULT_CHECKPOINT = "Viglong/Orient-Anything"


@PoseEstimatorRegistry.register("orient_anything", priority=50)
class OrientAnythingEstimator(PoseEstimatorBase):
    """Orient-Anything based pose estimation.

    Uses Orient-Anything neural network for accurate orientation estimation.
    Position and scale are still computed from depth map.

    Attributes:
        backend_name: "orient_anything"

    Configuration:
        - orient_anything_model_path: Path to model checkpoint or HuggingFace repo
        - orient_anything_device: Device for model inference ("cuda" or "cpu")
        - use_background_removal: Remove background before orientation estimation
        - use_inference_augmentation: Use test-time augmentation for robustness

    Note:
        Requires Orient-Anything dependencies:
            pip install rembg transformers torch
    """

    backend_name = "orient_anything"

    def __init__(self, config: PoseEstimationConfig):
        super().__init__(config)

        self._model_path = getattr(
            config, 'orient_anything_model_path', _DEFAULT_CHECKPOINT
        )
        self._device = getattr(config, 'orient_anything_device', 'cuda')
        self._use_bg_removal = getattr(config, 'use_background_removal', False)
        self._use_augmentation = getattr(config, 'use_inference_augmentation', False)

        # Lazy-loaded components
        self._model = None
        self._preprocessor = None
        self._rembg_session = None

    def _initialize(self) -> None:
        """Load Orient-Anything model and preprocessor."""
        logger.info(f"Initializing Orient-Anything on {self._device}")

        try:
            self._load_orient_anything()
            logger.info("Orient-Anything loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Orient-Anything: {e}")
            raise RuntimeError(
                f"Could not initialize Orient-Anything. "
                f"Ensure dependencies are installed and model path is valid. "
                f"Error: {e}"
            )

    def _load_orient_anything(self) -> None:
        """Load the Orient-Anything model and preprocessor."""
        import sys
        import torch
        from transformers import AutoImageProcessor

        # Add Orient-Anything to path if needed
        if str(_ORIENT_ANYTHING_ROOT) not in sys.path:
            sys.path.insert(0, str(_ORIENT_ANYTHING_ROOT))

        # Import Orient-Anything components
        from vision_tower import DINOv2_MLP
        from paths import DINO_LARGE

        # Download checkpoint if needed
        if self._model_path == _DEFAULT_CHECKPOINT:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=_DEFAULT_CHECKPOINT,
                filename="ronormsigma1/dino_weight.pt",
                repo_type="model",
                cache_dir=str(_ORIENT_ANYTHING_ROOT),
                resume_download=True
            )
        else:
            ckpt_path = self._model_path

        # Create model
        # Output dim: 360 (azimuth) + 180 (polar) + 360 (roll) + 2 (confidence)
        self._model = DINOv2_MLP(
            dino_mode='large',
            in_dim=1024,
            out_dim=360 + 180 + 360 + 2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        )

        # Load weights
        self._model.load_state_dict(
            torch.load(ckpt_path, map_location='cpu')
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        # Load preprocessor
        self._preprocessor = AutoImageProcessor.from_pretrained(
            DINO_LARGE,
            cache_dir=str(_ORIENT_ANYTHING_ROOT)
        )

        # Initialize background removal if needed
        if self._use_bg_removal:
            import rembg
            self._rembg_session = rembg.new_session()

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
        """Estimate pose using Orient-Anything for orientation.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Metric depth map (H, W)
            mask: Segmentation mask (H, W)
            box: 2D bounding box [x1, y1, x2, y2]
            K: Camera intrinsics (3, 3)
            label: Optional object label
            object_id: Unique identifier

        Returns:
            ObjectPose with neural orientation estimation
        """
        import torch
        from PIL import Image

        self._ensure_initialized()

        object_pixels = mask > 0
        min_pixels = getattr(self.config, 'min_mask_pixels', 10)

        if object_pixels.sum() < min_pixels:
            return None

        # Compute position and scale from depth (same as depth_based)
        position, scale = self._compute_position_scale(
            depth_map, mask, K
        )

        # Extract object crop for orientation estimation
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)

        # Get orientation from Orient-Anything
        try:
            azimuth, polar, roll, confidence = self._predict_orientation(crop_pil)
        except Exception as e:
            logger.warning(f"Orient-Anything prediction failed: {e}")
            return None

        # Convert Euler angles to quaternion
        quaternion = euler_to_quaternion(azimuth, polar, roll, degrees=True)

        # Build rotation matrix from quaternion
        rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)

        return ObjectPose(
            position=position,
            scale=scale,
            quaternion=quaternion,
            rotation_matrix=rotation_matrix,
            mask=mask,
            bbox_2d=box,
            confidence=float(confidence),
            label=label,
            object_id=object_id,
            euler_angles=np.array([azimuth, polar, roll]),
            metadata={
                "backend": self.backend_name,
                "azimuth": float(azimuth),
                "polar": float(polar),
                "roll": float(roll),
            }
        )

    def _compute_position_scale(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute position and scale from depth map.

        Uses the same method as depth_based backend.
        """
        H, W = depth_map.shape
        K_inv = np.linalg.inv(K)

        object_pixels = mask > 0

        # Create coordinate grids
        u_coords = np.arange(W)
        v_coords = np.arange(H)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # Get 3D points for this object
        object_depths = depth_map[object_pixels]
        object_u = u_grid[object_pixels]
        object_v = v_grid[object_pixels]

        # Back-project to 3D
        points_2d = np.stack([
            object_u,
            object_v,
            np.ones_like(object_u)
        ], axis=-1)

        directions = (K_inv @ points_2d.T).T
        points_3d = directions * object_depths[:, np.newaxis]

        # Position: centroid
        position = points_3d.mean(axis=0)

        # Scale: bounding box
        scale = points_3d.max(axis=0) - points_3d.min(axis=0)
        scale = np.maximum(scale, 0.01)

        return position, scale

    def _predict_orientation(
        self,
        image_pil,
    ) -> Tuple[float, float, float, float]:
        """Predict orientation using Orient-Anything.

        Args:
            image_pil: PIL Image of object crop

        Returns:
            Tuple of (azimuth, polar, roll, confidence)
        """
        import torch
        import torch.nn.functional as F
        import sys

        # Add Orient-Anything to path
        if str(_ORIENT_ANYTHING_ROOT) not in sys.path:
            sys.path.insert(0, str(_ORIENT_ANYTHING_ROOT))

        from utils import background_preprocess, get_3angle_infer_aug

        # Preprocess image
        if self._use_bg_removal:
            processed_img = background_preprocess(image_pil, do_remove_background=True)
        else:
            processed_img = image_pil

        if self._use_augmentation:
            # Use test-time augmentation
            rm_bkg_img = background_preprocess(image_pil, do_remove_background=True)
            angles = get_3angle_infer_aug(
                image_pil, rm_bkg_img,
                self._model, self._preprocessor, self._device
            )
        else:
            # Single forward pass
            angles = self._get_3angle(processed_img)

        azimuth = float(angles[0])
        polar = float(angles[1])
        roll = float(angles[2])
        confidence = float(angles[3])

        return azimuth, polar, roll, confidence

    def _get_3angle(self, image_pil) -> "torch.Tensor":
        """Get orientation angles from single image.

        Simplified version of Orient-Anything's get_3angle function.
        """
        import torch
        import torch.nn.functional as F

        # Preprocess
        image_inputs = self._preprocessor(images=image_pil)
        image_inputs['pixel_values'] = torch.from_numpy(
            np.array(image_inputs['pixel_values'])
        ).to(self._device)

        # Forward pass
        with torch.no_grad():
            dino_pred = self._model(image_inputs)

        # Parse predictions
        # Azimuth: 0-360 degrees
        gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
        # Polar: -90 to 90 degrees (stored as 0-180)
        gaus_pl_pred = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
        # Roll: -180 to 180 degrees (stored as 0-360)
        gaus_ro_pred = torch.argmax(dino_pred[:, 360+180:360+180+360], dim=-1)
        # Confidence: softmax of last 2 values
        confidence = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]

        angles = torch.zeros(4)
        angles[0] = gaus_ax_pred
        angles[1] = gaus_pl_pred - 90  # Convert to -90 to 90
        angles[2] = gaus_ro_pred - 180  # Convert to -180 to 180
        angles[3] = confidence

        return angles

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.

        Args:
            q: Unit quaternion [w, x, y, z]

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q

        # First row
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - w*z)
        r02 = 2*(x*z + w*y)

        # Second row
        r10 = 2*(x*y + w*z)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - w*x)

        # Third row
        r20 = 2*(x*z - w*y)
        r21 = 2*(y*z + w*x)
        r22 = 1 - 2*(x*x + y*y)

        return np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ])

    def get_capabilities(self) -> Dict[str, bool]:
        """Return backend capabilities."""
        return {
            "requires_depth": True,  # For position/scale
            "requires_mask": True,
            "estimates_orientation": True,
            "gpu_required": True,
            "provides_confidence": True,
            "orientation_source": "neural_network",
        }


def check_orient_anything_available() -> bool:
    """Check if Orient-Anything dependencies are available.

    Returns:
        True if all dependencies are installed
    """
    try:
        import torch
        import transformers
        import rembg

        # Check if Orient-Anything repo exists
        if not _ORIENT_ANYTHING_ROOT.exists():
            return False

        return True

    except ImportError:
        return False
