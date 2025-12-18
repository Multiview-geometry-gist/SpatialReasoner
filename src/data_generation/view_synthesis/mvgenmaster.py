"""MVGenMaster backend for novel view synthesis.

Wraps MVGenMaster for high-quality diffusion-based multi-view generation.
MVGenMaster generates consistent novel views using a trained diffusion model.

Note:
    MVGenMaster is a complex system with many dependencies. This wrapper
    provides subprocess-based isolation to avoid dependency conflicts.

Reference:
    https://github.com/ewrfcas/MVGenMaster
"""

import numpy as np
import cv2
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .base import ViewSynthesizerBase, SynthesizedView, SynthesisConfig
from .registry import ViewSynthesizerRegistry

logger = logging.getLogger(__name__)

# Default paths
_MVGENMASTER_ROOT = Path("/home/ubuntu/MVGenMaster")
_DEFAULT_MODEL_DIR = _MVGENMASTER_ROOT / "check_points" / "pretrained_model"


@ViewSynthesizerRegistry.register("mvgenmaster", priority=30)
class MVGenMasterSynthesizer(ViewSynthesizerBase):
    """MVGenMaster-based novel view synthesis.

    Uses MVGenMaster diffusion model for high-quality multi-view generation.
    Supports both single-view (with camera trajectory) and multi-view inputs.

    Attributes:
        backend_name: "mvgenmaster"

    Configuration:
        - mvgenmaster_root: Path to MVGenMaster installation
        - mvgenmaster_model_dir: Path to pretrained model checkpoints
        - num_frames: Total frames to generate (default 28)
        - guidance_scale: CFG scale for diffusion (default 2.0)
        - elevation: Initial elevation angle for single-view (default 5.0)
        - d_phi: Azimuth rotation range (default 45.0)
        - camera_longest_side: Camera normalization scale (default 5.0)
        - use_subprocess: Run in subprocess for isolation (default True)

    Requirements:
        - MVGenMaster repository cloned to mvgenmaster_root
        - Pre-trained model checkpoints downloaded
        - Dependencies: torch, diffusers, dust3r, depth_pro, etc.
    """

    backend_name = "mvgenmaster"

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)

        self._root = Path(getattr(config, 'mvgenmaster_root', _MVGENMASTER_ROOT))
        self._model_dir = Path(getattr(config, 'mvgenmaster_model_dir', _DEFAULT_MODEL_DIR))
        self._num_frames = getattr(config, 'mvgenmaster_num_frames', 28)
        self._guidance_scale = getattr(config, 'mvgenmaster_guidance_scale', 2.0)
        self._elevation = getattr(config, 'mvgenmaster_elevation', 5.0)
        self._d_phi = getattr(config, 'mvgenmaster_d_phi', 45.0)
        self._camera_longest_side = getattr(config, 'camera_longest_side', 5.0)
        self._use_subprocess = getattr(config, 'mvgenmaster_use_subprocess', True)

        # Model components (loaded if not using subprocess)
        self._pipeline = None
        self._depth_model = None

    def _check_available(self) -> bool:
        """Check if MVGenMaster is available."""
        if not self._root.exists():
            logger.warning(f"MVGenMaster root not found: {self._root}")
            return False

        run_script = self._root / "run_mvgen.py"
        if not run_script.exists():
            logger.warning(f"MVGenMaster run script not found: {run_script}")
            return False

        if not self._model_dir.exists():
            logger.warning(f"MVGenMaster model not found: {self._model_dir}")
            return False

        return True

    def _initialize(self) -> None:
        """Initialize MVGenMaster components."""
        if not self._check_available():
            raise RuntimeError(
                f"MVGenMaster not available. Ensure it is installed at {self._root} "
                f"with pretrained model at {self._model_dir}"
            )

        if not self._use_subprocess:
            # Load model directly (requires all dependencies)
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load MVGenMaster pipeline directly."""
        import sys
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        try:
            import torch
            from diffusers import AutoencoderKL
            from omegaconf import OmegaConf
            from easydict import EasyDict

            # Add MVGenMaster's custom diffusers
            sys.path.insert(0, str(self._root / "my_diffusers"))
            from models import UNet2DConditionModel
            from pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import (
                StableDiffusionMultiViewPipeline
            )
            sys.path.insert(0, str(self._root / "src" / "modules"))
            from schedulers import get_diffusion_scheduler

            # Load config
            config_path = self._model_dir / "config.yaml"
            config = EasyDict(OmegaConf.load(str(config_path)))

            device = "cuda"
            weight_dtype = torch.float16

            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder="vae",
                local_files_only=True
            )
            vae.requires_grad_(False)
            vae.to(device, dtype=weight_dtype)

            # Load UNet
            unet = UNet2DConditionModel.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder="unet",
                rank=0,
                model_cfg=config.model_cfg,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
            weights = torch.load(self._model_dir / "ema_unet.pt", map_location="cpu")
            unet.load_state_dict(weights)
            unet.requires_grad_(False)
            unet.eval()
            unet.to(device, dtype=weight_dtype)

            # Build pipeline
            scheduler = get_diffusion_scheduler(config, name="DDIM")
            self._pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                vae=vae,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                torch_dtype=weight_dtype,
                local_files_only=True
            ).to(device)

            self._config = config

            logger.info("MVGenMaster pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MVGenMaster pipeline: {e}")
            raise

    def synthesize(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> Dict[float, SynthesizedView]:
        """Synthesize novel views using MVGenMaster.

        For MVGenMaster, we generate a sequence of views along a camera trajectory.
        The rotation_angles from config are used to select which angles to return.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Optional depth map (not used directly, MVGenMaster has its own)
            K: Optional camera intrinsics (estimated if not provided)

        Returns:
            Dictionary mapping angle (float) -> SynthesizedView
        """
        self._ensure_initialized()

        H, W = image.shape[:2]
        rotation_angles = self._get_rotation_angles()

        if self._use_subprocess:
            views = self._synthesize_subprocess(image, rotation_angles)
        else:
            views = self._synthesize_direct(image, rotation_angles)

        # Ensure original view is included
        if 0.0 not in views:
            views[0.0] = SynthesizedView(
                image=image.copy(),
                angle=0.0,
                mask=np.ones((H, W), dtype=bool),
                hole_ratio=0.0,
                backend=self.backend_name,
                quality_score=1.0,
            )

        return views

    def _synthesize_subprocess(
        self,
        image: np.ndarray,
        rotation_angles: List[float],
    ) -> Dict[float, SynthesizedView]:
        """Run MVGenMaster in subprocess for isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save input image
            input_path = tmpdir / "input.png"
            cv2.imwrite(str(input_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            output_path = tmpdir / "output"
            output_path.mkdir()

            # Build command
            cmd = [
                "python", str(self._root / "run_mvgen.py"),
                "--input_path", str(input_path),
                "--model_dir", str(self._model_dir),
                "--output_path", str(output_path),
                "--nframe", str(self._num_frames),
                "--val_cfg", str(self._guidance_scale),
                "--elevation", str(self._elevation),
                "--d_phi", str(max(abs(a) for a in rotation_angles if a != 0) or self._d_phi),
                "--camera_longest_side", str(self._camera_longest_side),
                "--cam_traj", "free",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=str(self._root),
                )

                if result.returncode != 0:
                    logger.error(f"MVGenMaster failed: {result.stderr}")
                    return {}

            except subprocess.TimeoutExpired:
                logger.error("MVGenMaster timed out")
                return {}
            except Exception as e:
                logger.error(f"MVGenMaster subprocess error: {e}")
                return {}

            # Load generated views
            return self._load_generated_views(output_path, rotation_angles, image.shape[:2])

    def _synthesize_direct(
        self,
        image: np.ndarray,
        rotation_angles: List[float],
    ) -> Dict[float, SynthesizedView]:
        """Run MVGenMaster directly (requires all dependencies loaded)."""
        # This would require significant code from run_mvgen.py
        # For now, fall back to subprocess
        logger.info("Direct synthesis not implemented, using subprocess")
        return self._synthesize_subprocess(image, rotation_angles)

    def _load_generated_views(
        self,
        output_path: Path,
        rotation_angles: List[float],
        original_shape: tuple,
    ) -> Dict[float, SynthesizedView]:
        """Load generated views from MVGenMaster output."""
        views = {}
        H, W = original_shape

        images_dir = output_path / "images"
        if not images_dir.exists():
            logger.warning(f"No images found in {images_dir}")
            return views

        # Get all generated view files
        view_files = sorted(images_dir.glob("view*.png"))
        if not view_files:
            logger.warning("No view files generated")
            return views

        # Map frames to angles
        # MVGenMaster generates frames along trajectory, we map to requested angles
        num_frames = len(view_files)
        max_angle = max(abs(a) for a in rotation_angles if a != 0) if any(a != 0 for a in rotation_angles) else self._d_phi

        for angle in rotation_angles:
            if angle == 0:
                # Original view is typically view000_ref or first frame
                ref_files = list(images_dir.glob("*_ref.png"))
                if ref_files:
                    view_path = ref_files[0]
                else:
                    view_path = view_files[0]
            else:
                # Map angle to frame index
                # Assuming linear trajectory from 0 to max_angle
                if max_angle != 0:
                    frame_idx = int((abs(angle) / max_angle) * (num_frames - 1))
                    frame_idx = min(frame_idx, num_frames - 1)
                else:
                    frame_idx = 0

                view_path = view_files[frame_idx]

            if view_path.exists():
                img = cv2.imread(str(view_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize if needed
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H))

                    views[angle] = SynthesizedView(
                        image=img,
                        angle=angle,
                        mask=np.ones((H, W), dtype=bool),
                        hole_ratio=0.0,  # Diffusion-based, no holes
                        backend=self.backend_name,
                        quality_score=0.9,  # High quality from diffusion
                        metadata={
                            "source_file": view_path.name,
                            "num_frames": num_frames,
                        }
                    )

        return views

    def synthesize_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        angle: float,
    ) -> SynthesizedView:
        """Synthesize a single novel view.

        MVGenMaster generates all views at once, so this method generates
        the full set and returns the requested angle.

        Args:
            image: Original RGB image (H, W, 3)
            depth_map: Depth map (not used directly by MVGenMaster)
            K: Camera intrinsic matrix (3, 3)
            angle: Rotation angle in degrees

        Returns:
            SynthesizedView at the specified angle
        """
        # Generate all views (MVGenMaster is most efficient this way)
        views = self.synthesize(image, depth_map, K)

        if angle in views:
            return views[angle]

        # Find nearest angle
        nearest_angle = min(views.keys(), key=lambda a: abs(a - angle))
        view = views[nearest_angle]

        # Update angle metadata
        return SynthesizedView(
            image=view.image,
            angle=angle,
            mask=view.mask,
            hole_ratio=view.hole_ratio,
            backend=view.backend,
            quality_score=view.quality_score * 0.9,  # Slightly lower for interpolation
            metadata={**view.metadata, "nearest_angle": nearest_angle}
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities."""
        return {
            "requires_depth": False,  # Has its own depth estimation
            "can_hallucinate": True,  # Diffusion-based
            "supports_arbitrary_angles": False,  # Limited to trajectory
            "gpu_required": True,
            "generates_video": True,
            "max_frames": self._num_frames,
        }


def check_mvgenmaster_available() -> bool:
    """Check if MVGenMaster is properly installed.

    Returns:
        True if MVGenMaster and its dependencies are available
    """
    if not _MVGENMASTER_ROOT.exists():
        return False

    if not (_MVGENMASTER_ROOT / "run_mvgen.py").exists():
        return False

    if not _DEFAULT_MODEL_DIR.exists():
        return False

    return True
