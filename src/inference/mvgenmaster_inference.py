"""MVGenMaster Inference Pipeline for SpatialReasoner (Approach 2).

End-to-end inference pipeline that:
1. Takes a single input image
2. Generates multiple views using MVGenMaster
3. Feeds all views to the VLM for multi-view spatial reasoning

This implements the full inference-time multi-view reasoning strategy.

Example:
    from inference.mvgenmaster_inference import MVGenMasterInferencePipeline

    pipeline = MVGenMasterInferencePipeline(
        model_path="path/to/finetuned/model",
        device="cuda:0",
    )

    # Simple inference
    result = pipeline.infer(
        image="input.jpg",
        question="What is to the left of the chair?",
    )
    print(result.answer)

    # Inference with custom view configuration
    result = pipeline.infer(
        image="input.jpg",
        question="What is the spatial relationship between objects?",
        num_views=5,
        view_angles=[-20, -10, 0, 10, 20],
    )
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results.

    Attributes:
        answer: Generated answer text
        input_image: Original input image
        generated_views: List of generated view images
        view_angles: List of view angles used
        processing_time: Total processing time in seconds
        view_generation_time: Time for view generation
        inference_time: Time for VLM inference
        metadata: Additional result metadata
    """
    answer: str
    input_image: Image.Image
    generated_views: List[Image.Image]
    view_angles: List[float]
    processing_time: float = 0.0
    view_generation_time: float = 0.0
    inference_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        """Total number of views used (including original)."""
        return len(self.generated_views)

    def save_views(self, output_dir: str) -> List[str]:
        """Save all views to disk.

        Args:
            output_dir: Directory to save views

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # Save original
        original_path = os.path.join(output_dir, "view_original.jpg")
        self.input_image.save(original_path)
        paths.append(original_path)

        # Save generated views
        for i, (view, angle) in enumerate(zip(self.generated_views, self.view_angles)):
            if angle == 0.0:
                continue  # Already saved as original
            view_path = os.path.join(output_dir, f"view_{angle:+.1f}.jpg")
            view.save(view_path)
            paths.append(view_path)

        return paths


@dataclass
class MVGenInferenceConfig:
    """Configuration for MVGenMaster inference pipeline.

    Attributes:
        model_path: Path to fine-tuned VLM model
        device: Device for inference (e.g., "cuda:0")
        mvgen_gpu: GPU ID for MVGenMaster (can be different from VLM)
        num_views: Number of views to generate
        view_angles: Specific view angles to use
        max_angle: Maximum azimuth angle
        guidance_scale: CFG scale for MVGenMaster
        elevation: Camera elevation angle
        include_view_descriptions: Whether to add view descriptions to prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        cache_views: Whether to cache generated views
        cache_dir: Directory for view cache
    """
    # Model settings
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda:0"
    mvgen_gpu: int = 0

    # View generation settings
    num_views: int = 3
    view_angles: Optional[List[float]] = None
    max_angle: float = 15.0
    guidance_scale: float = 2.0
    elevation: float = 5.0

    # Prompt settings
    include_view_descriptions: bool = True

    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = False

    # Caching
    cache_views: bool = False
    cache_dir: str = "./view_cache"

    def get_view_angles(self) -> List[float]:
        """Get view angles based on configuration."""
        if self.view_angles is not None:
            return sorted(self.view_angles)

        if self.num_views == 1:
            return [0.0]
        elif self.num_views == 2:
            return [-self.max_angle, self.max_angle]
        elif self.num_views == 3:
            return [-self.max_angle, 0.0, self.max_angle]
        else:
            half_n = (self.num_views - 1) // 2
            step = self.max_angle / half_n if half_n > 0 else self.max_angle
            angles = [i * step for i in range(-half_n, half_n + 1)]
            return sorted(set(angles))

    @classmethod
    def from_yaml(cls, path: str) -> "MVGenInferenceConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested structure
        if "inference" in config_dict:
            config_dict = config_dict["inference"]

        return cls(**config_dict)


class MVGenMasterInferencePipeline:
    """End-to-end inference pipeline with MVGenMaster view generation.

    This pipeline combines:
    1. MVGenMaster for multi-view generation
    2. Qwen2.5-VL for multi-view spatial reasoning

    Example:
        pipeline = MVGenMasterInferencePipeline(
            model_path="path/to/finetuned/model",
            device="cuda:0",
        )

        # Single inference
        result = pipeline.infer(
            image="input.jpg",
            question="What is behind the chair?",
        )

        # Batch inference
        results = pipeline.batch_infer(
            images=["img1.jpg", "img2.jpg"],
            questions=["Question 1", "Question 2"],
        )
    """

    def __init__(
        self,
        config: Optional[MVGenInferenceConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the inference pipeline.

        Args:
            config: Full configuration object
            model_path: Path to VLM model (overrides config)
            device: Device for inference (overrides config)
        """
        self.config = config or MVGenInferenceConfig()

        if model_path:
            self.config.model_path = model_path
        if device:
            self.config.device = device

        # Lazy initialization
        self._model = None
        self._processor = None
        self._mvgen_generator = None
        self._formatter = None

    @property
    def model(self):
        """Lazy load the VLM model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy load the processor."""
        if self._processor is None:
            self._load_model()
        return self._processor

    @property
    def mvgen_generator(self):
        """Lazy load the MVGenMaster generator."""
        if self._mvgen_generator is None:
            self._load_mvgen()
        return self._mvgen_generator

    @property
    def formatter(self):
        """Get or create the prompt formatter."""
        if self._formatter is None:
            from data_generation.multiview_dataset import MultiViewPromptFormatter
            self._formatter = MultiViewPromptFormatter(
                include_view_descriptions=self.config.include_view_descriptions,
            )
        return self._formatter

    def _load_model(self):
        """Load the VLM model and processor."""
        logger.info(f"Loading model from {self.config.model_path}...")

        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self._model.eval()

        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")

    def _load_mvgen(self):
        """Load the MVGenMaster generator."""
        logger.info("Loading MVGenMaster generator...")

        from data_generation.mvgenmaster_integration import (
            MVGenMasterGenerator,
            MVGenConfig,
        )

        mvgen_config = MVGenConfig(
            num_views=self.config.num_views,
            azimuth_range=self.config.max_angle * 1.2,
            elevation=self.config.elevation,
            guidance_scale=self.config.guidance_scale,
            gpu_id=self.config.mvgen_gpu,
        )

        self._mvgen_generator = MVGenMasterGenerator(mvgen_config)
        logger.info("MVGenMaster generator loaded")

    def generate_views(
        self,
        image: Union[str, Image.Image],
        view_angles: Optional[List[float]] = None,
    ) -> Tuple[List[Image.Image], List[float]]:
        """Generate views for an input image.

        Args:
            image: Input image path or PIL Image
            view_angles: Specific angles to generate

        Returns:
            Tuple of (list of view images, list of angles)
        """
        import time
        start_time = time.time()

        # Handle input
        if isinstance(image, str):
            image_path = image
            input_image = Image.open(image).convert("RGB")
        else:
            input_image = image
            # Need to save temporarily for MVGenMaster
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                input_image.save(f.name)
                image_path = f.name

        # Get angles
        angles = view_angles or self.config.get_view_angles()

        # Check cache
        if self.config.cache_views:
            cached = self._load_from_cache(image_path, angles)
            if cached is not None:
                return cached

        # Generate views
        try:
            views = self.mvgen_generator.generate_views_at_angles(
                image_path,
                angles=angles,
            )
        finally:
            # Clean up temp file if we created one
            if not isinstance(image, str) and os.path.exists(image_path):
                os.remove(image_path)

        # Collect results
        result_images = []
        result_angles = []

        for angle in sorted(views.keys()):
            view = views[angle]
            result_images.append(view.image)
            result_angles.append(angle)

        # Cache if enabled
        if self.config.cache_views:
            self._save_to_cache(image_path, result_images, result_angles)

        generation_time = time.time() - start_time
        logger.debug(f"View generation took {generation_time:.2f}s")

        return result_images, result_angles

    def infer(
        self,
        image: Union[str, Image.Image],
        question: str,
        num_views: Optional[int] = None,
        view_angles: Optional[List[float]] = None,
    ) -> InferenceResult:
        """Run inference on a single image.

        Args:
            image: Input image path or PIL Image
            question: Question to answer
            num_views: Override number of views
            view_angles: Override view angles

        Returns:
            InferenceResult with answer and metadata
        """
        import time
        total_start = time.time()

        # Load input image
        if isinstance(image, str):
            input_image = Image.open(image).convert("RGB")
        else:
            input_image = image.convert("RGB")

        # Generate views
        view_start = time.time()
        angles = view_angles or self.config.get_view_angles()
        if num_views and num_views != len(angles):
            # Recalculate angles
            half_n = (num_views - 1) // 2
            step = self.config.max_angle / half_n if half_n > 0 else self.config.max_angle
            angles = [i * step for i in range(-half_n, half_n + 1)]

        view_images, actual_angles = self.generate_views(image, angles)
        view_time = time.time() - view_start

        # Prepare multi-view input
        inference_start = time.time()

        # Format prompt
        messages = self.formatter.format_for_processor(
            images=view_images,
            question=question,
            view_angles=actual_angles,
        )

        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=view_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
            )

        # Decode output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        inference_time = time.time() - inference_start
        total_time = time.time() - total_start

        return InferenceResult(
            answer=answer.strip(),
            input_image=input_image,
            generated_views=view_images,
            view_angles=actual_angles,
            processing_time=total_time,
            view_generation_time=view_time,
            inference_time=inference_time,
            metadata={
                "model_path": self.config.model_path,
                "question": question,
            },
        )

    def batch_infer(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str],
        num_views: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[InferenceResult]:
        """Run inference on multiple images.

        Args:
            images: List of image paths or PIL Images
            questions: List of questions
            num_views: Override number of views
            show_progress: Whether to show progress bar

        Returns:
            List of InferenceResult objects
        """
        assert len(images) == len(questions), "Images and questions must have same length"

        results = []

        iterator = zip(images, questions)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Inference")

        for image, question in iterator:
            try:
                result = self.infer(
                    image=image,
                    question=question,
                    num_views=num_views,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                # Create empty result for failed inference
                input_img = Image.open(image).convert("RGB") if isinstance(image, str) else image
                results.append(InferenceResult(
                    answer=f"[Error: {str(e)}]",
                    input_image=input_img,
                    generated_views=[],
                    view_angles=[],
                    metadata={"error": str(e)},
                ))

        return results

    def _load_from_cache(
        self,
        image_path: str,
        angles: List[float],
    ) -> Optional[Tuple[List[Image.Image], List[float]]]:
        """Load cached views if available."""
        if not self.config.cache_views:
            return None

        cache_key = self._get_cache_key(image_path, angles)
        cache_path = os.path.join(self.config.cache_dir, cache_key)

        if not os.path.exists(cache_path):
            return None

        try:
            # Load cached images
            images = []
            loaded_angles = []

            for angle in angles:
                view_path = os.path.join(cache_path, f"view_{angle:+.1f}.jpg")
                if os.path.exists(view_path):
                    images.append(Image.open(view_path).convert("RGB"))
                    loaded_angles.append(angle)

            if len(images) == len(angles):
                logger.debug(f"Loaded views from cache: {cache_path}")
                return images, loaded_angles

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")

        return None

    def _save_to_cache(
        self,
        image_path: str,
        images: List[Image.Image],
        angles: List[float],
    ):
        """Save views to cache."""
        if not self.config.cache_views:
            return

        cache_key = self._get_cache_key(image_path, angles)
        cache_path = os.path.join(self.config.cache_dir, cache_key)

        try:
            os.makedirs(cache_path, exist_ok=True)

            for img, angle in zip(images, angles):
                view_path = os.path.join(cache_path, f"view_{angle:+.1f}.jpg")
                img.save(view_path, quality=95)

            logger.debug(f"Saved views to cache: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _get_cache_key(self, image_path: str, angles: List[float]) -> str:
        """Generate cache key for an image and angles."""
        import hashlib

        # Hash based on image path and angles
        content = f"{image_path}_{sorted(angles)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class SingleViewFallback:
    """Fallback inference without view generation.

    Used when MVGenMaster is not available or view generation fails.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ):
        """Initialize fallback inference.

        Args:
            model_path: Path to VLM model
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def infer(
        self,
        image: Union[str, Image.Image],
        question: str,
    ) -> str:
        """Run single-view inference.

        Args:
            image: Input image
            question: Question to answer

        Returns:
            Generated answer
        """
        if self._model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            self._model.eval()
            self._processor = Qwen2_5_VLProcessor.from_pretrained(self.model_path)

        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Format input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return answer.strip()


def create_inference_pipeline(
    model_path: str,
    device: str = "cuda:0",
    use_multiview: bool = True,
    **kwargs,
) -> Union[MVGenMasterInferencePipeline, SingleViewFallback]:
    """Factory function to create appropriate inference pipeline.

    Args:
        model_path: Path to VLM model
        device: Device for inference
        use_multiview: Whether to use multi-view inference
        **kwargs: Additional configuration options

    Returns:
        Inference pipeline instance
    """
    if use_multiview:
        from data_generation.mvgenmaster_integration import check_mvgenmaster_available

        if check_mvgenmaster_available():
            config = MVGenInferenceConfig(
                model_path=model_path,
                device=device,
                **kwargs,
            )
            return MVGenMasterInferencePipeline(config)
        else:
            logger.warning(
                "MVGenMaster not available, falling back to single-view inference"
            )

    return SingleViewFallback(model_path, device)
