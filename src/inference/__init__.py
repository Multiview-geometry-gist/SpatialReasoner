"""Inference modules for SpatialReasoner.

This module provides inference pipelines for:
- Multi-view spatial reasoning with MVGenMaster view generation
- Single-view inference fallback

Usage:
    # Multi-view inference with MVGenMaster (Approach 2)
    from inference.mvgenmaster_inference import MVGenMasterInferencePipeline

    pipeline = MVGenMasterInferencePipeline(model_path="path/to/model")
    result = pipeline.infer(image="input.jpg", question="What is left of X?")

    # Standard multi-view spatial reasoner
    from inference.multiview_spatial_reasoner import MultiViewSpatialReasoner

    reasoner = MultiViewSpatialReasoner(device="cuda:0")
    multiview = reasoner.generate_multiview(image)
"""

from .multiview_spatial_reasoner import (
    MultiViewSpatialReasoner,
    MultiViewInput,
    generate_multiview_for_reasoning,
)

from .mvgenmaster_inference import (
    MVGenMasterInferencePipeline,
    MVGenInferenceConfig,
    InferenceResult,
    SingleViewFallback,
    create_inference_pipeline,
)

__all__ = [
    # Multi-view spatial reasoner
    "MultiViewSpatialReasoner",
    "MultiViewInput",
    "generate_multiview_for_reasoning",
    # MVGenMaster inference pipeline
    "MVGenMasterInferencePipeline",
    "MVGenInferenceConfig",
    "InferenceResult",
    "SingleViewFallback",
    "create_inference_pipeline",
]
