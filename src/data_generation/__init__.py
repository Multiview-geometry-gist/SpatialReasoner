"""Data generation pipeline for multi-view spatial reasoning.

This module provides tools for:
- Depth estimation using Depth Anything v2
- Instance segmentation using SAM2
- Novel view synthesis via depth-based warping
- 6-DOF pose estimation with quaternion representation
- Spatial reasoning QA pair generation
- MVGenMaster integration for multi-view generation
- View augmentation for training data expansion
- Bulk MVGen augmentation (50-100 views per image)
- CoT (Chain of Thought) augmentation with view awareness

Usage:
    from data_generation import DataGenerationPipeline

    pipeline = DataGenerationPipeline(config)
    pipeline.process_dataset(input_dir, output_dir)

    # For view augmentation (Approach 1)
    from data_generation.multiview_augmentation import ViewAugmentedDataGenerator

    # For multi-view dataset (Approach 2)
    from data_generation.multiview_dataset import MultiViewDatasetBuilder

    # For bulk MVGen augmentation (50-100 views)
    from data_generation.mvgen_bulk_augmentation import MVGenBulkAugmentor

    # For CoT view-aware augmentation
    from data_generation.cot_augmentation import CoTAugmentor
"""

from .config import DataGenerationConfig
from .pipeline import DataGenerationPipeline

# MVGenMaster integration
from .mvgenmaster_integration import (
    MVGenMasterGenerator,
    MVGenConfig,
    GeneratedView,
    check_mvgenmaster_available,
)

# View augmentation (Approach 1)
from .multiview_augmentation import (
    ViewAugmentedDataGenerator,
    AugmentationConfig,
    AugmentedSample,
    augment_training_data,
)

# Multi-view dataset (Approach 2)
from .multiview_dataset import (
    MultiViewDatasetBuilder,
    MultiViewConfig,
    MultiViewSample,
    MultiViewPromptFormatter,
    load_multiview_dataset,
    create_multiview_collate_fn,
)

# Bulk MVGen augmentation (50-100 views)
from .mvgen_bulk_augmentation import (
    MVGenBulkAugmentor,
    MVGenBulkConfig,
    MVGenMasterBulkRunner,
)

# CoT augmentation
from .cot_augmentation import (
    CoTAugmentor,
    CoTAugmentationConfig,
    ViewDescriptor,
    StrategyRegistry,
)

__all__ = [
    # Core pipeline
    "DataGenerationConfig",
    "DataGenerationPipeline",
    # MVGenMaster integration
    "MVGenMasterGenerator",
    "MVGenConfig",
    "GeneratedView",
    "check_mvgenmaster_available",
    # View augmentation (Approach 1)
    "ViewAugmentedDataGenerator",
    "AugmentationConfig",
    "AugmentedSample",
    "augment_training_data",
    # Multi-view dataset (Approach 2)
    "MultiViewDatasetBuilder",
    "MultiViewConfig",
    "MultiViewSample",
    "MultiViewPromptFormatter",
    "load_multiview_dataset",
    "create_multiview_collate_fn",
    # Bulk MVGen augmentation
    "MVGenBulkAugmentor",
    "MVGenBulkConfig",
    "MVGenMasterBulkRunner",
    # CoT augmentation
    "CoTAugmentor",
    "CoTAugmentationConfig",
    "ViewDescriptor",
    "StrategyRegistry",
]
