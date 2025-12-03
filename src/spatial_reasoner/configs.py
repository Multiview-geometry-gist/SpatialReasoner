# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, List

import trl


@dataclass
class MultiViewConfig:
    """Configuration for multi-view training."""
    enabled: bool = False
    num_views: int = 3  # [-5, 0, +5 degrees]
    view_selection: str = "random"  # "random", "all", "original_only"


@dataclass
class QuaternionConfig:
    """Configuration for quaternion-based training."""
    enabled: bool = False
    geodesic_loss_weight: float = 0.1


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_dir: str = field(
        default="data/",
        metadata={"help": "Directory containing the dataset"},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    stop_steps: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of steps for the early stop."},
    )
    # Quaternion settings
    quaternion_enabled: bool = field(
        default=False,
        metadata={"help": "Enable quaternion-based rotation rewards."},
    )
    rotation_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for rotation accuracy reward."},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_dir: str = field(
        default="data/",
        metadata={"help": "Directory containing the dataset"},
    )
    eval_dataset_name: str = field(
        default="",
        metadata={"help": "Name of the dataset"},
    )
    llava_dir: str = field(
        default="data/",
        metadata={"help": "Directory containing the llava dataset"},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    stop_steps: Optional[int] = field(
        default=7000,
        metadata={"help": "Number of steps for the early stop."},
    )
    # Multi-view settings
    multiview_enabled: bool = field(
        default=False,
        metadata={"help": "Enable multi-view training data."},
    )
    multiview_data_dir: str = field(
        default="./data/multiview",
        metadata={"help": "Directory containing multi-view data."},
    )
    multiview_selection: str = field(
        default="random",
        metadata={"help": "View selection strategy: random, all, original_only."},
    )
    rotation_query_weight: float = field(
        default=1.5,
        metadata={"help": "Weight multiplier for rotation-based queries."},
    )
