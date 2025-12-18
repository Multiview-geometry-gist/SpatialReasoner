# Multi-View Training with MVGenMaster

This document describes the multi-view data augmentation system for spatial reasoning training using MVGenMaster.

## Overview

The system provides three main components:

1. **Bulk View Generation** (`mvgen_bulk_augmentation.py`): Generate 50-100 different angle views from each training image
2. **CoT Augmentation** (`cot_augmentation.py`): Modify chain-of-thought answers to reflect the viewing angle
3. **Multi-View Training** (`sft_multiview.py`): Train with either single random views or two views per sample

## Architecture

```
src/data_generation/
    mvgen_bulk_augmentation.py   # Bulk view generation
    cot_augmentation.py          # CoT modification
src/spatial_reasoner/
    sft_multiview.py             # Multi-view training
configs/mvgen_augmentation/
    bulk_augmentation.yaml       # MVGenMaster settings
    cot_augmentation.yaml        # CoT settings
recipes/Qwen2.5-VL-7B-Instruct/sft/
    config_mvgen_single.yaml     # Single-view training
    config_mvgen_multi.yaml      # Multi-view training
```

## Step 1: Generate Multi-View Data

Generate 50-100 views per image using MVGenMaster:

```bash
python -m src.data_generation.mvgen_bulk_augmentation \
    --input_dataset ccvl/SpatialReasonerTrain-SFT \
    --output_dir ./data/mvgen_augmented \
    --image_base_dir ./data/openimages \
    --num_views 50 \
    --azimuth_range 60.0 \
    --seed 42
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_views` | 50 | Target number of views per image |
| `--azimuth_range` | 60.0 | Rotation range in degrees (+/-) |
| `--elevation_range` | 15.0 | Elevation range in degrees |
| `--config` | None | YAML config file path |
| `--resume` | None | Resume from checkpoint |

### Output Format

```
./data/mvgen_augmented/
    augmented_dataset.json   # Main dataset with view metadata
    images/
        sample_001/
            view_+0.0.jpg    # Original view
            view_+10.0.jpg   # Rotated views
            view_+20.0.jpg
            ...
        sample_002/
            ...
    augmentation_config.yaml
    metadata.json
```

## Step 2: CoT Augmentation

The CoT augmentation modifies the `answer_cot` field to include view awareness:

```python
from src.data_generation.cot_augmentation import CoTAugmentor

augmentor = CoTAugmentor(strategy="prefix")

# Original: "Looking at the objects, the ball is to the left..."
# Augmented: "From a 15-degree right rotation, I can observe...
#            Looking at the objects, the ball is to the left..."
modified_cot = augmentor.augment(original_cot, view_angle=15.0)
```

### Augmentation Strategies

| Strategy | Description |
|----------|-------------|
| `prefix` | Add view context at the beginning |
| `inline` | Insert view references throughout |
| `suffix` | Add view summary at the end |
| `full` | Complete CoT rewrite with view awareness |

## Step 3: Training

### Single-View Training

Train with one randomly selected view per sample:

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    src/spatial_reasoner/sft_multiview.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_mvgen_single.yaml
```

Key settings:
- `mvgen_mode: single` - Use one view per sample
- `view_selection: random` - Randomly select view each epoch
- Views change each epoch for data augmentation effect

### Multi-View Training

Train with two views per sample (original + rotated):

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    src/spatial_reasoner/sft_multiview.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_mvgen_multi.yaml
```

Key settings:
- `mvgen_mode: multi` - Use two views per sample
- `multi_view_format: sequential` - How to present the two images
- `secondary_view_selection: diverse` - Select maximally different second view

### Multi-View Formats

| Format | Description | Example |
|--------|-------------|---------|
| `sequential` | Present views in order | "View 1: [img1] View 2: [img2] Question..." |
| `comparison` | Explicitly compare views | "Compare these views: [img1] and [img2]..." |
| `interleaved` | Weave views into prompt | "Looking at [img1], then from [img2]..." |

## Configuration Reference

### config_mvgen_single.yaml

```yaml
# MVGen settings
mvgen_enabled: true
mvgen_data_dir: ./data/mvgen_augmented/
mvgen_mode: single
view_selection: random

# CoT settings
cot_augmentation_enabled: true
cot_augmentation_strategy: prefix

# Training settings
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 3.0e-06
```

### config_mvgen_multi.yaml

```yaml
# MVGen settings
mvgen_enabled: true
mvgen_data_dir: ./data/mvgen_augmented/
mvgen_mode: multi
view_selection: random_pair

# Multi-view specific
multi_view_format: sequential
secondary_view_selection: diverse

# CoT settings
cot_augmentation_enabled: true
cot_augmentation_strategy: full

# Training settings (reduced batch size for 2 images)
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_length: 3072
```

## Best Practices

1. **View Generation**
   - Start with 50 views per image; increase to 100 if GPU time allows
   - Use azimuth range of 45-60 degrees for good coverage
   - Run on a single GPU (MVGenMaster is GPU-bound)

2. **Single-View Training**
   - Good for initial experiments and smaller GPU memory
   - Random view selection provides natural data augmentation
   - Each epoch sees different views of the same samples

3. **Multi-View Training**
   - Better for teaching view correspondence
   - Requires more GPU memory (2 images per sample)
   - Use `diverse` secondary selection for maximum learning

4. **CoT Augmentation**
   - `prefix` strategy is safest (minimal change to original reasoning)
   - `full` strategy best for multi-view (explicit view comparison)
   - Always set `preserve_answer: true` to keep final answer unchanged

## Extending the System

### Adding New View Selection Strategies

```python
# In sft_multiview.py, extend ViewSelector class
class ViewSelector:
    def select_single_view(self, view_images):
        if self.config.view_selection == "my_custom":
            # Custom selection logic
            ...
```

### Adding New CoT Strategies

```python
# In cot_augmentation.py
@StrategyRegistry.register
class MyCustomStrategy(AugmentationStrategy):
    name = "my_custom"

    def augment(self, original_cot, angle, metadata=None):
        # Custom augmentation logic
        ...
```

### Using with Custom Datasets

```python
from src.data_generation.mvgen_bulk_augmentation import MVGenBulkAugmentor, MVGenBulkConfig

config = MVGenBulkConfig(num_views=75, azimuth_range=45.0)
augmentor = MVGenBulkAugmentor(config)

# Augment from local JSON
augmentor.augment_dataset(
    input_dataset="./my_data/training.json",
    output_dir="./my_data/augmented",
    image_base_dir="./my_data/images",
)
```

## Troubleshooting

### MVGenMaster Timeout

If MVGenMaster times out during view generation:
- Increase `timeout_per_batch` in config (default: 600 seconds)
- Reduce `frames_per_trajectory` (default: 28)
- Check GPU memory usage

### Out of Memory During Training

For multi-view training:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to compensate
- Reduce `max_pixels` per image

### Missing Views

If some samples have fewer views than expected:
- Check MVGenMaster logs for errors
- Verify image quality (very dark/bright images may fail)
- Use `--resume` to continue from checkpoint
