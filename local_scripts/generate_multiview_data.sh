#!/bin/bash
# Generate multi-view spatial reasoning dataset
#
# Usage:
#   bash local_scripts/generate_multiview_data.sh [NUM_IMAGES]
#
# Example:
#   bash local_scripts/generate_multiview_data.sh 1000  # Process 1000 images
#   bash local_scripts/generate_multiview_data.sh       # Use default (20000)

NUM_IMAGES=${1:-20000}
INPUT_DIR="./data/openimages"
OUTPUT_DIR="./data/multiview"
CONFIG="configs/data_generation/default.yaml"

echo "======================================"
echo "Multi-View Data Generation"
echo "======================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of images: $NUM_IMAGES"
echo "Config: $CONFIG"
echo "======================================"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    echo "Please ensure your images are in $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run generation
python scripts/generate_data.py \
    --config "$CONFIG" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_images "$NUM_IMAGES" \
    --seed 42

echo ""
echo "Generation complete!"
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - qa_pairs.json: QA pairs for training"
echo "  - metadata.json: Dataset metadata"
echo "  - images/: Multi-view images"
echo "  - depth/: Depth maps"
echo "  - poses/: Object poses"
