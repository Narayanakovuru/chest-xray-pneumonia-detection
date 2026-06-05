#!/usr/bin/env bash
set -e

# Default to raw image directory if not provided
INPUT_PATH=${1:-"data/raw/Training/Images/"}

echo "Running Inference Pipeline on target: $INPUT_PATH"

python -m src.pneumonia_detection.pipelines.inference_pipeline --input "$INPUT_PATH" --config-dir configs/
