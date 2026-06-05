#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Pneumonia Detection Training Pipeline..."

# Run training pipeline with path relative configs
python -m src.pneumonia_detection.pipelines.train_pipeline --config configs/train.yaml
