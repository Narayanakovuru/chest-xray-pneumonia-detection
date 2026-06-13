# Pneumonia Detection Using Deep Learning

A deep learning pipeline for automated pneumonia detection from chest X-ray images using PyTorch. This project is built with a strong emphasis on reproducibility, modular software design, experiment management, and maintainable machine learning workflows.

## Overview

Pneumonia remains one of the leading causes of respiratory-related mortality worldwide. Early detection from chest radiographs can significantly improve treatment outcomes, but manual interpretation is time-consuming and dependent on clinical expertise.

This project explores binary classification of chest X-ray images to identify the presence of pneumonia. Beyond model development, the focus is on building a structured training and inference pipeline that can be extended, tested, and reproduced efficiently. The repository demonstrates how a medical imaging problem can be approached using a custom convolutional neural network architecture enhanced with Residual Connections and Squeeze-and-Excitation (SE) blocks, strictly adhering to modern machine learning engineering practices.

## Performance Metrics

After executing the training and evaluation pipeline, the model achieved the following metrics on the test dataset split:

- **Test Accuracy**: 37.50%
- **Test F1 Score**: 0.5455
- **Test AUC-ROC**: 0.8667
- **Test Loss**: 0.0867

*(Note: These metrics reflect evaluation on a lightweight dummy dataset split used for rapid local pipeline validation. Real-world performance on the full RSNA dataset will differ and require extended training epochs.)*

## Key Features

- **Modular Architecture**: Layered separation between data processing, model development, training logic, and inference pipelines.
- **Configuration-Driven**: All parameters (dataset paths, hyperparameters, architecture details) are managed via YAML files, enabling experiment reproduction without code changes.
- **Custom CNN Architecture**: Features Residual Blocks for stable gradient propagation, Squeeze-and-Excitation Blocks for channel attention, and Global Average Pooling.
- **Class Imbalance Handling**: Dynamic computation of class weights for weighted binary cross-entropy or focal loss to prevent bias towards the majority class.
- **Reproducibility & Tracking**: Fixed seed support, deterministic splits, config versioning, and checkpoint management.
- **CI/CD Ready**: Automated testing for all critical paths.

## Project Structure

```text
.
├── .github/workflows/         # CI pipelines
├── configs/                   # YAML configuration files (data, model, train, inference)
├── data/raw/                  # Dataset storage (gitignored)
├── checkpoints/               # Saved model weights and metadata
├── logs/                      # Training and inference logs
├── notebooks/                 # Exploratory data analysis and prototyping
├── src/pneumonia_detection/   # Core source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Architecture definitions and losses
│   ├── training/              # Training loops and validation
│   ├── evaluation/            # Evaluation metrics calculation
│   ├── inference/             # Inference wrappers
│   ├── pipelines/             # Entrypoint scripts
│   └── utils/                 # Utility functions (logging, device, seeds)
├── tests/                     # Unit and integration test suite
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation setup
```

## Installation

### Prerequisites
- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<username>/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Create and Activate a Virtual Environment**
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Quick Pipeline Validation

A lightweight synthetic dataset generator is provided to validate the complete workflow without needing to download the full medical dataset.

```bash
python scripts/generate_dummy_data.py
```
This tests data loading, training loops, validation, and inference locally.

## Training

Execute the training pipeline using the provided configuration:

```bash
python -m src.pneumonia_detection.pipelines.train_pipeline --config configs/train.yaml
```

The pipeline handles stratified splitting, dynamic class weight calculation, early stopping, and checkpoint persistence automatically. Artifacts are saved to `checkpoints/` and `logs/`.

## Inference

### Single Image Prediction
```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline \
    --config configs/inference.yaml \
    --input path/to/image.png
```

### Batch Directory Prediction
```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline \
    --config configs/inference.yaml \
    --input path/to/image_directory/
```

Predictions are automatically exported as JSON files for downstream use.

## Docker Support

Build and execute the project within a containerized environment:

```bash
docker-compose up --build
```
Volumes are mounted so that configurations, checkpoints, logs, and datasets persist locally.

## Testing

Execute the comprehensive test suite and generate a coverage report:

```bash
pytest
pytest --cov=src
```

## Dataset

The project relies on the **RSNA Pneumonia Detection Challenge** dataset. After acquiring the raw images and metadata, update `configs/data.yaml` appropriately:

```yaml
data:
  metadata_csv: "data/raw/stage2_train_labels.csv"
  image_dir: "data/raw/stage2_train_images"
```

## Engineering Considerations

- **Test-first Validation**: Critical modules feature comprehensive test coverage.
- **Structured Logging**: Consistent logging across all components for simplified debugging.
- **Separation of Concerns**: Training, validation, and inference workflows run totally independent of one another.

## Future Improvements

- Incorporating Transfer Learning with robust pretrained backbones (e.g., ResNet50, DenseNet).
- Adding Explainable AI (XAI) using Grad-CAM to highlight infectious regions.
- Enabling experiment tracking through MLflow or Weights & Biases.
- Expanding into a distributed training setup.
- Containerized REST API for real-time model serving.

## License

This project is open-source and intended solely for **educational and research purposes**. 

> **Disclaimer:** Medical predictions generated by this model should not be used as a substitute for professional clinical diagnosis.
