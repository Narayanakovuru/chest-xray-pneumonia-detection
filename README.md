# Pneumonia Detection Using Deep Learning

A deep learning pipeline for automated pneumonia detection from chest X-ray images using PyTorch. This project was built with a strong emphasis on reproducibility, modular software design, experiment management, and maintainable machine learning workflows.

The repository demonstrates how a medical imaging problem can be approached using a custom convolutional neural network architecture enhanced with Residual Connections and Squeeze-and-Excitation (SE) blocks while following modern machine learning engineering practices.

## Overview

Pneumonia remains one of the leading causes of respiratory-related mortality worldwide. Early detection from chest radiographs can significantly improve treatment outcomes, but manual interpretation is time-consuming and dependent on clinical expertise.

This project explores binary classification of chest X-ray images to identify the presence of pneumonia. Beyond model development, the focus is on building a structured training and inference pipeline that can be extended, tested, and reproduced efficiently.

---

## Key Features

### Modular Project Architecture

The codebase follows a layered architecture with clear separation between:

* Data processing
* Model development
* Training logic
* Evaluation workflows
* Inference pipelines
* Utility services

This structure enables independent development, testing, and maintenance of each component.

### Configuration-Driven Experiments

All major parameters are managed through YAML configuration files, including:

* Dataset paths
* Image preprocessing settings
* Model architecture parameters
* Training hyperparameters
* Inference settings

Experiments can be reproduced or modified without changing source code.

### Custom CNN Architecture

The model incorporates:

* Residual Blocks for stable gradient propagation
* Squeeze-and-Excitation Blocks for channel attention
* Global Average Pooling
* Fully Connected Classification Layers
* Dropout Regularization

This design balances model capacity, computational efficiency, and generalization performance.

### Class Imbalance Handling

The training pipeline dynamically computes class weights from the training split and applies weighted binary cross-entropy loss to reduce bias toward the majority class.

### Reproducibility

The pipeline includes:

* Fixed random seed support
* Deterministic dataset splitting
* Configuration versioning
* Experiment metadata tracking
* Checkpoint management

### Testing and Continuous Integration

Automated workflows validate:

* Data processing components
* Model outputs
* Training loops
* Utility functions
* End-to-end pipeline execution

GitHub Actions automatically execute validation checks on pull requests and code updates.

---

## Project Structure

```text
.
├── .github/
│   └── workflows/             # CI pipelines
│
├── configs/                   # YAML configuration files
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── inference.yaml
│
├── data/
│   └── raw/                   # Dataset storage (gitignored)
│
├── checkpoints/               # Saved model weights
├── logs/                      # Training and inference logs
├── notebooks/                 # Initial experimentation notebooks
│
├── src/
│   └── pneumonia_detection/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── pipelines/
│       └── utils/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

### Prerequisites

* Python 3.9 or higher
* pip
* Virtual environment (recommended)

### Clone the Repository

```bash
git clone https://github.com/<username>/pneumonia-detection.git

cd pneumonia-detection
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt

pip install -e .
```

---

## Quick Pipeline Validation

A lightweight synthetic dataset generator is provided for validating the complete workflow without downloading the full dataset.

```bash
python scripts/generate_dummy_data.py
```

This generates sample images and metadata required to test:

* Data loading
* Training
* Validation
* Checkpoint creation
* Inference

---

## Training

Run the training pipeline using:

```bash
python -m src.pneumonia_detection.pipelines.train_pipeline \
    --config configs/train.yaml
```

The training workflow performs:

1. Dataset loading and preprocessing
2. Stratified dataset splitting
3. Dynamic class weight computation
4. Model training and validation
5. Early stopping
6. Checkpoint persistence
7. Metric logging

Training artifacts are stored in:

```text
checkpoints/
logs/
```

---

## Inference

Single-image prediction:

```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline \
    --config configs/inference.yaml \
    --input path/to/image.png
```

Batch prediction:

```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline \
    --config configs/inference.yaml \
    --input path/to/image_directory/
```

Predictions are exported as JSON files for downstream consumption.

---

## Docker Support

Build and execute the project using Docker:

```bash
docker-compose up --build
```

Mounted volumes ensure persistence of:

```text
configs/
data/
checkpoints/
logs/
predictions/
```

This allows experiments to be reproduced consistently across environments.

---

## Testing

Execute the full test suite:

```bash
pytest
```

Generate a coverage report:

```bash
pytest --cov=src
```

The testing strategy includes:

* Unit tests
* Integration tests
* Pipeline validation tests
* Configuration validation

---

## Dataset

The project is designed to work with the RSNA Pneumonia Detection Challenge dataset.

After downloading the dataset, update the paths in:

```yaml
configs/data.yaml
```

Example:

```yaml
data:
  metadata_csv: "data/raw/stage2_train_labels.csv"
  image_dir: "data/raw/stage2_train_images"
```

No code modifications are required to switch from the dummy dataset to the full dataset.

---

## Engineering Considerations

Several design decisions were incorporated during development:

* Configuration-driven experimentation
* Separation of training and inference workflows
* Automatic class imbalance handling
* Checkpoint versioning
* Structured logging
* Test-first validation of critical modules
* Containerized execution environments

These practices improve maintainability and make the project easier to extend for future experimentation.

---

## Future Improvements

Potential extensions include:

* Transfer learning with pretrained backbones
* Model explainability using Grad-CAM
* Hyperparameter optimization
* Experiment tracking with MLflow
* Distributed training support
* REST API deployment
* Model monitoring and drift detection

---

## License

This project is intended for educational and research purposes.

Medical predictions generated by this model should not be used as a substitute for professional clinical diagnosis.
