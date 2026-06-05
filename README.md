# Pneumonia Detection Pipeline

[![Code Quality CI](https://github.com/Narayanakovuru/pneumonia-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Narayanakovuru/pneumonia-detection/actions/workflows/ci.yml)
[![Unit Tests CI](https://github.com/Narayanakovuru/pneumonia-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Narayanakovuru/pneumonia-detection/actions/workflows/tests.yml)

A production-grade, highly portable, and modular PyTorch deep learning pipeline for Pneumonia Detection in Chest X-Rays. The repository represents a professional Deep Learning Engineer project focusing strictly on backend engineering, reproducible pipelines, and test-driven development.

---

## 🚀 Key Features

* **Modular Core Design:** Adheres strictly to SOLID, DRY, and clean coding principles with strict separation of concerns.
* **Flexible YAML Configuration:** 100% configuration-driven. All hyperparameters, paths, and training choices are managed in `configs/` without code changes.
* **Custom CNN Architecture:** Uses an advanced modular network with `Residual Blocks`, `Squeeze-and-Excitation (SE) Blocks` for channel-wise feature calibration, global average pooling, and dropout layers.
* **Dynamic Loss Weighting:** Auto-calculates `pos_weight` ratio (`negatives / positives`) directly on training splits to counter class imbalance.
* **Data Leakage Safeguards:** Split boundaries are defined at the metadata level first. Data augmentations (affine, jitter, contrast) are applied strictly inside the training dataloader.
* **Local Containerization:** Docker support for GPU/CPU training, mapping local mounts for configurations, checkpoints, logs, and outputs.
* **Automated CI/CD:** GitHub Actions workflows configuration for code formatting check and test automation on push/pull requests.
* **Sanity Testing Utility:** Script to generate mock datasets locally to run the entire training/inference lifecycle out of the box in seconds.

---

## 📂 Project Structure

```text
├── .github/workflows/          # CI/CD pipelines (linter, unit tests)
├── configs/                    # Declarative YAML configurations
│   ├── data.yaml               # Paths, image size, split ratios, loaders
│   ├── model.yaml              # Stem, block channels, SE reduction, classifier hidden dim
│   ├── train.yaml              # Optimizer, learning rate, scheduler, loss type, checkpointing
│   └── inference.yaml          # Checkpoint path, decision threshold, outputs paths
├── data/                       # Dataset directories (gitignored)
│   └── raw/                    # Raw inputs and CSV files
├── checkpoints/                # Model weight file checkpoints (gitignored)
├── logs/                       # Running execution logging (gitignored)
├── notebooks/                  # Original experimental Jupyter X-Ray notebook
├── src/                        # Main source code package
│   └── pneumonia_detection/
│       ├── data/               # Custom datasets, augmentations, and split builders
│       ├── models/             # SEBlock, main pneumoniaCNN model, and BCE loss functions
│       ├── training/           # Epoch managers and Trainer loops
│       ├── evaluation/         # Test validation loaders and evaluation runs
│       ├── inference/          # Predictor engine for single or batch processing
│       ├── pipelines/          # Training and inference pipeline runners
│       └── utils/              # Hardware setups, seeding, metrics, logging, checkpoints
├── tests/                      # Pytest unit and integration test suite
├── Dockerfile                  # Container configurations (GPU enabled)
├── docker-compose.yml          # Container multi-services orchestrator
├── requirements.txt            # Package direct dependencies
├── setup.py                    # Backward compatibility packaging setup
└── README.md                   # Setup and execution guide
```

---

## 🛠️ Quick Start Setup

### Prerequisites
* Python 3.9+
* Virtual environment utility (`venv`)

### 1. Clone & Initialize Environment
```bash
git clone <repository_url>
cd pneumonia-detection

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Sanity Test Dataset Utility
To test the pipeline execution immediately without downloading the full 30GB Kaggle dataset, run the dummy data generator:
```bash
python scripts/generate_dummy_data.py
```
This generates 50 synthetic grayscale X-ray images and the matching `stage2_train_metadata.csv` under `data/raw/` automatically.

---

## 🏋️ Running the Training Pipeline

To run the training, validation, early stopping, and automatic checkpoint saving:
```bash
python -m src.pneumonia_detection.pipelines.train_pipeline --config configs/train.yaml
```
* **Imbalance Calculation:** Calculates dynamic weight scale (negative count / positive count) of training classes.
* **Optimizations:** Automatic GPU (CUDA) memory pinning and allocation if supported hardware is available.
* **Outputs:** Saves the best and last weights to `checkpoints/` and logs runtime details in `logs/training.log`.
* **Experiment Metadata:** Generates `checkpoints/experiment_metadata.json` containing total parameters count, hyperparameter states, and history of metrics.

---

## 🔍 Running the Inference Pipeline

Use a saved model checkpoint to predict chest X-ray labels:
```bash
# Predict a single image
python -m src.pneumonia_detection.pipelines.inference_pipeline --config configs/inference.yaml --input data/raw/Training/Images/dummy_patient_0000.png

# Batch predict an entire folder
python -m src.pneumonia_detection.pipelines.inference_pipeline --config configs/inference.yaml --input data/raw/Training/Images/
```
Output results are written directly to `predictions/batch_predictions.json` and logged to `logs/inference.log`.

---

## 🐳 Docker Containerized Training

You can run the pipeline inside a container with full GPU capability:
```bash
# Build and run the training pipeline
docker-compose up --build
```
Host folders (`configs/`, `data/`, `checkpoints/`, `logs/`, `predictions/`) are mounted into the container to persist all weights and predictions.

---

## 🧪 Testing and CI/CD

Unit and integration tests are automated using `pytest`. They cover data transforms, split builders, model layer outputs, trainer optimization loops, and checkpoint states.

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src
```

The workflows configured in `.github/workflows/` automatically run linting and the test suite on every commit to GitHub.

---

## 📈 Adapting to the Full Kaggle Dataset

No code changes are required to use the full 30GB Kaggle RSNA Pneumonia Detection dataset:
1. Download the dataset and place the images and `stage2_train_labels.csv` inside `data/raw/`.
2. Open `configs/data.yaml` and update the paths:
   ```yaml
   data:
     metadata_csv: "data/raw/stage2_train_labels.csv"
     image_dir: "data/raw/stage2_train_images"
   ```
3. Run the training pipeline command. The pipeline automatically loads the CSV, deduplicates patient IDs, calculates dynamic class weights, and splits the data into stratified subsets.
