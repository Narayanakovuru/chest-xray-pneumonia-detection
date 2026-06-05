# Pneumonia Detection Pipeline

[![Code Quality CI](https://github.com/lakshminarayana2409/pneumonia/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshminarayana2409/pneumonia/actions/workflows/ci.yml)
[![Unit Tests CI](https://github.com/lakshminarayana2409/pneumonia/actions/workflows/tests.yml/badge.svg)](https://github.com/lakshminarayana2409/pneumonia/actions/workflows/tests.yml)

A production-grade, highly portable, and modular PyTorch deep learning pipeline for Pneumonia Detection in Chest X-Rays. The project features a custom CNN classifier with skip-connections, Squeeze-and-Excitation (SE) blocks, dynamic class-imbalance loss weighting, and a robust training & evaluation suite.

---

## 🚀 Key Features

* **Modular Design:** Adheres strictly to SOLID, DRY, and clean coding principles with proper separation of concerns (configs, data processing, modeling, pipelines, testing).
* **Flexible Configurations:** Zero hardcoding. All hyperparameters, paths, and configurations are declared in YAML files under `configs/`.
* **Dynamic Class Imbalance Handling:** Supports dynamic calculation of positive weights for BCE loss (`auto_pos_weight: true`) based on the target class distribution in the training set.
* **Leakage-Free Augmentation:** Applies data augmentations (e.g. RandomRotation, RandomAffine, ColorJitter) strictly to the training split while keeping validation/test inputs clean.
* **High Portability:** Fully cross-platform (Windows, Linux, macOS) utilizing standard `pathlib.Path` objects. Contains no environment-specific hardcoded paths.
* **Fully Runnable Tests:** Synthetic on-the-fly dataset generation allows running `pytest` locally or on CI runners without requiring 30GB dataset downloads.

---

## 📂 Project Structure

```text
d:/Nanii/Pneumonia Detection/
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
├── src/                        # Main source code
│   └── pneumonia_detection/
│       ├── data/               # Custom datasets, augmentations, and split builders
│       ├── models/             # SEBlock, main pneumoniaCNN model, and BCE loss functions
│       ├── training/           # Epoch managers and Trainer loops
│       ├── evaluation/         # Test validation loaders and evaluation runs
│       ├── inference/          # Predictor engine for single or batch processing
│       ├── pipelines/          # Training and inference pipeline runners
│       └── utils/              # Hardware setups, seeding, metrics, logging, checkpoints
├── tests/                      # Pytest unit and integration test suite
├── requirements.txt            # Package direct dependencies
├── setup.py                    # Backward compatibility packaging setup
└── README.md                   # Setup and execution guide
```

---

## 🛠️ Quick Start Setup

### Prerequisites

* Python 3.9, 3.10, or 3.11 installed.
* Virtual environment utility (`venv`).

### 1. Clone & Initialize Environment
```bash
git clone <repository_url>
cd "Pneumonia Detection"

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

### 2. Dataset Placement
By default, the training pipeline searches for data inside `data/raw/`. Place your Kaggle dataset files as follows:

```text
data/
└── raw/
    ├── stage2_train_metadata.csv
    └── Training/
        └── Images/
            ├── <image_id_1>.png
            ├── <image_id_2>.png
            └── ...
```

*Note: You can easily modify paths, image locations, and CSV structures by editing `configs/data.yaml`.*

---

## 🏋️ Running the Training Pipeline

To launch training, validation, early stopping, and automatic checkpoint saving:

```bash
python -m src.pneumonia_detection.pipelines.train_pipeline
```

* **Optimizations:** CUDA memory optimizations are activated automatically if a compatible GPU is detected.
* **Outputs:** The best model weights will save to `checkpoints/best_pneumonia_model.pth` and logging events will print to the console and save in `logs/training.log`.

---

## 🔍 Running the Inference Pipeline

You can execute predictions using a saved model checkpoint. The inference input targets can be customized:

### Run Inference on a Single Image:
```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline --input data/raw/Training/Images/sample_xray.png
```
*Saves the single prediction details inside `predictions/prediction_sample_xray.json`.*

### Run Inference on a Directory of Images:
```bash
python -m src.pneumonia_detection.pipelines.inference_pipeline --input data/raw/Training/Images/
```
*Processes all matching PNG files inside the target folder and writes a summary report inside `predictions/batch_predictions.json`.*

---

## 🧪 Running Unit Tests

The project includes unit and integration tests covering dataset loading, conversions, models shape verification, trainers weight optimization, and prediction loading.

```bash
# Run all tests
pytest tests/

# Run tests with coverage reporting
pytest tests/ --cov=src
```


