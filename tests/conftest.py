import sys
from pathlib import Path

# Ensure package is visible in test runs executed without python -m
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from PIL import Image
import tempfile
from typing import Generator, Dict, Any

@pytest.fixture(scope="session")
def dummy_dataset_dir() -> Generator[Path, None, None]:
    """Create a temporary directory containing synthetic PNG files and a metadata CSV.

    Yields:
        Path: The path to the temporary dataset directory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create image directories
        img_dir = tmp_path / "Training" / "Images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dummy dataset details
        num_samples = 20
        patient_ids = [f"test_patient_{i}" for i in range(num_samples)]
        targets = [0 if i % 3 != 0 else 1 for i in range(num_samples)]  # Imbalanced
        
        # Create and save metadata CSV file
        df = pd.DataFrame({
            "patientId": patient_ids,
            "Target": targets
        })
        csv_path = tmp_path / "stage2_train_metadata.csv"
        df.to_csv(csv_path, index=False)
        
        # Create and save synthetic grayscale PNG images
        for pid in patient_ids:
            # Create a dummy 224x224 grayscale image (mode "L")
            img_data = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_data, mode="L")
            img.save(img_dir / f"{pid}.png")
            
        yield tmp_path

@pytest.fixture
def dummy_config(dummy_dataset_dir: Path) -> Dict[str, Any]:
    """Generate a baseline mock configuration matching the YAML schemas.

    Args:
        dummy_dataset_dir (Path): Temporary dataset directory fixture.

    Returns:
        Dict[str, Any]: Standardized configurations dictionary.
    """
    tmp_path = dummy_dataset_dir
    return {
        "data": {
            "metadata_csv": str(tmp_path / "stage2_train_metadata.csv"),
            "image_dir": str(tmp_path / "Training" / "Images"),
            "img_size": [224, 224],
            "num_channels": 1,
            "splits": {
                "train": 0.60,
                "val": 0.20,
                "test": 0.20,
                "seed": 42
            },
            "loader": {
                "batch_size": 4,          # Small batch size for tests
                "num_workers": 0,          # Set to 0 to avoid subprocess overhead in tests
                "pin_memory": False
            }
        },
        "model": {
            "name": "pneumoniaCNN",
            "in_channels": 1,
            "stem": {"out_channels": 8},   # Small values for rapid tests execution
            "block1": {"out_channels": 16},
            "block2": {"out_channels": 32},
            "block3": {"out_channels": 64},
            "block4": {"out_channels": 128},
            "se": {"reduction": 4},
            "classifier": {
                "dropout_1": 0.1,
                "dropout_2": 0.1,
                "hidden_dim": 32,
                "out_features": 1
            }
        },
        "training": {
            "seed": 42,
            "epochs": 2,
            "optimizer": {
                "type": "AdamW",
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4
            },
            "scheduler": {
                "type": "CosineAnnealingLR",
                "eta_min": 1.0e-5
            },
            "loss": {
                "type": "BCEWithLogitsLoss",
                "auto_pos_weight": True,
                "pos_weight": None
            },
            "grad_clipping": {
                "enabled": True,
                "max_norm": 1.0
            },
            "early_stopping": {
                "enabled": False,
                "patience": 5,
                "monitor": "val_loss",
                "mode": "min"
            },
            "checkpoint": {
                "dir": str(tmp_path / "checkpoints"),
                "best_filename": "best_model.pth",
                "last_filename": "last_model.pth",
                "resume": False
            },
            "logging": {
                "dir": str(tmp_path / "logs"),
                "log_file": "test_run.log",
                "level": "DEBUG"
            }
        },
        "inference": {
            "checkpoint_path": str(tmp_path / "checkpoints" / "best_model.pth"),
            "device": "cpu",
            "threshold": 0.5,
            "output_dir": str(tmp_path / "predictions"),
            "batch_size": 4
        }
    }
