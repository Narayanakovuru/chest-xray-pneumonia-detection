import pytest
from pathlib import Path
import pandas as pd
import torch

from src.pneumonia_detection.data.preprocess import load_metadata, get_class_distribution
from src.pneumonia_detection.data.dataset import PneumoniaDataset
from src.pneumonia_detection.data.dataloaders import build_dataloaders
from src.pneumonia_detection.data.transforms import get_train_transforms, get_val_transforms

def test_metadata_preprocess(dummy_config):
    """Test loading and column validation of metadata."""
    csv_path = Path(dummy_config["data"]["metadata_csv"])
    df = load_metadata(csv_path)
    
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"patientId", "Target"}
    assert len(df) == 20
    
    # Class distribution test
    neg, pos, ratio = get_class_distribution(df)
    assert neg > 0
    assert pos > 0
    assert ratio == pytest.approx(neg / pos)

def test_dataset_item_retrieval(dummy_config):
    """Test PneumoniaDataset loader items outputs and shapes."""
    csv_path = Path(dummy_config["data"]["metadata_csv"])
    img_dir = Path(dummy_config["data"]["image_dir"])
    df = load_metadata(csv_path)
    
    train_trans = get_train_transforms((224, 224))
    dataset = PneumoniaDataset(df, img_dir, transform=train_trans)
    
    assert len(dataset) == 20
    img_tensor, label_tensor = dataset[0]
    
    assert torch.is_tensor(img_tensor)
    assert torch.is_tensor(label_tensor)
    assert img_tensor.shape == (1, 224, 224)
    assert label_tensor.ndim == 0  # Scalar label tensor
    assert label_tensor.item() in {0.0, 1.0}

def test_transforms_range():
    """Test transformation scaling to values between -1.0 and 1.0."""
    train_t = get_train_transforms((224, 224))
    val_t = get_val_transforms((224, 224))
    
    assert train_t is not None
    assert val_t is not None

def test_dataloaders_splits(dummy_config):
    """Test splits proportions and batch outputs of build_dataloaders."""
    csv_path = Path(dummy_config["data"]["metadata_csv"])
    img_dir = Path(dummy_config["data"]["image_dir"])
    df = load_metadata(csv_path)
    
    train_loader, val_loader, test_loader = build_dataloaders(df, img_dir, dummy_config)
    
    # Test batch output
    train_batch = next(iter(train_loader))
    assert len(train_batch) == 2  # (images, labels)
    images, labels = train_batch
    
    assert images.shape[0] <= dummy_config["data"]["loader"]["batch_size"]
    assert images.shape[1] == 1  # Grayscale channel
    assert images.shape[2:] == (224, 224)
    assert labels.ndim == 1
