import pytest
from pathlib import Path
import torch

from src.pneumonia_detection.inference.predict import Predictor
from src.pneumonia_detection.utils.checkpoint import save_checkpoint
from src.pneumonia_detection.models.model import pneumoniaCNN

@pytest.fixture
def mock_checkpoint(dummy_config):
    """Save a dummy checkpoint state dict to test reload and inference operations.

    Args:
        dummy_config (Dict[str, Any]): Mock configuration fixture.
    """
    model = pneumoniaCNN(dummy_config)
    ckpt_dir = Path(dummy_config["training"]["checkpoint"]["dir"])
    ckpt_name = dummy_config["training"]["checkpoint"]["best_filename"]
    
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": None,
        "scheduler": None,
        "best_metric": 0.5,
        "config": dummy_config
    }
    
    save_checkpoint(state, str(ckpt_dir), ckpt_name)

def test_predictor_single_image(dummy_config, mock_checkpoint):
    """Test Predictor loads weights and scores a single test image correctly."""
    predictor = Predictor(dummy_config)
    
    # Select first synthetic sample file
    img_dir = Path(dummy_config["data"]["image_dir"])
    sample_img = next(img_dir.glob("*.png"))
    
    prob, class_idx, label = predictor.predict_image(sample_img)
    
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert class_idx in {0, 1}
    assert label in {"NORMAL", "PNEUMONIA"}
    
    # Assert classification decisions align with probability
    if prob > dummy_config["inference"]["threshold"]:
        assert class_idx == 1
        assert label == "PNEUMONIA"
    else:
        assert class_idx == 0
        assert label == "NORMAL"

def test_predictor_batch_directory(dummy_config, mock_checkpoint):
    """Test Predictor scans folders and generates complete metrics for each file."""
    predictor = Predictor(dummy_config)
    img_dir = Path(dummy_config["data"]["image_dir"])
    
    results = predictor.predict_batch(img_dir)
    
    assert isinstance(results, list)
    assert len(results) == 20
    
    first_res = results[0]
    required_keys = {"file_path", "file_name", "probability", "class_idx", "label"}
    assert required_keys.issubset(first_res.keys())
    assert first_res["class_idx"] in {0, 1}
    assert 0.0 <= first_res["probability"] <= 1.0
