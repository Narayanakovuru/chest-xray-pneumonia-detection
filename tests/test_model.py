import pytest
import torch

from src.pneumonia_detection.models.blocks import SEBlock
from src.pneumonia_detection.models.model import pneumoniaCNN

def test_se_block_forward():
    """Test standard forward pass and recalibration scaling of SEBlock."""
    batch_size = 4
    channels = 16
    height, width = 28, 28
    
    x = torch.randn(batch_size, channels, height, width)
    se = SEBlock(channels=channels, reduction=4)
    out = se(x)
    
    assert out.shape == x.shape
    # Output should not be identical but scaled
    assert not torch.allclose(out, x)

def test_cnn_architecture_shapes(dummy_config):
    """Test target outputs logits dimensions of the custom pneumoniaCNN architecture."""
    model = pneumoniaCNN(dummy_config)
    batch_size = 2
    
    # Input matching grayscale 224x224 X-Ray image
    x = torch.randn(batch_size, 1, 224, 224)
    logits = model(x)
    
    # Target label logit should be shape (batch_size, 1)
    assert logits.shape == (batch_size, 1)

def test_cnn_default_parameters():
    """Test cnn builds correctly with default settings (no config dictionary passed)."""
    model = pneumoniaCNN()
    x = torch.randn(1, 1, 224, 224)
    logits = model(x)
    
    assert logits.shape == (1, 1)
