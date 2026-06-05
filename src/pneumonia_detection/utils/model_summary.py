import torch
from torchinfo import summary
import logging
import sys
from pathlib import Path

# Ensure model packages are visible in system path BEFORE imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pneumonia_detection.models.model import pneumoniaCNN

logger = logging.getLogger(__name__)

def print_model_summary(input_size: tuple = (1, 1, 224, 224)) -> None:
    """Print structural layers, shape sizes, and total parameter statistics of pneumoniaCNN.

    Args:
        input_size (tuple): Tensor dimensions of mock input batch (batch_size, channels, height, width).
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing pneumoniaCNN architecture for summary extraction...")
    
    # Instantiate default model configuration
    model = pneumoniaCNN()
    
    logger.info(f"Generating summary with input dimensions: {input_size}")
    
    # Reconfigure stdout to use utf-8 on Windows console environments to handle box characters
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Extract structural details using torchinfo
    summary_info = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        verbose=0
    )
    print(summary_info)

if __name__ == "__main__":
    print_model_summary()
