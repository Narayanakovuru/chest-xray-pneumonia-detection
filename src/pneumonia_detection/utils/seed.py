import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to apply.
    """
    logger.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch backend settings for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
