import torch
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str
) -> Path:
    """Save the PyTorch training checkpoint state dict to a file.

    Args:
        state (Dict[str, Any]): Dict containing model/optimizer state dicts and metadata.
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        filename (str): Filename for the checkpoint file.

    Returns:
        Path: The path to the saved checkpoint file.
    """
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    full_path = save_path / filename
    
    # Save target dictionary
    torch.save(state, full_path)
    logger.info(f"Checkpoint saved: {full_path}")
    return full_path

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], int, float]:
    """Load model weight parameters and training stats from a saved checkpoint file.

    Args:
        checkpoint_path (str): Path to the target checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to restore state.
        scheduler (Optional[Any]): The scheduler to restore state.
        device (Optional[torch.device]): Target device.

    Returns:
        Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], int, float]:
            Loaded model, optimizer, scheduler, start_epoch, and best_metric.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
    logger.info(f"Loading checkpoint from: {path}")
    
    # Load mapping states
    if device is not None:
        checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path, map_location="cpu")
        
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Restored optimizer state from checkpoint.")
        
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Restored scheduler state from checkpoint.")
        
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    
    logger.info(f"Successfully loaded checkpoint: epoch={epoch} | best_metric={best_metric:.4f}")
    return model, optimizer, scheduler, epoch, best_metric
