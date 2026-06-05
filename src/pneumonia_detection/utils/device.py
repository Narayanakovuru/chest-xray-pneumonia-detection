import os
import gc
import torch
import logging

logger = logging.getLogger(__name__)

def setup_device(device_setting: str = "auto") -> torch.device:
    """Initialize torch.device based on the configurations and system compatibility.

    Args:
        device_setting (str): Options are "auto", "cuda", or "cpu".

    Returns:
        torch.device: The selected PyTorch device.
    """
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_setting == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA is requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    logger.info(f"Initialized training device: {device}")
    
    # Configure memory allocator optimizations if CUDA is active
    if device.type == "cuda":
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("CUDA optimizations applied: PYTORCH_ALLOC_CONF=expandable_segments:True")
        
    return device

def clean_memory() -> None:
    """Explicitly release unused cache memory from system RAM and GPU VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug("GPU cache and garbage collector cleared.")
