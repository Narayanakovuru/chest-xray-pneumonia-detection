import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_criterion(
    config: Dict[str, Any],
    neg_count: Optional[int] = None,
    pos_count: Optional[int] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Instantiate and configure the Loss Function (criterion).

    Supports dynamic estimation of pos_weight based on positive and negative
    class distributions (imbalance management) or statically specified values.

    Args:
        config (Dict[str, Any]): Full training configurations.
        neg_count (Optional[int]): Number of negative class targets in train split.
        pos_count (Optional[int]): Number of positive class targets in train split.
        device (Optional[torch.device]): Target GPU/CPU execution device.

    Returns:
        nn.Module: Configured Loss Criterion (BCEWithLogitsLoss).
    """
    training_config = config.get("training", {}) or {}
    loss_config = training_config.get("loss", {}) or {}
    loss_type = loss_config.get("type", "BCEWithLogitsLoss")
    auto_pos_weight = loss_config.get("auto_pos_weight", True)
    
    if loss_type != "BCEWithLogitsLoss":
        raise ValueError(f"Unsupported loss function: {loss_type}. Currently only BCEWithLogitsLoss is implemented.")
        
    pos_weight_tensor = None
    
    if auto_pos_weight:
        if neg_count is None or pos_count is None:
            logger.warning("auto_pos_weight=True but class counts were not provided. Defaulting to unweighted BCE.")
        else:
            if pos_count == 0:
                raise ValueError("Positive class count is 0, cannot calculate dynamic pos_weight.")
            computed_ratio = float(neg_count / pos_count)
            pos_weight_tensor = torch.tensor([computed_ratio], dtype=torch.float32)
            logger.info(f"Loss criterion configured with DYNAMIC pos_weight: {computed_ratio:.4f}")
    else:
        static_val = loss_config.get("pos_weight", None)
        if static_val is not None:
            pos_weight_tensor = torch.tensor([float(static_val)], dtype=torch.float32)
            logger.info(f"Loss criterion configured with STATIC pos_weight: {static_val:.4f}")
        else:
            logger.info("Loss criterion configured with unweighted BCE (pos_weight=None).")
            
    if pos_weight_tensor is not None and device is not None:
        pos_weight_tensor = pos_weight_tensor.to(device)
        
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
