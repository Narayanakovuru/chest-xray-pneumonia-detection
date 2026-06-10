import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Focuses learning on hard examples by down-weighting easy ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initialize FocalLoss.

        Args:
            alpha (float): Positive class weight factor.
            gamma (float): Focusing parameter for hard examples.
            reduction (str): Reduction mode: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): Logits predictions from the model.
            targets (torch.Tensor): Target labels of identical shape.

        Returns:
            torch.Tensor: Computed loss.
        """
        targets = targets.to(inputs.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

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
        nn.Module: Configured Loss Criterion (FocalLoss or BCEWithLogitsLoss).
    """
    training_config = config.get("training", {}) or {}
    loss_config = training_config.get("loss", {}) or {}
    loss_type = loss_config.get("type", "BCEWithLogitsLoss")
    auto_pos_weight = loss_config.get("auto_pos_weight", True)
    
    if loss_type == "FocalLoss":
        alpha = float(loss_config.get("alpha", 0.25))
        gamma = float(loss_config.get("gamma", 2.0))
        logger.info(f"Loss criterion configured with FocalLoss (alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma)
        
    elif loss_type == "BCEWithLogitsLoss":
        pos_weight_tensor = None
        if auto_pos_weight:
            if neg_count is None or pos_count is None:
                logger.warning("auto_pos_weight=True but class counts were not provided. Defaulting to unweighted BCE.")
            else:
                if pos_count == 0:
                    raise ValueError("Positive class count is 0, cannot calculate dynamic pos_weight.")
                computed_ratio = neg_count / pos_count
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
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}. Supported types: FocalLoss, BCEWithLogitsLoss.")
