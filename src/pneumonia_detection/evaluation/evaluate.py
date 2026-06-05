import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.models.losses import get_criterion
from src.pneumonia_detection.utils.metrics import calculate_metrics
from src.pneumonia_detection.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """Evaluate a model's performance on the test dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for testing split.
        criterion (torch.nn.Module): The evaluation loss criterion.
        device (torch.device): Device to run evaluation on.
        threshold (float): Prediction probability threshold.

    Returns:
        Tuple[float, Dict[str, float]]:
            Average test loss and dictionary of evaluation metrics (accuracy, f1, auc).
    """
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_probs = []
    
    logger.info("Starting test set evaluation.")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_test_loss = test_loss / len(test_loader)
    metrics = calculate_metrics(all_labels, all_probs, threshold=threshold)
    
    logger.info(
        f"Evaluation Results - Test Loss: {avg_test_loss:.4f} | "
        f"Test Accuracy: {metrics['accuracy'] * 100.0:.2f}% | "
        f"Test F1: {metrics['f1_score']:.4f} | "
        f"Test AUC-ROC: {metrics['auc_roc']:.4f}"
    )
    
    return avg_test_loss, metrics

def load_and_evaluate(
    config: Dict[str, Any],
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Instantiate a model, load weights from a checkpoint path, and run evaluation.

    Args:
        config (Dict[str, Any]): Full configurations.
        test_loader (DataLoader): Testing dataset loader.
        device (torch.device): Device to evaluate on.

    Returns:
        Tuple[float, Dict[str, float]]: Average loss and metrics.
    """
    # 1. Instantiate Model
    model = pneumoniaCNN(config).to(device)
    
    # 2. Setup Criterion
    criterion = get_criterion(config, device=device)
    
    # 3. Load Checkpoint
    checkpoint_path = config["inference"]["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Requested checkpoint not found at: {checkpoint_path}")
        
    model, _, _, _, _ = load_checkpoint(checkpoint_path, model, device=device)
    
    # 4. Evaluate
    threshold = config["inference"].get("threshold", 0.5)
    return evaluate_model(model, test_loader, criterion, device, threshold)
