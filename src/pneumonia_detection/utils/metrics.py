import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from typing import Dict, Union, List

def calculate_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred_probs: Union[np.ndarray, List[float]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate accuracy, binary F1 score, and AUC-ROC score.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth binary array.
        y_pred_probs (Union[np.ndarray, List[float]]): Predicted probability scores.
        threshold (float): Decision threshold to convert probability to binary prediction.

    Returns:
        Dict[str, float]: Calculated scores dictionary.
    """
    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(y_pred_probs)
    
    # Binarize probabilities based on threshold
    preds_arr = (probs_arr > threshold).astype(float)
    
    # Calculate accuracy
    correct = (preds_arr == y_true_arr).sum()
    accuracy = float(correct / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0
    
    # Calculate F1 score
    f1 = float(f1_score(y_true_arr, preds_arr, zero_division=0))
    
    # Calculate AUC-ROC score
    # ROC metric requires both classes to be present in true labels
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) == 2:
        auc = float(roc_auc_score(y_true_arr, probs_arr))
    else:
        auc = 0.5  # Default baseline if validation set contains only 1 class
        
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc
    }
