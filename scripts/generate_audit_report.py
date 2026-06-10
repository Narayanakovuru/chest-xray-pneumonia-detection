import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.pneumonia_detection.utils.config import load_all_configs
from src.pneumonia_detection.data.preprocess import load_metadata
from src.pneumonia_detection.data.dataloaders import build_dataloaders
from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.utils.checkpoint import load_checkpoint
from src.pneumonia_detection.utils.device import setup_device

def main():
    # 1. Load configuration
    config = load_all_configs("configs")
    
    # 2. Setup device
    device = setup_device(config["inference"].get("device", "cpu"))
    
    # 3. Load metadata and build loaders
    metadata_path = Path(config["data"]["metadata_csv"])
    img_dir = Path(config["data"]["image_dir"])
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)
        
    df = load_metadata(metadata_path)
    _, _, test_loader = build_dataloaders(df, img_dir, config)
    
    # 4. Load model
    model = pneumoniaCNN(config).to(device)
    checkpoint_path = Path(config["training"]["checkpoint"]["dir"]) / config["training"]["checkpoint"]["best_filename"]
    
    if not checkpoint_path.exists():
        print(f"Error: Best model checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    # Load weights
    load_checkpoint(str(checkpoint_path), model, device=device)
    model.eval()
    
    # 5. Collect predictions
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    
    # Decision threshold
    threshold = config["inference"].get("threshold", 0.5)
    y_pred = (y_prob > threshold).astype(int)
    
    # 6. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.5
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 7. Print Report
    report_str = f"""==================================================
              PNEUMONIA DETECTION AUDIT REPORT
==================================================
Decision Threshold: {threshold}
Total Test Samples: {len(y_true)}
Positive (Pneumonia) Samples: {sum(y_true == 1)}
Negative (Normal) Samples: {sum(y_true == 0)}

CLASSIFICATION METRICS:
--------------------------------------------------
Accuracy:  {acc:.4f} ({acc*100:.2f}%)
Precision: {prec:.4f} ({prec*100:.2f}%)
Recall:    {rec:.4f} ({rec*100:.2f}%)
F1-Score:  {f1:.4f} ({f1*100:.2f}%)
AUC-ROC:   {auc:.4f}

CONFUSION MATRIX:
--------------------------------------------------
                 Predicted Normal   Predicted Pneumonia
Actual Normal         {tn:<17}  {fp:<19}
Actual Pneumonia      {fn:<17}  {tp:<19}

DETAILED CLASSIFICATION REPORT:
--------------------------------------------------
{classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'], zero_division=0)}
==================================================
"""
    
    print(report_str)
    
    # Save report to a text file
    report_file = Path("predictions/audit_report.txt")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_str)
    print(f"\nAudit report successfully saved to {report_file.resolve()}")

if __name__ == "__main__":
    main()
