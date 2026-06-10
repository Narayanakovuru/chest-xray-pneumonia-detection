import sys
import logging
import copy
from pathlib import Path

# Ensure package visibility
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torchvision.models import resnet18, ResNet18_Weights

from src.pneumonia_detection.utils.config import load_all_configs
from src.pneumonia_detection.data.preprocess import load_metadata
from src.pneumonia_detection.data.dataloaders import build_dataloaders
from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.models.losses import get_criterion
from src.pneumonia_detection.data.dataset import PneumoniaDataset
from src.pneumonia_detection.data.transforms import get_strong_train_transforms

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=3):
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        if scheduler is not None:
            scheduler.step()
            
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
    return model

def evaluate_on_test(model, test_loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, auc, cm

def main():
    print("=" * 75)
    print("   CUSTOM CNN VS CUSTOM CNN V2 (REGULARIZED) VS RESNET-18 BASELINE")
    print("=" * 75)
    
    # 1. Load Configurations and Data
    config = load_all_configs("configs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_cfg = config["data"]
    metadata_path = Path(data_cfg["metadata_csv"])
    img_dir = Path(data_cfg["image_dir"])
    
    df = load_metadata(metadata_path)
    train_loader, val_loader, test_loader = build_dataloaders(df, img_dir, config)
    
    # Construct strong train loader for CNN v2
    split_config = data_cfg["splits"]
    loader_config = data_cfg["loader"]
    img_size = tuple(data_cfg["img_size"])
    seed = split_config["seed"]
    
    train_df, _ = train_test_split(
        df,
        test_size=(split_config["val"] + split_config["test"]),
        random_state=seed,
        stratify=df["Target"]
    )
    
    strong_train_dataset = PneumoniaDataset(
        dataframe=train_df,
        img_dir=img_dir,
        transform=get_strong_train_transforms(img_size)
    )
    strong_train_loader = torch.utils.data.DataLoader(
        strong_train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"]
    )
    
    epochs = 3
    
    # 2. Build Models
    # Custom CNN Model v1
    logger.info("Initializing Custom pneumoniaCNN Model v1...")
    custom_cnn_v1 = pneumoniaCNN(config).to(device)
    
    # Custom CNN Model v2 (Regularized)
    logger.info("Initializing Custom pneumoniaCNN Model v2 (Regularized & Augmented)...")
    config_v2 = copy.deepcopy(config)
    if "model" in config_v2:
        # Increase dropout rates
        if "classifier" in config_v2["model"]:
            config_v2["model"]["classifier"]["dropout_1"] = 0.5
            config_v2["model"]["classifier"]["dropout_2"] = 0.5
    custom_cnn_v2 = pneumoniaCNN(config_v2).to(device)
    
    # ResNet-18 Model
    logger.info("Initializing ResNet-18 Model...")
    try:
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        logger.info("Successfully loaded pre-trained ResNet-18 weights.")
    except Exception as e:
        logger.warning(f"Could not load pre-trained weights ({e}). Initializing untrained ResNet-18.")
        resnet = resnet18(weights=None)
        
    # Adapt ResNet-18 for 1-channel grayscale input
    pretrained_weights = resnet.conv1.weight.clone()
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.conv1.weight.data = pretrained_weights.sum(dim=1, keepdim=True)
    
    # Adapt ResNet-18 final fully connected layer for binary classification
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    resnet = resnet.to(device)
    
    # 3. Setup Optimizers, Schedulers, and Criterion
    criterion = get_criterion(config, device=device)
    
    training_cfg = config.get("training", {}) or {}
    opt_cfg = training_cfg.get("optimizer", {}) or {}
    base_lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    
    # V1 Optimizer (standard weight decay)
    custom_opt_v1 = torch.optim.AdamW(custom_cnn_v1.parameters(), lr=base_lr, weight_decay=wd)
    custom_sched_v1 = torch.optim.lr_scheduler.CosineAnnealingLR(custom_opt_v1, T_max=epochs)
    
    # V2 Optimizer (strong weight decay regularization = 1e-2)
    custom_opt_v2 = torch.optim.AdamW(custom_cnn_v2.parameters(), lr=base_lr, weight_decay=1e-2)
    custom_sched_v2 = torch.optim.lr_scheduler.CosineAnnealingLR(custom_opt_v2, T_max=epochs)
    
    # ResNet-18 Optimizer
    resnet_opt = torch.optim.AdamW(resnet.parameters(), lr=base_lr, weight_decay=wd)
    resnet_sched = torch.optim.lr_scheduler.CosineAnnealingLR(resnet_opt, T_max=epochs)
    
    # 4. Train Custom CNN v1
    logger.info("\n--- Training Custom CNN v1 (Standard) ---")
    custom_cnn_v1 = train_and_validate(custom_cnn_v1, train_loader, val_loader, criterion, custom_opt_v1, custom_sched_v1, device, epochs)
    
    # 5. Train Custom CNN v2
    logger.info("\n--- Training Custom CNN v2 (Strong Regularization & Augmentation) ---")
    custom_cnn_v2 = train_and_validate(custom_cnn_v2, strong_train_loader, val_loader, criterion, custom_opt_v2, custom_sched_v2, device, epochs)
    
    # 6. Train ResNet-18
    logger.info("\n--- Training ResNet-18 Baseline ---")
    resnet = train_and_validate(resnet, train_loader, val_loader, criterion, resnet_opt, resnet_sched, device, epochs)
    
    # 7. Evaluate on Test Set
    logger.info("\n--- Evaluating Models on Test Split ---")
    v1_acc, v1_prec, v1_rec, v1_f1, v1_auc, v1_cm = evaluate_on_test(custom_cnn_v1, test_loader, device)
    v2_acc, v2_prec, v2_rec, v2_f1, v2_auc, v2_cm = evaluate_on_test(custom_cnn_v2, test_loader, device)
    r_acc, r_prec, r_rec, r_f1, r_auc, r_cm = evaluate_on_test(resnet, test_loader, device)
    
    # Generate Comparison Report
    report = f"""===========================================================================
        CUSTOM CNN V1 VS V2 (REGULARIZED + AUG) VS RESNET-18 BASELINE
===========================================================================
Total Test Samples: {len(test_loader.dataset)}

METRICS SUMMARY:
---------------------------------------------------------------------------
Metric    | Custom CNN v1 (Standard) | Custom CNN v2 (Regularized) | ResNet-18
---------------------------------------------------------------------------
Accuracy  | {v1_acc:.4f} ({v1_acc*100:.2f}%)       | {v2_acc:.4f} ({v2_acc*100:.2f}%)          | {r_acc:.4f} ({r_acc*100:.2f}%)
Precision | {v1_prec:.4f} ({v1_prec*100:.2f}%)       | {v2_prec:.4f} ({v2_prec*100:.2f}%)          | {r_prec:.4f} ({r_prec*100:.2f}%)
Recall    | {v1_rec:.4f} ({v1_rec*100:.2f}%)       | {v2_rec:.4f} ({v2_rec*100:.2f}%)          | {r_rec:.4f} ({r_rec*100:.2f}%)
F1-Score  | {v1_f1:.4f} ({v1_f1*100:.2f}%)       | {v2_f1:.4f} ({v2_f1*100:.2f}%)          | {r_f1:.4f} ({r_f1*100:.2f}%)
AUC-ROC   | {v1_auc:.4f}                     | {v2_auc:.4f}                        | {r_auc:.4f}

CONFUSION MATRICES:
---------------------------------------------------------------------------
Custom CNN v1 (Standard):
                 Predicted Normal   Predicted Pneumonia
Actual Normal         {v1_cm[0,0]}                  {v1_cm[0,1]}                  
Actual Pneumonia      {v1_cm[1,0]}                  {v1_cm[1,1]}                  

Custom CNN v2 (Regularized):
                 Predicted Normal   Predicted Pneumonia
Actual Normal         {v2_cm[0,0]}                  {v2_cm[0,1]}                  
Actual Pneumonia      {v2_cm[1,0]}                  {v2_cm[1,1]}                  

ResNet-18 Baseline:
                 Predicted Normal   Predicted Pneumonia
Actual Normal         {r_cm[0,0]}                  {r_cm[0,1]}                  
Actual Pneumonia      {r_cm[1,0]}                  {r_cm[1,1]}                  
===========================================================================
"""
    print(report)
    
    # Write report file
    report_path = Path("predictions/resnet_comparison_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Comparison report saved to {report_path.resolve()}")

if __name__ == "__main__":
    main()
