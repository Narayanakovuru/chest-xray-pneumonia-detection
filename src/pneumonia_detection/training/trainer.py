import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from src.pneumonia_detection.utils.metrics import calculate_metrics
from src.pneumonia_detection.utils.checkpoint import save_checkpoint, load_checkpoint
from src.pneumonia_detection.utils.device import clean_memory

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for Pneumonia Detection CNN.

    Manages training steps, validation steps, metrics calculations, early stopping,
    and checkpoint persistence (resuming training, saving best checkpoints).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ) -> None:
        """Initialize Trainer.

        Args:
            model (nn.Module): The model to train.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): Optimization parameter manager.
            scheduler (Optional[Any]): Learning rate scheduler.
            device (torch.device): GPU or CPU execution device.
            train_loader (DataLoader): DataLoader for the training split.
            val_loader (DataLoader): DataLoader for the validation split.
            config (Dict[str, Any]): Full YAML configurations.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Parse train configurations
        train_config = config["training"]
        self.epochs = train_config.get("epochs", 50)
        
        # Regularization (Gradient clipping)
        clip_config = train_config.get("grad_clipping", {})
        self.clip_enabled = clip_config.get("enabled", True)
        self.clip_max_norm = clip_config.get("max_norm", 1.0)
        
        # Checkpoint directories
        ckpt_config = train_config.get("checkpoint", {})
        self.checkpoint_dir = Path(ckpt_config.get("dir", "checkpoints"))
        self.best_filename = ckpt_config.get("best_filename", "best_pneumonia_model.pth")
        self.last_filename = ckpt_config.get("last_filename", "last_pneumonia_model.pth")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early Stopping configurations
        es_config = train_config.get("early_stopping", {})
        self.es_enabled = es_config.get("enabled", True)
        self.es_patience = es_config.get("patience", 10)
        self.es_monitor = es_config.get("monitor", "val_loss")
        self.es_mode = es_config.get("mode", "min")
        self.es_counter = 0
        
        # Initialize tracking metrics
        self.start_epoch = 0
        if self.es_mode == "min":
            self.best_metric_value = float("inf")
        else:
            self.best_metric_value = -float("inf")
            
        self.metrics_history: List[Dict[str, Any]] = []
        logger.info("Trainer initialized successfully.")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Perform one epoch of model training.

        Args:
            epoch (int): The current epoch index (1-indexed).

        Returns:
            Tuple[float, float]: Average training loss and average training accuracy.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.clip_enabled:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
                
            self.optimizer.step()
            
            # Record loss and accuracy statistics
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = (correct / total) if total > 0 else 0.0
        
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model over the validation set.

        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and dict of validation metrics.
        """
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        avg_val_loss = val_loss / len(self.val_loader)
        metrics = calculate_metrics(all_labels, all_probs, threshold=0.5)
        
        return avg_val_loss, metrics

    def fit(self) -> None:
        """Run the full training, validation, checkpointing, and evaluation lifecycle."""
        logger.info(f"Starting model fitting. Total epochs: {self.epochs}")
        
        # Check if user requested to resume from last checkpoint
        resume_flag = self.config["training"]["checkpoint"].get("resume", False)
        last_ckpt_path = self.checkpoint_dir / self.last_filename
        
        if resume_flag and last_ckpt_path.exists():
            logger.info(f"Resuming training from last checkpoint: {last_ckpt_path}")
            self.model, self.optimizer, self.scheduler, self.start_epoch, loaded_best = load_checkpoint(
                str(last_ckpt_path),
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            # Retain best metric value if it matches the monitored metric
            # (Note: we reset this if the loader is different, but for continuous run we retain it)
            if loaded_best != 0.0:
                self.best_metric_value = loaded_best
                
        for epoch in range(self.start_epoch, self.epochs):
            clean_memory()
            
            logger.info(f"Epoch [{epoch + 1}/{self.epochs}] starting.")
            
            # 1. Train step
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 2. Validation step
            val_loss, val_metrics = self.validate_epoch()
            
            # Extract metrics
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_score"]
            val_auc = val_metrics["auc_roc"]
            
            # Step scheduler if defined
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}] completed. "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100.0:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100.0:.2f}% | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Record epoch results to metrics history
            epoch_results = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_f1_score": float(val_f1),
                "val_auc_roc": float(val_auc),
                "learning_rate": float(current_lr)
            }
            self.metrics_history.append(epoch_results)
            self.save_experiment_metadata()
            
            # Save checkpoint state dict dictionary
            checkpoint_state = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_metric": self.best_metric_value,
                "config": self.config
            }
            
            # Always save as last checkpoint to support resuming
            save_checkpoint(checkpoint_state, str(self.checkpoint_dir), self.last_filename)
            
            # Check if validation metric improved
            is_better = False
            monitored_val = val_loss if self.es_monitor == "val_loss" else (val_acc if self.es_monitor == "val_acc" else val_f1)
            
            if self.es_mode == "min":
                if monitored_val < self.best_metric_value:
                    self.best_metric_value = monitored_val
                    is_better = True
            else:
                if monitored_val > self.best_metric_value:
                    self.best_metric_value = monitored_val
                    is_better = True
                    
            if is_better:
                logger.info(f"Validation metric {self.es_monitor} improved. Saving new BEST model checkpoint.")
                checkpoint_state["best_metric"] = self.best_metric_value
                save_checkpoint(checkpoint_state, str(self.checkpoint_dir), self.best_filename)
                self.es_counter = 0  # Reset early stopping counter
            else:
                self.es_counter += 1
                logger.info(f"Early stopping metric did not improve. Counter: {self.es_counter}/{self.es_patience}")
                
            # Early Stopping Check
            if self.es_enabled and self.es_counter >= self.es_patience:
                logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Training terminated.")
                break
                
        logger.info("Training complete.")

    def save_experiment_metadata(self) -> None:
        """Save training experiment run configurations, parameters, and epoch metrics history."""
        # Calculate parameters count safely
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        metadata = {
            "model_name": self.config.get("model", {}).get("name", "pneumoniaCNN"),
            "parameters": {
                "total": total_params,
                "trainable": trainable_params
            },
            "hyperparameters": {
                "optimizer": self.config.get("training", {}).get("optimizer", {}),
                "scheduler": self.config.get("training", {}).get("scheduler", {}),
                "loss": self.config.get("training", {}).get("loss", {}),
                "epochs": self.epochs,
                "early_stopping": self.config.get("training", {}).get("early_stopping", {})
            },
            "best_metric_value": self.best_metric_value,
            "metrics_history": self.metrics_history
        }
        
        metadata_path = self.checkpoint_dir / "experiment_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Experiment metadata logged successfully: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save experiment metadata to {metadata_path}: {e}")
