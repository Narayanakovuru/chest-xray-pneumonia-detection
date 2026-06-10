import sys
from pathlib import Path

# Ensure model packages are visible in system path BEFORE imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
# pyrefly: ignore [missing-import]
import torch

from src.pneumonia_detection.utils.config import load_all_configs
from src.pneumonia_detection.utils.logging import configure_logger
from src.pneumonia_detection.utils.seed import set_seed
from src.pneumonia_detection.utils.device import setup_device
from src.pneumonia_detection.data.preprocess import load_metadata, get_class_distribution
from src.pneumonia_detection.data.dataset import PneumoniaDataset
from src.pneumonia_detection.data.dataloaders import build_dataloaders
from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.models.losses import get_criterion
from src.pneumonia_detection.training.trainer import Trainer
from src.pneumonia_detection.evaluation.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def run_training_pipeline(config_dir: str = "configs") -> None:
    """Run the complete end-to-end training, validation, and testing pipeline.

    Args:
        config_dir (str): Directory containing the configuration files.
    """
    try:
        # 1. Load Configurations
        config = load_all_configs(config_dir)
        
        # 2. Setup Central Logging
        log_cfg = config["training"]["logging"]
        configure_logger(
            log_dir=log_cfg.get("dir", "logs"),
            log_file=log_cfg.get("log_file", "training.log"),
            level=log_cfg.get("level", "INFO")
        )
        
        logger.info("Initializing Training Pipeline...")
        
        # 3. Apply Reproducibility Seeds
        seed = config["training"].get("seed", 42)
        set_seed(seed)
        
        # 4. Setup Device
        device_setting = config["inference"].get("device", "auto")
        device = setup_device(device_setting)
        
        # 5. Load and Preprocess Annotations Metadata CSV
        data_cfg = config["data"]
        metadata_path = Path(data_cfg["metadata_csv"])
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found at: {metadata_path.resolve()}. "
                         "Please check the path configuration in configs/data.yaml.")
            sys.exit(1)
            
        df = load_metadata(metadata_path)
        
        # 6. Extract Dataset distributions for dynamic class weighting
        neg_count, pos_count, _ = get_class_distribution(df)
        
        # 7. Construct Datasets and DataLoaders
        img_dir = Path(data_cfg["image_dir"])
        if not img_dir.exists():
            logger.error(f"Image directory not found at: {img_dir.resolve()}. "
                         "Please check the path configuration in configs/data.yaml.")
            sys.exit(1)
            
        train_loader, val_loader, test_loader = build_dataloaders(df, img_dir, config)
        
        if not isinstance(train_loader.dataset, PneumoniaDataset):
            raise TypeError("Expected train_loader.dataset to be an instance of PneumoniaDataset")
            
        train_df = train_loader.dataset.df
        t_neg = (train_df["Target"] == 0).sum()
        t_pos = (train_df["Target"] == 1).sum()
        logger.info(f"Training split imbalance - Negatives: {t_neg} | Positives: {t_pos}")
        
        # 8. Instantiate Model
        logger.info("Instantiating pneumoniaCNN model...")
        model = pneumoniaCNN(config).to(device)
        
        # 9. Configure Loss Criterion
        criterion = get_criterion(
            config=config,
            neg_count=t_neg,
            pos_count=t_pos,
            device=device
        )
        
        # 10. Instantiate Optimizer
        opt_cfg = config["training"]["optimizer"]
        opt_type = opt_cfg.get("type", "AdamW")
        
        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(opt_cfg.get("lr", 1e-4)),
                weight_decay=float(opt_cfg.get("weight_decay", 1e-4))
            )
        elif opt_type == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(opt_cfg.get("lr", 1e-4))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
            
        # 11. Instantiate Scheduler
        sched_cfg = config["training"]["scheduler"]
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")
        epochs = config["training"].get("epochs", 50)
        
        if sched_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=float(sched_cfg.get("eta_min", 1e-6))
            )
        else:
            scheduler = None
            logger.info("No learning rate scheduler configured.")
            
        # 12. Run Trainer Fitting
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        trainer.fit()
        
        # 13. Evaluate final performance of Best Checkpoint on Test loader
        logger.info("Loading best model checkpoint for final test validation evaluation...")
        best_ckpt_path = Path(config["training"]["checkpoint"]["dir"]) / config["training"]["checkpoint"]["best_filename"]
        
        if best_ckpt_path.exists():
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device)["state_dict"])
            logger.info("Successfully loaded best model checkpoint state dict.")
            
            threshold = config["inference"].get("threshold", 0.5)
            evaluate_model(model, test_loader, criterion, device, threshold)
        else:
            logger.warning(f"Could not locate best checkpoint for test evaluation at: {best_ckpt_path}")
            
    except Exception as e:
        logger.exception(f"Unhandled exception during training pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training_pipeline()
