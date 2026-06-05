import pytest
import torch
from pathlib import Path

from src.pneumonia_detection.data.preprocess import load_metadata
from src.pneumonia_detection.data.dataloaders import build_dataloaders
from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.models.losses import get_criterion
from src.pneumonia_detection.training.trainer import Trainer
from src.pneumonia_detection.utils.device import setup_device

def test_trainer_fit_loop(dummy_config):
    """Test trainer initialization and execution of fit methods for 1 epoch."""
    device = setup_device("cpu")
    
    # 1. Load Data
    csv_path = Path(dummy_config["data"]["metadata_csv"])
    img_dir = Path(dummy_config["data"]["image_dir"])
    df = load_metadata(csv_path)
    train_loader, val_loader, _ = build_dataloaders(df, img_dir, dummy_config)
    
    # 2. Instantiate Model
    model = pneumoniaCNN(dummy_config).to(device)
    
    # Verify weights update before training
    initial_weights = model.stem[0].weight.clone()
    
    # 3. Instantiate Loss
    criterion = get_criterion(dummy_config, neg_count=10, pos_count=10, device=device)
    
    # 4. Instantiate Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    
    # Override epochs count to 1 for validation run speed
    dummy_config["training"]["epochs"] = 1
    
    # 5. Trainer fit
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        config=dummy_config
    )
    
    # Execute training
    trainer.fit()
    
    # Assert weights updated
    updated_weights = model.stem[0].weight
    assert not torch.equal(initial_weights, updated_weights)
    
    # Assert best checkpoint file was saved
    best_ckpt = Path(dummy_config["training"]["checkpoint"]["dir"]) / dummy_config["training"]["checkpoint"]["best_filename"]
    last_ckpt = Path(dummy_config["training"]["checkpoint"]["dir"]) / dummy_config["training"]["checkpoint"]["last_filename"]
    metadata_json = Path(dummy_config["training"]["checkpoint"]["dir"]) / "experiment_metadata.json"
    
    assert last_ckpt.exists()
    assert best_ckpt.exists()
    assert metadata_json.exists()
