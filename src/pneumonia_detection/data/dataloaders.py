import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import logging

from src.pneumonia_detection.data.dataset import PneumoniaDataset
from src.pneumonia_detection.data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

def build_dataloaders(
    df: pd.DataFrame,
    img_dir: Path,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split the metadata DataFrame and return train, validation, and test PyTorch DataLoaders.

    Ensures data leakage is prevented by instantiating separate datasets with
    dedicated train/validation augmentations and normalizations.

    Args:
        df (pd.DataFrame): The base DataFrame containing columns 'patientId' and 'Target'.
        img_dir (Path): Base directory containing PNG images.
        config (Dict[str, Any]): Full configuration dictionary containing data, split, and loader settings.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            Train, Validation, and Test dataloaders.
    """
    data_config = config["data"]
    split_config = data_config["splits"]
    loader_config = data_config["loader"]
    
    img_size = tuple(data_config["img_size"])
    seed = split_config["seed"]
    
    # Ratios
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]
    
    # Validate sum
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")
        
    logger.info("Splitting dataset into train, validation, and test splits.")
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["Target"]
    )
    
    # Second split: Val vs Test
    # Calculate proportion of val in the temp split
    val_proportion = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_proportion),
        random_state=seed,
        stratify=temp_df["Target"]
    )
    
    logger.info(f"Dataset Split Sizes - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Instantiate custom datasets with respective transform pipelines
    train_dataset = PneumoniaDataset(
        dataframe=train_df,
        img_dir=img_dir,
        transform=get_train_transforms(img_size)
    )
    
    val_dataset = PneumoniaDataset(
        dataframe=val_df,
        img_dir=img_dir,
        transform=get_val_transforms(img_size)
    )
    
    test_dataset = PneumoniaDataset(
        dataframe=test_df,
        img_dir=img_dir,
        transform=get_val_transforms(img_size)
    )
    
    # Instantiate PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
