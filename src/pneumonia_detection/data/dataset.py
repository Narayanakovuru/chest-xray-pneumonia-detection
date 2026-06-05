import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Optional, Any

class PneumoniaDataset(Dataset):
    """Custom PyTorch dataset for loading Pneumonia Detection X-Ray images and annotations."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: Path,
        transform: Optional[Any] = None
    ) -> None:
        """Initialize PneumoniaDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing columns 'patientId' and 'Target'.
            img_dir (Path): Base directory containing training/test images.
            transform (Optional[Any]): Image transformations to apply.
        """
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        """Return length of dataset.

        Returns:
            int: The total count of images.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch image and corresponding binary target label.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed X-Ray image tensor and target tensor.
        """
        img_id = self.df.iloc[idx]["patientId"]
        label = self.df.iloc[idx]["Target"]
        
        # Assemble file path using pathlib
        img_path = self.img_dir / f"{img_id}.png"
        
        if not img_path.exists():
            raise FileNotFoundError(f"X-Ray Image file not found at: {img_path}")
            
        try:
            # Open image as single-channel grayscale ("L")
            image = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error reading image {img_path}: {e}")
            
        if self.transform:
            transformed = self.transform(image)
            if isinstance(transformed, torch.Tensor):
                image_tensor = transformed
            else:
                from torchvision.transforms.functional import to_tensor
                image_tensor = to_tensor(transformed)
        else:
            from torchvision.transforms.functional import to_tensor
            image_tensor = to_tensor(image)
            
        return image_tensor, torch.tensor(label, dtype=torch.float32)
