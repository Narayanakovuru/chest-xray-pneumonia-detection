import torch
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, Any, Union, List, Tuple
import numpy as np

from src.pneumonia_detection.models.model import pneumoniaCNN
from src.pneumonia_detection.data.transforms import get_val_transforms
from src.pneumonia_detection.utils.checkpoint import load_checkpoint
from src.pneumonia_detection.utils.device import setup_device

logger = logging.getLogger(__name__)

class Predictor:
    """Predictor class for Pneumonia Detection in Chest X-Rays.

    Loads a pre-trained model checkpoint and provides methods for single image
    and batch predictions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Predictor.

        Args:
            config (Dict[str, Any]): Full configurations containing model and inference parameters.
        """
        self.config = config
        
        # Setup device
        device_setting = config["inference"].get("device", "auto")
        self.device = setup_device(device_setting)
        
        # Load image settings
        img_size_list = config["data"].get("img_size", [224, 224])
        self.img_size = (img_size_list[0], img_size_list[1])
        
        # Set transforms
        self.transform = get_val_transforms(self.img_size)
        
        # Setup model
        self.model = pneumoniaCNN(config).to(self.device)
        
        # Reload checkpoint
        checkpoint_path = config["inference"]["checkpoint_path"]
        logger.info(f"Predictor loading weights from: {checkpoint_path}")
        load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            device=self.device
        )
        self.model.eval()
        
        self.threshold = config["inference"].get("threshold", 0.5)

    def predict_image(self, image_path: Union[str, Path]) -> Tuple[float, int, str]:
        """Run prediction on a single X-Ray image.

        Args:
            image_path (Union[str, Path]): Path to the target PNG or JPG image.

        Returns:
            Tuple[float, int, str]:
                probability score (float), predicted class index (0 or 1), and class label.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")
            
        try:
            # Load and convert to single-channel grayscale
            image = Image.open(path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {e}")
            
        # Apply validation/testing transformations
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor).squeeze(1)
            prob = torch.sigmoid(logits).item()
            
        pred_class = 1 if prob > self.threshold else 0
        label = "PNEUMONIA" if pred_class == 1 else "NORMAL"
        
        logger.debug(f"Image: {path.name} | Prob: {prob:.4f} | Class: {pred_class} ({label})")
        return prob, pred_class, label

    def predict_batch(self, image_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """Run predictions on all images located within a target folder.

        Args:
            image_dir (Union[str, Path]): Path to the folder containing image files.

        Returns:
            List[Dict[str, Any]]: List of dictionary structures containing file name,
                                  predicted class, confidence probability, and string labels.
        """
        dir_path = Path(image_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Image directory not found at: {dir_path}")
            
        supported_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
        image_files = [p for p in dir_path.iterdir() if p.suffix in supported_exts]
        
        if len(image_files) == 0:
            logger.warning(f"No valid image files found in {dir_path}")
            return []
            
        logger.info(f"Predicting batch of {len(image_files)} images from {dir_path.name}")
        results = []
        
        for p in image_files:
            try:
                prob, pred_class, label = self.predict_image(p)
                results.append({
                    "file_path": str(p),
                    "file_name": p.name,
                    "probability": prob,
                    "class_idx": pred_class,
                    "label": label
                })
            except Exception as e:
                logger.error(f"Failed to predict image {p.name}: {e}")
                
        return results
