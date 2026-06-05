import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.pneumonia_detection.models.blocks import SEBlock

class pneumoniaCNN(nn.Module):
    """Custom Deep Convolutional Network for Pneumonia Detection in Chest X-Rays.

    Features:
        - Input Stem with kernel 7x7 and 3x3 convolutions.
        - Four residual blocks with shortcut identity projection.
        - Squeeze-and-Excitation (SE) channel recalibration inside blocks.
        - Global Average Pooling (GAP).
        - Dropout-regularized Dense Classifier.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize pneumoniaCNN.

        Args:
            config (Optional[Dict[str, Any]]): Config dictionary. If None, default
                                               parameters are loaded.
        """
        super(pneumoniaCNN, self).__init__()
        
        # Load hyperparameters from config or default values
        model_config = config.get("model", {}) if config is not None else {}
        
        in_channels = model_config.get("in_channels", 1)
        
        se_config = model_config.get("se", {}) or {}
        se_reduction = se_config.get("reduction", 16)
        
        stem_config = model_config.get("stem", {}) or {}
        stem_out = stem_config.get("out_channels", 64)
        
        b1_config = model_config.get("block1", {}) or {}
        c1 = b1_config.get("out_channels", 128)
        
        b2_config = model_config.get("block2", {}) or {}
        c2 = b2_config.get("out_channels", 256)
        
        b3_config = model_config.get("block3", {}) or {}
        c3 = b3_config.get("out_channels", 512)
        
        b4_config = model_config.get("block4", {}) or {}
        c4 = b4_config.get("out_channels", 1024)
        
        cls_config = model_config.get("classifier", {}) or {}
        drop_1 = cls_config.get("dropout_1", 0.4)
        drop_2 = cls_config.get("dropout_2", 0.3)
        hidden_dim = cls_config.get("hidden_dim", 256)
        out_features = cls_config.get("out_features", 1)

        # ── STEM BLOCK ───────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(stem_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_out, stem_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_out),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── RESIDUAL BLOCK 1 (64 -> 128 channels) ─────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(stem_out, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1)
        )
        self.block1_se = SEBlock(c1, reduction=se_reduction)
        self.block1_skip = nn.Sequential(
            nn.Conv2d(stem_out, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )

        # ── RESIDUAL BLOCK 2 (128 -> 256 channels) ────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2)
        )
        self.block2_se = SEBlock(c2, reduction=se_reduction)
        self.block2_skip = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.BatchNorm2d(c2)
        )

        # ── RESIDUAL BLOCK 3 (256 -> 512 channels) ────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3)
        )
        self.block3_se = SEBlock(c3, reduction=se_reduction)
        self.block3_skip = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=1),
            nn.BatchNorm2d(c3)
        )

        # ── RESIDUAL BLOCK 4 (512 -> 1024 channels) ───────────────────────
        self.block4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4)
        )
        self.block4_se = SEBlock(c4, reduction=se_reduction)
        self.block4_skip = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=1),
            nn.BatchNorm2d(c4)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(drop_1),
            nn.Linear(c4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_2),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 224, 224).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 1).
        """
        # Stem
        x = self.stem(x)

        # Block 1
        identity = self.block1_skip(x)
        out = self.block1(x)
        out = self.block1_se(out)
        x = F.silu(out + identity)

        # Block 2
        identity = self.block2_skip(x)
        out = self.block2(x)
        out = self.block2_se(out)
        x = F.silu(out + identity)

        # Block 3
        identity = self.block3_skip(x)
        out = self.block3(x)
        out = self.block3_se(out)
        x = F.silu(out + identity)

        # Block 4
        identity = self.block4_skip(x)
        out = self.block4(x)
        out = self.block4_se(out)
        x = F.silu(out + identity)

        # Pooling & Flatten
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        # Classifier
        return self.classifier(x)
