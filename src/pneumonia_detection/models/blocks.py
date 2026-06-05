import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block.

    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SEBlock.

        Args:
            channels (int): The number of input channels.
            reduction (int): The reduction ratio for the bottleneck layer.
        """
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SEBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Channel-scaled output tensor of identical shape.
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze operation
        y = self.pool(x).view(batch_size, channels)
        
        # Excitation operation
        y = self.fc(y).view(batch_size, channels, 1, 1)
        
        # Scale input tensor
        return x * y
