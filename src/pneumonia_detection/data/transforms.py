from torchvision import transforms
from typing import Tuple

def get_train_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get training data augmentations composition.

    Args:
        img_size (Tuple[int, int]): Size of output images.

    Returns:
        transforms.Compose: Training transforms pipeline.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        # Preserve original augmentations
        transforms.RandomRotation(0.5),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03)
        ),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get validation and testing data transforms composition.

    Args:
        img_size (Tuple[int, int]): Size of output images.

    Returns:
        transforms.Compose: Validation/testing transforms pipeline.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
