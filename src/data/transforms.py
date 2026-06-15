"""
MONAI Transforms
================
Data augmentation transforms for Swin-UNETR training.

Implements:
- Training transforms: spatial + intensity augmentations
- Validation transforms: minimal preprocessing only
"""

from typing import Dict, List, Tuple
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
    CropForegroundd,
    NormalizeIntensityd,
    ToTensord,
)


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    samples_per_volume: int = 4,
    config: Dict = None,
) -> Compose:
    """
    Get training transforms with data augmentation.
    
    Args:
        patch_size: Size of random patches to crop
        samples_per_volume: Number of patches to sample per volume
        config: Optional config dict with augmentation parameters
    
    Returns:
        Composed transform pipeline
    """
    # Default augmentation parameters
    if config is None:
        config = {}
    
    aug_config = config.get('augmentation', {})
    
    transforms = [
        # Ensure correct data types
        EnsureTyped(keys=["image", "label"]),
        
        # Crop to foreground (brain region)
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10,
        ),
        
        # Pad if necessary to ensure minimum size
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant",
        ),
        
        # Random crop patches (simple random sampling - no class balancing warnings)
        RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=patch_size,
            num_samples=samples_per_volume,
            random_size=False,
        ),
        
        # Spatial augmentations
        RandFlipd(
            keys=["image", "label"],
            prob=aug_config.get('rand_flip_prob', 0.5),
            spatial_axis=0,  # Left-right flip
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=aug_config.get('rand_flip_prob', 0.5),
            spatial_axis=1,
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=aug_config.get('rand_flip_prob', 0.5),
            spatial_axis=2,
        ),
        
        RandRotate90d(
            keys=["image", "label"],
            prob=0.3,
            max_k=3,
            spatial_axes=(0, 1),
        ),
        
        # Random affine (rotation + scaling)
        RandAffined(
            keys=["image", "label"],
            prob=aug_config.get('rand_scale_prob', 0.3),
            rotate_range=aug_config.get('rand_rotate_range', [0.26, 0.26, 0.26]),
            scale_range=[
                [aug_config.get('rand_scale_range', [0.9, 1.1])[0] - 1, 
                 aug_config.get('rand_scale_range', [0.9, 1.1])[1] - 1]
            ] * 3,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        
        # Intensity augmentations (image only, not label)
        RandShiftIntensityd(
            keys=["image"],
            offsets=aug_config.get('rand_shift_intensity', 0.1),
            prob=aug_config.get('intensity_prob', 0.5),
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=aug_config.get('rand_scale_intensity', 0.1),
            prob=aug_config.get('intensity_prob', 0.5),
        ),
        
        RandGaussianNoised(
            keys=["image"],
            prob=aug_config.get('noise_prob', 0.2),
            mean=0.0,
            std=aug_config.get('rand_gaussian_noise_std', 0.05),
        ),
        
        RandGaussianSmoothd(
            keys=["image"],
            prob=aug_config.get('rand_gaussian_smooth_prob', 0.1),
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        
        # Convert to PyTorch tensors
        ToTensord(keys=["image", "label"]),
    ]
    
    return Compose(transforms)


def get_val_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """
    Get validation transforms (minimal preprocessing, no augmentation).
    
    Args:
        patch_size: Size for padding/cropping
    
    Returns:
        Composed transform pipeline
    """
    transforms = [
        # Ensure correct data types
        EnsureTyped(keys=["image", "label"]),
        
        # Crop to foreground (brain region)
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10,
        ),
        
        # Pad if necessary
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant",
        ),
        
        # Convert to PyTorch tensors
        ToTensord(keys=["image", "label"]),
    ]
    
    return Compose(transforms)


def get_inference_transforms() -> Compose:
    """
    Get transforms for inference (no label required).
    
    Returns:
        Composed transform pipeline
    """
    transforms = [
        # Ensure correct data types
        EnsureTyped(keys=["image"]),
        
        # Crop to foreground
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10,
        ),
        
        # Convert to PyTorch tensor
        ToTensord(keys=["image"]),
    ]
    
    return Compose(transforms)
