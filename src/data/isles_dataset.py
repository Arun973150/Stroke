"""
ISLES Dataset
=============
PyTorch Dataset for loading ISLES-2022 data from nnU-Net raw NIfTI format.

This module provides:
- ISLESDataset: Dataset class loading from nnU-Net raw NIfTI files
- get_train_val_dataloaders: Factory function for train/val dataloaders
"""

import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from monai.data import list_data_collate


class ISLESDataset(Dataset):
    """
    PyTorch Dataset for ISLES-2022 stroke segmentation.
    
    Loads data from nnU-Net raw format (NIfTI files).
    
    Args:
        raw_dir: Path to nnU-Net raw data directory (with imagesTr, labelsTr)
        preprocessed_dir: Path to nnU-Net preprocessed directory (for splits_final.json)
        fold: Fold number (0-4) for cross-validation
        is_train: Whether this is training set
        transform: Optional transform to apply to data
    """
    
    def __init__(
        self,
        raw_dir: str,
        preprocessed_dir: str,
        fold: int = 0,
        is_train: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.fold = fold
        self.is_train = is_train
        self.transform = transform
        
        # Verify directories exist
        self.images_dir = self.raw_dir / "imagesTr"
        self.labels_dir = self.raw_dir / "labelsTr"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Load split information from nnU-Net preprocessed folder
        self.case_ids = self._load_split()
        
        print(f"[ISLESDataset] Loaded {len(self.case_ids)} cases for "
              f"{'train' if is_train else 'val'} fold {fold}")
    
    def _load_split(self) -> List[str]:
        """Load case IDs from nnU-Net splits file."""
        splits_file = self.preprocessed_dir / "splits_final.json"
        
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_file}. "
                "Run nnU-Net preprocessing first."
            )
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # nnU-Net format: list of dicts with 'train' and 'val' keys
        fold_split = splits[self.fold]
        
        if self.is_train:
            return fold_split['train']
        else:
            return fold_split['val']
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single case.
        
        Returns:
            Dictionary with:
            - 'image': Tensor of shape [C, H, W, D] (3 channels: DWI, ADC, FLAIR)
            - 'label': Tensor of shape [1, H, W, D] (binary segmentation)
            - 'case_id': String identifier
        """
        case_id = self.case_ids[idx]
        
        # Load image channels (DWI=0000, ADC=0001, FLAIR=0002)
        channels = []
        for ch_idx in range(3):
            img_path = self.images_dir / f"{case_id}_{ch_idx:04d}.nii.gz"
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            nii = nib.load(str(img_path))
            channel_data = nii.get_fdata().astype(np.float32)
            channels.append(channel_data)
        
        # Stack channels: [C, H, W, D]
        image = np.stack(channels, axis=0)
        
        # Normalize each channel independently (z-score)
        for c in range(image.shape[0]):
            channel = image[c]
            mask = channel > 0  # Non-background
            if mask.sum() > 0:
                mean = channel[mask].mean()
                std = channel[mask].std()
                if std > 0:
                    image[c] = (channel - mean) / std
        
        # Load segmentation label
        label_path = self.labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        label_nii = nib.load(str(label_path))
        label = label_nii.get_fdata().astype(np.float32)
        
        # Ensure label has channel dimension [1, H, W, D]
        if label.ndim == 3:
            label = label[np.newaxis, ...]
        
        # Create sample dict
        sample = {
            'image': image,
            'label': label,
            'case_id': case_id,
        }
        
        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def get_train_val_dataloaders(
    data_dir: str,  # preprocessed_dir (for splits)
    fold: int,
    train_transform: Optional[Callable],
    val_transform: Optional[Callable],
    batch_size: int = 2,
    val_batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    raw_dir: str = None,
    use_cache: bool = False,
    cache_rate: float = 0.5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to nnU-Net preprocessed data (for splits_final.json)
        fold: Cross-validation fold (0-4)
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        raw_dir: Path to nnU-Net raw data (with imagesTr, labelsTr)
        use_cache: Whether to cache data in memory
        cache_rate: Fraction of data to cache (if use_cache=True)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Infer raw_dir from data_dir if not provided
    if raw_dir is None:
        # data_dir is like /home/ubuntu/nnUNet/nnUNet_preprocessed/Dataset001_ISLES2022
        # raw_dir should be /home/ubuntu/nnUNet/nnUNet_raw/Dataset001_ISLES2022
        preprocessed_path = Path(data_dir)
        dataset_name = preprocessed_path.name  # Dataset001_ISLES2022
        raw_dir = preprocessed_path.parent.parent / "nnUNet_raw" / dataset_name
    
    # Create datasets
    train_dataset = ISLESDataset(
        raw_dir=str(raw_dir),
        preprocessed_dir=data_dir,
        fold=fold,
        is_train=True,
        transform=train_transform,
    )
    
    val_dataset = ISLESDataset(
        raw_dir=str(raw_dir),
        preprocessed_dir=data_dir,
        fold=fold,
        is_train=False,
        transform=val_transform,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=list_data_collate,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
    )
    
    return train_loader, val_loader
