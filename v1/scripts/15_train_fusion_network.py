#!/usr/bin/env python3
"""
15. Train Fusion Network
========================
Train the learned adaptive fusion network on pre-computed base model predictions.

This script:
1. Loads pre-computed predictions from all 3 base models
2. Trains a lightweight fusion network to combine them
3. Uses original image context for adaptive weighting
4. Outputs fused predictions with improved performance

Usage:
    python scripts/15_train_fusion_network.py \
        --data-dir fusion_data \
        --output-dir fusion_checkpoints \
        --epochs 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_network import AdaptiveFusionNetwork, create_fusion_network
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)

# MONAI imports
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


class FusionDataset(Dataset):
    """
    Dataset for fusion network training.
    
    Loads pre-computed predictions and original images from NPZ files.
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        num_samples: int = 4,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        
        # Get all NPZ files
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        print(f"[FusionDataset] Found {len(self.files)} cases")
    
    def __len__(self):
        return len(self.files) * self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples
        
        # Load NPZ file
        data = np.load(self.files[file_idx])
        
        # Stack predictions: [3, H, W, D] in order: [nnunet, swin, segresnet]
        # This order is CRITICAL for the Residual Skip Connection in the model
        predictions = [
            data['nnunet'] if 'nnunet' in data else np.zeros_like(label),
            data['swin_unetr'] if 'swin_unetr' in data else np.zeros_like(label),
            data['segresnet'] if 'segresnet' in data else np.zeros_like(label)
        ]
        
        predictions = np.stack(predictions, axis=0).astype(np.float32)
        
        # Load image: [3, H, W, D]
        image = data['image'].astype(np.float32)
        
        # Load label: [H, W, D]
        label = data['label'].astype(np.float32)
        
        # Random crop
        if self.is_train:
            predictions, image, label = self._random_crop(predictions, image, label)
        else:
            predictions, image, label = self._center_crop(predictions, image, label)
        
        return {
            'predictions': torch.from_numpy(predictions),
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label[np.newaxis, ...]),  # Add channel dim
        }
    
    def _random_crop(
        self,
        predictions: np.ndarray,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Balanced random crop to patch size (tries to hit lesion 50% of time)."""
        ph, pw, pd = self.patch_size
        _, h, w, d = predictions.shape
        
        # 50% chance to force a crop containing the lesion
        if np.any(label > 0) and np.random.random() > 0.5:
            # Find all coordinates where label > 0
            indices = np.argwhere(label > 0)
            # Pick a random lesion point as the center (approx)
            choice = indices[np.random.randint(len(indices))]
            
            # Calculate start coords, keeping within bounds
            h_start = max(0, min(h - ph, choice[0] - ph // 2))
            w_start = max(0, min(w - pw, choice[1] - pw // 2))
            d_start = max(0, min(d - pd, choice[2] - pd // 2))
        else:
            # Uniform random crop
            h_start = np.random.randint(0, max(1, h - ph + 1))
            w_start = np.random.randint(0, max(1, w - pw + 1))
            d_start = np.random.randint(0, max(1, d - pd + 1))
        
        predictions = predictions[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        image = image[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        label = label[h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        
        # Pad if needed
        predictions = self._pad_to_size(predictions, self.patch_size)
        image = self._pad_to_size(image, self.patch_size)
        label = self._pad_to_size(label, self.patch_size, is_label=True)
        
        return predictions, image, label
    
    def _center_crop(
        self,
        predictions: np.ndarray,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Center crop to patch size."""
        ph, pw, pd = self.patch_size
        _, h, w, d = predictions.shape
        
        h_start = max(0, (h - ph) // 2)
        w_start = max(0, (w - pw) // 2)
        d_start = max(0, (d - pd) // 2)
        
        predictions = predictions[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        image = image[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        label = label[h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        
        # Pad if needed
        predictions = self._pad_to_size(predictions, self.patch_size)
        image = self._pad_to_size(image, self.patch_size)
        label = self._pad_to_size(label, self.patch_size, is_label=True)
        
        return predictions, image, label
    
    def _pad_to_size(
        self,
        arr: np.ndarray,
        target_size: Tuple[int, int, int],
        is_label: bool = False,
    ) -> np.ndarray:
        """Pad array to target size."""
        if len(arr.shape) == 3:
            # Label: [H, W, D]
            h, w, d = arr.shape
            ph, pw, pd = target_size
        else:
            # Image/predictions: [C, H, W, D]
            _, h, w, d = arr.shape
            ph, pw, pd = target_size
        
        if h >= ph and w >= pw and d >= pd:
            return arr
        
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        
        if len(arr.shape) == 3:
            pad_width = [(0, pad_h), (0, pad_w), (0, pad_d)]
        else:
            pad_width = [(0, 0), (0, pad_h), (0, pad_w), (0, pad_d)]
        
        return np.pad(arr, pad_width, mode='constant', constant_values=0)


def compute_dice(pred: torch.Tensor, label: torch.Tensor) -> float:
    """Compute Dice coefficient."""
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * label).sum()
    union = pred_bin.sum() + label.sum()
    dice = (2 * intersection) / (union + 1e-8)
    return dice.item()


def train_fusion(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda",
):
    """
    Train the fusion network.
    
    Args:
        data_dir: Directory with pre-computed predictions
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    print_step("Loading data...")
    dataset = FusionDataset(data_dir, is_train=True)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset.files))
    train_files = dataset.files[:train_size]
    val_files = dataset.files[train_size:]
    
    train_dataset = FusionDataset(data_dir, is_train=True)
    train_dataset.files = train_files
    
    val_dataset = FusionDataset(data_dir, is_train=False)
    val_dataset.files = val_files
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print_info(f"Train cases: {len(train_files)}, Val cases: {len(val_files)}")
    
    # Create model
    print_step("Creating fusion network...")
    model = AdaptiveFusionNetwork(
        in_channels=8,  # 3 predictions + 3 image channels + 1 physics + 1 uncertainty
        hidden_channels=(16, 32, 16),
        out_channels=1,
        dropout_prob=0.1,
        use_residual=True,
    )
    model = model.to(device)
    
    # Loss function - combined Dice and BCE
    dice_loss = DiceCELoss(
        to_onehot_y=False,
        sigmoid=False,  # Already sigmoid in model
        squared_pred=True,
    )
    bce_loss = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    
    # Mixed precision not needed for this small network
    use_amp = False
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    
    # Training loop
    print_header("Training Fusion Network")
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_batches = 0
        
        for batch in train_loader:
            predictions = batch['predictions'].to(device)
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=False):  # Keep in float32 for stability
                output = model(predictions, image)
                loss = dice_loss(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += compute_dice(output, label)
            num_batches += 1
        
        train_loss /= num_batches
        train_dice /= num_batches
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = batch['predictions'].to(device)
                image = batch['image'].to(device)
                label = batch['label'].to(device)
                
                output = model(predictions, image)
                loss = bce_loss(output, label)
                
                val_loss += loss.item()
                val_dice += compute_dice(output, label)
                num_val_batches += 1
        
        if num_val_batches > 0:
            val_loss /= num_val_batches
            val_dice /= num_val_batches
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/dice', train_dice, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/dice', val_dice, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            model.save_checkpoint(
                output_dir / "checkpoint_best.pth",
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_dice,
            )
            print_success(f"  New best Dice: {best_dice:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            model.save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_dice,
            )
    
    # Save final model
    model.save_checkpoint(
        output_dir / "checkpoint_final.pth",
        optimizer=optimizer,
        epoch=num_epochs - 1,
        best_metric=best_dice,
    )
    
    writer.close()
    
    print_header("Training Complete!")
    print_success(f"Best Validation Dice: {best_dice:.4f}")
    print_info(f"Checkpoints saved to: {output_dir}")
    
    return {'best_dice': best_dice}


def main():
    parser = argparse.ArgumentParser(description="Train fusion network")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with pre-computed predictions')
    parser.add_argument('--output-dir', type=str, default='fusion_checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train_fusion(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
