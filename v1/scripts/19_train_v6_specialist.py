#!/usr/bin/env python3
"""
19. Train v6 Specialist Fusion
==============================
Improved fusion network targeting 0.73+ Dice on small lesions.

Key changes from v4:
1. Focal Tversky loss (better for small objects)
2. SE attention blocks (learn which model to trust)
3. More aggressive volume weighting for tiny lesions
4. Early stopping based on SMALL LESION dice

Usage:
    # Full v6 (Focal Tversky + SE attention)
    python scripts/19_train_v6_specialist.py --data-dir fusion_data --output-dir fusion_results_v6
    
    # Ablation A: Focal Tversky only (no SE)
    python scripts/19_train_v6_specialist.py --data-dir fusion_data --output-dir ablation_a --no-se
    
    # Ablation B: SE attention only (DiceCE loss)
    python scripts/19_train_v6_specialist.py --data-dir fusion_data --output-dir ablation_b --use-dice-ce
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
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_network_v6 import AdaptiveFusionNetworkV6, create_fusion_network_v6
from src.losses.focal_tversky import FocalTverskyLoss
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)

# MONAI imports
from monai.losses import DiceCELoss


class SpecialistDatasetV6(Dataset):
    """
    Dataset with aggressive volume weighting for tiny lesions.
    
    Key change from v4: More aggressive weighting for <500 voxel lesions.
    """
    
    # Size thresholds (in voxels)
    TINY_THRESHOLD = 500
    SMALL_THRESHOLD = 1000
    MEDIUM_THRESHOLD = 5000
    
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
        print(f"[SpecialistDatasetV6] Loaded {len(self.files)} cases")
        
        # Compute aggressive volume-based weights
        self.case_weights = []
        self.case_sizes = []
        self.size_categories = []
        
        for f in tqdm(self.files, desc="Analyzing lesion sizes"):
            data = np.load(f)
            voxels = int(np.sum(data['label'] > 0))
            self.case_sizes.append(voxels)
            
            # Categorize
            if voxels < self.TINY_THRESHOLD:
                category = 'tiny'
                weight = 5.0  # Maximum boost for tiny
            elif voxels < self.SMALL_THRESHOLD:
                category = 'small'
                weight = 3.0  # Strong boost for small
            elif voxels < self.MEDIUM_THRESHOLD:
                category = 'medium'
                weight = 1.5  # Slight boost for medium
            else:
                category = 'large'
                weight = 1.0  # Normal weight for large
            
            self.case_weights.append(weight)
            self.size_categories.append(category)
        
        self.case_weights = np.array(self.case_weights)
        
        # Print distribution
        categories, counts = np.unique(self.size_categories, return_counts=True)
        print(f"[SpecialistDatasetV6] Size distribution:")
        for cat, cnt in zip(categories, counts):
            print(f"  {cat}: {cnt} cases")
    
    def __len__(self):
        return len(self.files) * self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.num_samples
        
        data = np.load(self.files[file_idx])
        label = data['label'].astype(np.float32)
        weight = float(self.case_weights[file_idx])
        size_category = self.size_categories[file_idx]
        
        # Stack predictions: [nnunet, swin, segresnet]
        predictions = np.stack([
            data.get('nnunet', np.zeros_like(label)),
            data.get('swin_unetr', np.zeros_like(label)),
            data.get('segresnet', np.zeros_like(label))
        ], axis=0).astype(np.float32)
        
        image = data['image'].astype(np.float32)
        
        # Balanced sampling (70% foreground bias for small lesions)
        if self.is_train:
            predictions, image, label = self._random_crop(predictions, image, label)
        else:
            predictions, image, label = self._center_crop(predictions, image, label)
        
        return {
            'predictions': torch.from_numpy(predictions),
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label[np.newaxis, ...]),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'size_category': size_category,
            'voxels': self.case_sizes[file_idx],
        }
    
    def _random_crop(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        
        # Higher foreground bias (70%) for specialist training
        if np.any(l > 0) and np.random.random() > 0.3:
            indices = np.argwhere(l > 0)
            choice = indices[np.random.randint(len(indices))]
            h_start = max(0, min(h - ph, choice[0] - ph // 2))
            w_start = max(0, min(w - pw, choice[1] - pw // 2))
            d_start = max(0, min(d - pd, choice[2] - pd // 2))
        else:
            h_start = np.random.randint(0, max(1, h - ph))
            w_start = np.random.randint(0, max(1, w - pw))
            d_start = np.random.randint(0, max(1, d - pd))
        
        p = p[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        i = i[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        l = l[h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        return self._pad(p, i, l)
    
    def _center_crop(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        h_s, w_s, d_s = max(0, (h-ph)//2), max(0, (w-pw)//2), max(0, (d-pd)//2)
        return self._pad(
            p[:, h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd],
            i[:, h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd],
            l[h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd]
        )
    
    def _pad(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        if h >= ph and w >= pw and d >= pd:
            return p, i, l
        pad = [(0, 0), (0, max(0, ph-h)), (0, max(0, pw-w)), (0, max(0, pd-d))]
        lpad = [(0, max(0, ph-h)), (0, max(0, pw-w)), (0, max(0, pd-d))]
        return np.pad(p, pad), np.pad(i, pad), np.pad(l, lpad)


def compute_dice_by_category(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    categories: List[str],
) -> Dict[str, float]:
    """Compute Dice scores stratified by size category."""
    pred_bin = (predictions > 0.5).float()
    
    results = {}
    for cat in ['tiny', 'small', 'medium', 'large']:
        mask = [c == cat for c in categories]
        if not any(mask):
            continue
        
        cat_pred = pred_bin[mask]
        cat_label = labels[mask]
        
        intersection = (cat_pred * cat_label).sum()
        union = cat_pred.sum() + cat_label.sum()
        dice = (2 * intersection / (union + 1e-8)).item()
        results[cat] = dice
    
    return results


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dataset
    print_step("Initializing v6 Dataset with aggressive weighting...")
    full_dataset = SpecialistDatasetV6(args.data_dir, is_train=True)
    
    # Train/val split (80/20)
    n_files = len(full_dataset.files)
    split_idx = int(0.8 * n_files)
    
    train_ds = SpecialistDatasetV6(args.data_dir, is_train=True)
    train_ds.files = full_dataset.files[:split_idx]
    train_ds.case_weights = full_dataset.case_weights[:split_idx]
    train_ds.case_sizes = full_dataset.case_sizes[:split_idx]
    train_ds.size_categories = full_dataset.size_categories[:split_idx]
    
    val_ds = SpecialistDatasetV6(args.data_dir, is_train=False)
    val_ds.files = full_dataset.files[split_idx:]
    val_ds.case_weights = full_dataset.case_weights[split_idx:]
    val_ds.case_sizes = full_dataset.case_sizes[split_idx:]
    val_ds.size_categories = full_dataset.size_categories[split_idx:]
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print_info(f"Train: {len(train_ds.files)} cases, Val: {len(val_ds.files)} cases")
    
    # Model
    print_step("Creating v6 Fusion Network...")
    use_se = not args.no_se
    model = create_fusion_network_v6(use_se=use_se).to(device)
    
    # Loss function
    if args.use_dice_ce:
        print_info("Using DiceCE loss (ablation mode)")
        criterion = DiceCELoss(sigmoid=False, squared_pred=True)
        loss_name = "DiceCE"
    else:
        print_info(f"Using Focal Tversky loss (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})")
        criterion = FocalTverskyLoss(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            apply_sigmoid=False,  # Model already outputs sigmoid
        )
        loss_name = "FocalTversky"
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training state
    best_dice = 0
    best_small_dice = 0
    patience_counter = 0
    history = []
    
    print_header(f"Starting v6 Training (SE={use_se}, Loss={loss_name})")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            p = batch['predictions'].to(device)
            i = batch['image'].to(device)
            l = batch['label'].to(device)
            w = batch['weight'].to(device)
            
            optimizer.zero_grad()
            out = model(p, i)
            
            # Compute loss with volume weighting
            if args.use_dice_ce:
                loss = criterion(out, l)
            else:
                loss = criterion(out, l, weights=w)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Dice calculation
            pred_bin = (out > 0.5).float()
            dice = (2 * (pred_bin * l).sum() / (pred_bin.sum() + l.sum() + 1e-8)).item()
            train_dice += dice
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation
        model.eval()
        val_dice_sum = 0
        val_categories = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                p = batch['predictions'].to(device)
                i = batch['image'].to(device)
                l = batch['label'].to(device)
                
                out = model(p, i)
                
                pred_bin = (out > 0.5).float()
                dice = (2 * (pred_bin * l).sum() / (pred_bin.sum() + l.sum() + 1e-8)).item()
                val_dice_sum += dice
                
                # Collect for stratified analysis
                val_preds.append(out.cpu())
                val_labels.append(l.cpu())
                val_categories.extend(batch['size_category'])
        
        val_dice = val_dice_sum / len(val_loader)
        
        # Stratified Dice
        all_preds = torch.cat(val_preds, dim=0)
        all_labels = torch.cat(val_labels, dim=0)
        stratified = compute_dice_by_category(all_preds, all_labels, val_categories)
        
        small_dice = stratified.get('small', stratified.get('tiny', 0))
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}"
        for cat, dice in stratified.items():
            log_msg += f" | {cat}: {dice:.3f}"
        log_msg += f" | LR: {lr:.2e}"
        print(log_msg)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'stratified': stratified,
            'lr': lr,
        })
        
        # Save best based on SMALL lesion dice (our target!)
        small_combined = stratified.get('tiny', 0) * 0.5 + stratified.get('small', 0) * 0.5
        if small_combined > best_small_dice:
            best_small_dice = small_combined
            best_dice = val_dice
            patience_counter = 0
            
            model.save_checkpoint(
                Path(args.output_dir) / "checkpoint_best.pth",
                optimizer=optimizer,
                epoch=epoch,
                best_metric=val_dice,
                stratified=stratified,
            )
            print_success(f"  New best small lesion dice: {small_combined:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print_warning(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
    
    # Save training history
    with open(Path(args.output_dir) / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print_header(f"Training Complete!")
    print_info(f"Best validation dice: {best_dice:.4f}")
    print_info(f"Best small lesion dice: {best_small_dice:.4f}")
    print_info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v6 Specialist Fusion")
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True, help='Path to fusion_data directory')
    parser.add_argument('--output-dir', type=str, default='fusion_results_v6', help='Output directory')
    
    # Model
    parser.add_argument('--no-se', action='store_true', help='Disable SE attention (ablation)')
    
    # Loss
    parser.add_argument('--use-dice-ce', action='store_true', help='Use DiceCE instead of Focal Tversky (ablation)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Focal Tversky alpha (FN weight)')
    parser.add_argument('--beta', type=float, default=0.3, help='Focal Tversky beta (FP weight)')
    parser.add_argument('--gamma', type=float, default=0.75, help='Focal Tversky gamma (focal param)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    args = parser.parse_args()
    train(args)
