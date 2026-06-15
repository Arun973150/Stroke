#!/usr/bin/env python3
"""
19. Train v7 Balanced Fusion
============================
Balanced approach: SE attention + DiceCE + Boundary Loss + EMA

Goals:
- Improve tiny lesions without overfitting
- Maintain or improve all other categories
- Beat v4 on overall dice

Key features:
1. SE attention (proven to help tiny)
2. Combined Dice + Boundary loss (helps all sizes, especially edges)
3. EMA weights (smooths training, better generalization)
4. Gentle volume weighting (log-based, not aggressive)
5. Early stopping on OVERALL dice (not just small)

Usage:
    python scripts/19_train_v7_balanced.py \
        --data-dir fusion_data \
        --output-dir fusion_results_v7
"""

import os
import sys
import argparse
import json
import copy
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
from src.losses.focal_tversky import CombinedBoundaryDiceLoss
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


class ExponentialMovingAverage:
    """
    Maintains EMA of model parameters for better generalization.
    
    The EMA model typically generalizes better than the final trained model.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class BalancedDataset(Dataset):
    """
    Dataset with gentle log-based volume weighting.
    
    Uses smooth weighting curve instead of aggressive step function.
    """
    
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
        
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        print(f"[BalancedDataset] Loaded {len(self.files)} cases")
        
        # Gentle log-based weighting
        self.case_weights = []
        self.case_sizes = []
        self.size_categories = []
        
        for f in tqdm(self.files, desc="Analyzing lesion sizes"):
            data = np.load(f)
            voxels = int(np.sum(data['label'] > 0))
            self.case_sizes.append(voxels)
            
            # Gentle log-based weight: ranges from ~2x (tiny) to ~1x (large)
            weight = 1.0 + 2.0 / (np.log(voxels + 10) + 1)
            self.case_weights.append(weight)
            
            # Categorize
            if voxels < self.TINY_THRESHOLD:
                self.size_categories.append('tiny')
            elif voxels < self.SMALL_THRESHOLD:
                self.size_categories.append('small')
            elif voxels < self.MEDIUM_THRESHOLD:
                self.size_categories.append('medium')
            else:
                self.size_categories.append('large')
        
        self.case_weights = np.array(self.case_weights)
        
        # Print distribution
        print(f"[BalancedDataset] Weight range: {self.case_weights.min():.2f} - {self.case_weights.max():.2f}")
        categories, counts = np.unique(self.size_categories, return_counts=True)
        for cat, cnt in zip(categories, counts):
            print(f"  {cat}: {cnt} cases")
    
    def __len__(self):
        return len(self.files) * self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.num_samples
        
        data = np.load(self.files[file_idx])
        label = data['label'].astype(np.float32)
        weight = float(self.case_weights[file_idx])
        
        predictions = np.stack([
            data.get('nnunet', np.zeros_like(label)),
            data.get('swin_unetr', np.zeros_like(label)),
            data.get('segresnet', np.zeros_like(label))
        ], axis=0).astype(np.float32)
        
        image = data['image'].astype(np.float32)
        
        if self.is_train:
            predictions, image, label = self._random_crop(predictions, image, label)
        else:
            predictions, image, label = self._center_crop(predictions, image, label)
        
        return {
            'predictions': torch.from_numpy(predictions),
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label[np.newaxis, ...]),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'size_category': self.size_categories[file_idx],
        }
    
    def _random_crop(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        
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


def compute_stratified_dice(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    categories: List[str],
) -> Dict[str, float]:
    """Compute Dice stratified by size category."""
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
    print_step("Initializing Balanced Dataset...")
    full_dataset = BalancedDataset(args.data_dir, is_train=True)
    
    n_files = len(full_dataset.files)
    split_idx = int(0.8 * n_files)
    
    train_ds = BalancedDataset(args.data_dir, is_train=True)
    train_ds.files = full_dataset.files[:split_idx]
    train_ds.case_weights = full_dataset.case_weights[:split_idx]
    train_ds.case_sizes = full_dataset.case_sizes[:split_idx]
    train_ds.size_categories = full_dataset.size_categories[:split_idx]
    
    val_ds = BalancedDataset(args.data_dir, is_train=False)
    val_ds.files = full_dataset.files[split_idx:]
    val_ds.case_weights = full_dataset.case_weights[split_idx:]
    val_ds.case_sizes = full_dataset.case_sizes[split_idx:]
    val_ds.size_categories = full_dataset.size_categories[split_idx:]
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print_info(f"Train: {len(train_ds.files)} cases, Val: {len(val_ds.files)} cases")
    
    # Model with SE attention
    print_step("Creating v7 Fusion Network (SE + Boundary + EMA)...")
    model = create_fusion_network_v6(use_se=True).to(device)
    
    # EMA
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    print_info(f"EMA enabled with decay={args.ema_decay}")
    
    # Combined Dice + Boundary loss
    criterion = CombinedBoundaryDiceLoss(
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
    )
    print_info(f"Loss: {args.dice_weight:.1f}*Dice + {args.boundary_weight:.1f}*Boundary")
    
    # Optimizer with cosine annealing + warm restarts
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )
    
    # Training state
    best_overall_dice = 0
    patience_counter = 0
    history = []
    
    print_header("Starting v7 Balanced Training")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            p = batch['predictions'].to(device)
            i = batch['image'].to(device)
            l = batch['label'].to(device)
            w = batch['weight'].to(device)
            
            optimizer.zero_grad()
            out = model(p, i)
            
            loss = criterion(out, l, sample_weights=w)
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation with EMA model
        ema.apply_shadow(model)
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
                
                val_preds.append(out.cpu())
                val_labels.append(l.cpu())
                val_categories.extend(batch['size_category'])
        
        val_dice = val_dice_sum / len(val_loader)
        
        # Stratified metrics
        all_preds = torch.cat(val_preds, dim=0)
        all_labels = torch.cat(val_labels, dim=0)
        stratified = compute_stratified_dice(all_preds, all_labels, val_categories)
        
        # Restore original weights for next training step
        ema.restore(model)
        
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
            'val_dice': val_dice,
            'stratified': stratified,
            'lr': lr,
        })
        
        # Save best based on OVERALL dice (for generalization)
        if val_dice > best_overall_dice:
            best_overall_dice = val_dice
            patience_counter = 0
            
            # Save EMA weights (better generalization)
            ema.apply_shadow(model)
            model.save_checkpoint(
                Path(args.output_dir) / "checkpoint_best.pth",
                optimizer=optimizer,
                epoch=epoch,
                best_metric=val_dice,
                stratified=stratified,
            )
            ema.restore(model)
            
            print_success(f"  New best overall dice: {val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print_warning(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save history
    with open(Path(args.output_dir) / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print_header("Training Complete!")
    print_info(f"Best overall dice: {best_overall_dice:.4f}")
    print_info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v7 Balanced Fusion")
    
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='fusion_results_v7')
    
    # Loss weights
    parser.add_argument('--dice-weight', type=float, default=0.7)
    parser.add_argument('--boundary-weight', type=float, default=0.3)
    
    # EMA
    parser.add_argument('--ema-decay', type=float, default=0.999)
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()
    train(args)
