#!/usr/bin/env python3
"""
24. Train v8 Tiny-Focus Fusion
==============================
Smaller patch training (48³) specifically designed to improve tiny lesion detection.

Key Changes from v6:
1. 48³ patches instead of 96³ (4x more patches per volume)
2. Extreme oversampling of tiny lesion cases (10x weight)
3. 90% lesion-centered sampling (vs 70% in v6)
4. More training iterations per epoch

Why This Helps Tiny Lesions:
- Tiny lesions fill more of the 48³ patch (better signal-to-noise)
- More patches = more training examples from tiny lesion cases
- Extreme oversampling forces model to learn tiny patterns

Usage:
    python scripts/24_train_v8_tiny_focus.py \
        --data-dir fusion_data \
        --output-dir fusion_results_v8_tiny
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_network_v6 import AdaptiveFusionNetworkV6, create_fusion_network_v6

try:
    from src.utils import print_header, print_step, print_success, print_warning, print_info
except ImportError:
    def print_header(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")
    def print_step(a, b, msg): print(f"[{a}/{b}] {msg}")
    def print_success(msg): print(f"[OK] {msg}")
    def print_warning(msg): print(f"[WARN] {msg}")
    def print_info(msg): print(f"[INFO] {msg}")


class TinyFocusDataset(Dataset):
    """
    Dataset with 48³ patches and extreme tiny lesion oversampling.
    """
    
    TINY_THRESHOLD = 500
    SMALL_THRESHOLD = 1000
    MEDIUM_THRESHOLD = 5000
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (48, 48, 48),  # Smaller patches!
        num_samples: int = 8,  # More samples per volume
        is_train: bool = True,
        lesion_center_prob: float = 0.9,  # 90% lesion-centered
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        self.lesion_center_prob = lesion_center_prob
        
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        print(f"[TinyFocusDataset] Loaded {len(self.files)} cases")
        print(f"[TinyFocusDataset] Patch size: {patch_size}")
        print(f"[TinyFocusDataset] Samples per volume: {num_samples}")
        
        # Extreme weighting for tiny lesions
        self.case_weights = []
        self.case_sizes = []
        self.size_categories = []
        
        for f in tqdm(self.files, desc="Analyzing lesion sizes"):
            data = np.load(f)
            voxels = int(np.sum(data['label'] > 0))
            self.case_sizes.append(voxels)
            
            # EXTREME weighting for tiny lesions
            if voxels < self.TINY_THRESHOLD:
                category = 'tiny'
                weight = 10.0  # 10x weight for tiny!
            elif voxels < self.SMALL_THRESHOLD:
                category = 'small'
                weight = 5.0   # 5x for small
            elif voxels < self.MEDIUM_THRESHOLD:
                category = 'medium'
                weight = 2.0   # 2x for medium
            else:
                category = 'large'
                weight = 1.0
            
            self.case_weights.append(weight)
            self.size_categories.append(category)
        
        self.case_weights = np.array(self.case_weights)
        
        # Print distribution
        print(f"[TinyFocusDataset] Weight range: {self.case_weights.min():.1f} - {self.case_weights.max():.1f}")
        categories, counts = np.unique(self.size_categories, return_counts=True)
        for cat, cnt in zip(categories, counts):
            avg_weight = np.mean([w for w, c in zip(self.case_weights, self.size_categories) if c == cat])
            print(f"  {cat}: {cnt} cases (weight: {avg_weight:.1f}x)")
    
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
            predictions, image, label = self._smart_crop(predictions, image, label)
        else:
            predictions, image, label = self._center_crop(predictions, image, label)
        
        return {
            'predictions': torch.from_numpy(predictions),
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label[np.newaxis, ...]),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'size_category': self.size_categories[file_idx],
        }
    
    def _smart_crop(self, p, i, l):
        """Lesion-centered cropping with high probability."""
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        
        # 90% chance to center on lesion
        if np.any(l > 0) and np.random.random() < self.lesion_center_prob:
            indices = np.argwhere(l > 0)
            choice = indices[np.random.randint(len(indices))]
            
            # Add small random offset so lesion isn't always exactly centered
            offset = np.random.randint(-ph//4, ph//4+1, size=3)
            
            h_start = max(0, min(h - ph, choice[0] - ph // 2 + offset[0]))
            w_start = max(0, min(w - pw, choice[1] - pw // 2 + offset[1]))
            d_start = max(0, min(d - pd, choice[2] - pd // 2 + offset[2]))
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
        h_s = max(0, (h - ph) // 2)
        w_s = max(0, (w - pw) // 2)
        d_s = max(0, (d - pd) // 2)
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


class DiceLoss(nn.Module):
    """Simple Dice loss."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target, weights=None):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        loss = 1 - dice
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


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
    
    # Dataset with 48³ patches
    print_step(1, 5, "Initializing Tiny-Focus Dataset (48³ patches)...")
    full_dataset = TinyFocusDataset(
        args.data_dir, 
        patch_size=(48, 48, 48),
        num_samples=8,  # More samples per volume
        is_train=True
    )
    
    n_files = len(full_dataset.files)
    split_idx = int(0.8 * n_files)
    
    train_ds = TinyFocusDataset(args.data_dir, patch_size=(48, 48, 48), num_samples=8, is_train=True)
    train_ds.files = full_dataset.files[:split_idx]
    train_ds.case_weights = full_dataset.case_weights[:split_idx]
    train_ds.case_sizes = full_dataset.case_sizes[:split_idx]
    train_ds.size_categories = full_dataset.size_categories[:split_idx]
    
    val_ds = TinyFocusDataset(args.data_dir, patch_size=(48, 48, 48), num_samples=4, is_train=False)
    val_ds.files = full_dataset.files[split_idx:]
    val_ds.case_weights = full_dataset.case_weights[split_idx:]
    val_ds.case_sizes = full_dataset.case_sizes[split_idx:]
    val_ds.size_categories = full_dataset.size_categories[split_idx:]
    
    # Weighted sampler to oversample tiny lesion cases
    sample_weights = []
    for i in range(len(train_ds)):
        file_idx = i // train_ds.num_samples
        sample_weights.append(train_ds.case_weights[file_idx])
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=sampler,  # Weighted sampling!
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print_info(f"Train: {len(train_ds.files)} cases ({len(train_ds)} patches)")
    print_info(f"Val: {len(val_ds.files)} cases ({len(val_ds)} patches)")
    
    # Model with SE attention
    print_step(2, 5, "Creating v8 Tiny-Focus Network...")
    model = create_fusion_network_v6(use_se=True).to(device)
    
    # Dice loss (proven to work well in v6)
    criterion = DiceLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    # Training state
    best_tiny_dice = 0
    best_overall_dice = 0
    patience_counter = 0
    history = []
    
    print_header("Starting v8 Tiny-Focus Training")
    print_info(f"Patch size: 48³ (vs 96³ in v6)")
    print_info(f"Tiny weight: 10x (vs 5x in v6)")
    print_info(f"Lesion-centered: 90% (vs 70% in v6)")
    print()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            p = batch['predictions'].to(device)
            i = batch['image'].to(device)
            l = batch['label'].to(device)
            w = batch['weight'].to(device)
            
            optimizer.zero_grad()
            out = model(p, i)
            
            loss = criterion(out, l, weights=w)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        train_loss /= len(train_loader)
        
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
                
                val_preds.append(out.cpu())
                val_labels.append(l.cpu())
                val_categories.extend(batch['size_category'])
        
        val_dice = val_dice_sum / len(val_loader)
        
        # Stratified metrics
        all_preds = torch.cat(val_preds, dim=0)
        all_labels = torch.cat(val_labels, dim=0)
        stratified = compute_stratified_dice(all_preds, all_labels, val_categories)
        
        tiny_dice = stratified.get('tiny', 0)
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}"
        for cat in ['tiny', 'small', 'medium', 'large']:
            if cat in stratified:
                log_msg += f" | {cat}: {stratified[cat]:.3f}"
        log_msg += f" | LR: {lr:.2e}"
        print(log_msg)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_dice': val_dice,
            'tiny_dice': tiny_dice,
            'stratified': stratified,
            'lr': lr,
        })
        
        # Save best based on TINY dice (our target!)
        improved = False
        if tiny_dice > best_tiny_dice:
            best_tiny_dice = tiny_dice
            improved = True
        
        if val_dice > best_overall_dice:
            best_overall_dice = val_dice
            improved = True
        
        if improved:
            patience_counter = 0
            model.save_checkpoint(
                Path(args.output_dir) / "checkpoint_best.pth",
                optimizer=optimizer,
                epoch=epoch,
                best_metric=val_dice,
                tiny_dice=tiny_dice,
                stratified=stratified,
            )
            print_success(f"  New best! Tiny: {tiny_dice:.4f}, Overall: {val_dice:.4f}")
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
    print_info(f"Best tiny dice: {best_tiny_dice:.4f}")
    print_info(f"Best overall dice: {best_overall_dice:.4f}")
    print_info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v8 Tiny-Focus Fusion")
    
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='fusion_results_v8_tiny')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)  # Larger batch with smaller patches
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=40)
    
    args = parser.parse_args()
    train(args)
