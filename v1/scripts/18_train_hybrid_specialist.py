#!/usr/bin/env python3
"""
18. Train Hybrid Specialist Fusion
===================================
Final refinement script targeting the "Small Lesion Barrier".

Implementation:
1. Logit-Space Residual: Preserves SegResNet baseline as an anchor.
2. Volume-Weighted Loss: Boosts gradients for tiny strokes.
3. Physics-Aware Features: Uses DWI-ADC coupling to prune noise.
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

from src.models.fusion_network import AdaptiveFusionNetwork
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)

# MONAI imports
from monai.losses import DiceCELoss

class SpecialistDataset(Dataset):
    """Dataset with size-aware metadata for volume-weighted training."""
    
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
        print(f"[SpecialistDataset] Loaded {len(self.files)} cases")
        
        # Compute case sizes for weighting (Small < 1000 voxels)
        self.case_weights = []
        for f in tqdm(self.files, desc="Analyzing lesion sizes"):
            data = np.load(f)
            voxels = np.sum(data['label'] > 0)
            # Inverse square root weighting for volume balance
            weight = 1.0 / (np.sqrt(voxels) + 1.0)
            self.case_weights.append(weight)
        
        # Normalize weights
        self.case_weights = np.array(self.case_weights)
        self.case_weights = self.case_weights / np.mean(self.case_weights)

    def __len__(self):
        return len(self.files) * self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.num_samples
        
        data = np.load(self.files[file_idx])
        label = data['label'].astype(np.float32)
        weight = float(self.case_weights[file_idx])
        
        # Stack in CRITICAL order: [nnunet, swin, segresnet]
        predictions = np.stack([
            data.get('nnunet', np.zeros_like(label)),
            data.get('swin_unetr', np.zeros_like(label)),
            data.get('segresnet', np.zeros_like(label))
        ], axis=0).astype(np.float32)
        
        image = data['image'].astype(np.float32)
        
        # Balanced sampling logic
        if self.is_train:
            predictions, image, label = self._random_crop(predictions, image, label)
        else:
            predictions, image, label = self._center_crop(predictions, image, label)
        
        return {
            'predictions': torch.from_numpy(predictions),
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label[np.newaxis, ...]),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }

    def _random_crop(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        
        # Higher bias for hits in Specialist mode (70%)
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
        h_s, w_s, d_s = (h-ph)//2, (w-pw)//2, (d-pd)//2
        return self._pad(p[:, h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd], 
                        i[:, h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd], 
                        l[h_s:h_s+ph, w_s:w_s+pw, d_s:d_s+pd])

    def _pad(self, p, i, l):
        ph, pw, pd = self.patch_size
        _, h, w, d = p.shape
        pad = [(0, 0), (0, max(0, ph-h)), (0, max(0, pw-w)), (0, max(0, pd-d))]
        lpad = [(0, max(0, ph-h)), (0, max(0, pw-w)), (0, max(0, pd-d))]
        return np.pad(p, pad), np.pad(i, pad), np.pad(l, lpad)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print_step("Initializing Specialist Dataset...")
    dataset = SpecialistDataset(args.data_dir, is_train=True)
    split = int(0.8 * len(dataset.files))
    
    train_ds = SpecialistDataset(args.data_dir, is_train=True)
    train_ds.files = dataset.files[:split]
    val_ds = SpecialistDataset(args.data_dir, is_train=False)
    val_ds.files = dataset.files[split:]
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print_step("Creating Logit-Residual Hybrid Network...")
    model = AdaptiveFusionNetwork(in_channels=8, use_residual=True).to(device)
    
    # Focal-influenced Dice Loss for small objects
    criterion = DiceCELoss(sigmoid=False) # Manual sigmoid in model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0
    print_header("Starting Specialized Training")
    
    for epoch in range(args.epochs):
        model.train()
        t_loss, t_dice = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            p, i, l, w = batch['predictions'].to(device), batch['image'].to(device), \
                         batch['label'].to(device), batch['weight'].to(device)
            
            optimizer.zero_grad()
            out = model(p, i)
            
            # Apply volume-weighted loss
            loss = criterion(out, l)
            # Add volume penalty factor to help small cases
            weighted_loss = (loss * w.view(-1, 1, 1, 1, 1)).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            t_loss += weighted_loss.item()
            # Dice calculation
            pred_bin = (out > 0.5).float()
            t_dice += (2*(pred_bin*l).sum()/(pred_bin.sum()+l.sum()+1e-8)).item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        v_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                p, i, l = batch['predictions'].to(device), batch['image'].to(device), batch['label'].to(device)
                out = model(p, i)
                pred_bin = (out > 0.5).float()
                v_dice += (2*(pred_bin*l).sum()/(pred_bin.sum()+l.sum()+1e-8)).item()
        
        v_dice /= len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {t_loss/len(train_loader):.4f} | Val Dice: {v_dice:.4f}")
        
        if v_dice > best_dice:
            best_dice = v_dice
            model.save_checkpoint(Path(args.output_dir)/"checkpoint_best.pth", epoch=epoch, best_metric=best_dice)
            print_success(f"  New Specialist Best: {best_dice:.4f}")

    print_header(f"Final Specialist Result: {best_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='fusion_v4_specialist')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5) # Lower LR for residual logit adjustments
    train(parser.parse_args())
