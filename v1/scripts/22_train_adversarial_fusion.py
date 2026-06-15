#!/usr/bin/env python3
"""
22. Train Adversarial Hard-Case Fusion (v5)
===========================================
The "Perfectionist" Trainer. 

Key Upgrades:
1. Error-Based Sampling: Identifies cases where SegResNet failed and trains on them 5x more.
2. Tversky Loss: Optimized for high recall on tiny structures.
3. Logit-Residual: Refines the v4 Specialist brain.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

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

class AdversarialDataset(Dataset):
    """Dataset that prioritizes samples the model currently gets WRONG."""
    
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
        print(f"[Adversarial] Analyzing {len(self.files)} cases for difficulty...")
        
        # Calculate Error-Based Weights
        self.case_weights = []
        for f in tqdm(self.files, desc="Finding Hard Cases"):
            data = np.load(f)
            label = data['label']
            segresnet = data.get('segresnet', np.zeros_like(label))
            
            # Binary Dice for the 'Expert' (SegResNet)
            p_bin = (segresnet > 0.5).astype(np.float32)
            l_bin = (label > 0).astype(np.float32)
            
            intersection = np.sum(p_bin * l_bin)
            union = np.sum(p_bin) + np.sum(l_bin)
            dice = (2 * intersection) / (union + 1e-8)
            
            # Adversarial Weight: Higher weight for LOW dice (failed cases)
            # We also boost Small cases significantly
            size_factor = 1.0 if np.sum(l_bin) > 1000 else 5.0
            error_factor = 1.0 - dice # Higher for harder cases
            
            self.case_weights.append(error_factor * size_factor)
            
        self.case_weights = np.array(self.case_weights)
        # Normalize so average weight is 1.0
        self.case_weights = self.case_weights / np.mean(self.case_weights)
        print_success(f"Adversarial weights assigned (Max: {np.max(self.case_weights):.2f})")

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
            'weight': torch.tensor(weight, dtype=torch.float32)
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
        return self._pad(p[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd],
                        i[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd],
                        l[h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd])

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

class TverskyLoss(nn.Module):
    """Tversky Loss to optimize Recall (Alpha=0.7) for small lesions."""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        tversky = (tp + 1e-6) / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        return 1 - tversky

def train_v5(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print_step("Initializing Adversarial Dataset...")
    dataset = AdversarialDataset(args.data_dir, is_train=True)
    split = int(0.8 * len(dataset.files))
    
    train_ds = AdversarialDataset(args.data_dir, is_train=True)
    train_ds.files = dataset.files[:split]
    val_ds = AdversarialDataset(args.data_dir, is_train=False)
    val_ds.files = dataset.files[split:]
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = AdaptiveFusionNetwork(in_channels=8, use_residual=True).to(device)
    # Start with v4 weights for finishing school
    if args.resume_from:
        print_info(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    criterion = TverskyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    best_dice = 0
    print_header("Training v5 Perfectionist (Adversarial)")
    
    for epoch in range(args.epochs):
        model.train()
        t_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            p, i, l, w = batch['predictions'].to(device), batch['image'].to(device), \
                         batch['label'].to(device), batch['weight'].to(device)
            
            optimizer.zero_grad()
            out = model(p, i)
            loss = (criterion(out, l) * w.mean()) # Case-weighted loss
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        # Eval
        model.eval()
        v_dice = 0
        with torch.no_grad():
            for b in val_loader:
                out = model(b['predictions'].to(device), b['image'].to(device))
                pb = (out > 0.5).float()
                lb = b['label'].to(device)
                v_dice += (2*(pb*lb).sum()/(pb.sum()+lb.sum()+1e-8)).item()
        
        v_dice /= len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {t_loss/len(train_loader):.4f} | Val Dice: {v_dice:.4f}")
        
        if v_dice > best_dice:
            best_dice = v_dice
            model.save_checkpoint(Path(args.output_dir)/"checkpoint_best.pth", epoch=epoch, best_metric=best_dice)
            print_success(f"  v5 Best: {best_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='fusion_v5_perfectionist')
    parser.add_argument('--epochs', type=int, default=50) # Short refinement
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5) # Very low for fine-tuning
    parser.add_argument('--resume-from', type=str, help='v4 checkpoint path')
    train_v5(parser.parse_args())
