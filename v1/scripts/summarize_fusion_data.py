#!/usr/bin/env python3
"""
Summarize generated fusion data metrics.
"""
import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def compute_dice(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0 if np.sum(gt) == 0 else 0.0
    return 2.0 * intersection / union

def main():
    data_dir = Path("fusion_data")
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        return

    npz_files = list(data_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} cases in {data_dir}")

    metrics = {
        'nnunet': [],
        'swin_unetr': [],
        'segresnet': []
    }

    for f in tqdm(npz_files, desc="Computing metrics"):
        data = np.load(f)
        if 'label' not in data:
            continue
            
        gt = data['label']
        
        if 'nnunet' in data:
            metrics['nnunet'].append(compute_dice(data['nnunet'], gt))
        if 'swin_unetr' in data:
            metrics['swin_unetr'].append(compute_dice(data['swin_unetr'], gt))
        if 'segresnet' in data:
            metrics['segresnet'].append(compute_dice(data['segresnet'], gt))

    print("\n" + "="*40)
    print("Base Model Performance Summary")
    print("="*40)
    
    for model, scores in metrics.items():
        if scores:
            print(f"{model.upper():<12}: Mean Dice = {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    print("="*40)

if __name__ == "__main__":
    main()
