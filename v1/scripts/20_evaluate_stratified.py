#!/usr/bin/env python3
"""
20. Stratified Evaluation
=========================
Comprehensive evaluation with metrics stratified by lesion size.

This script:
1. Evaluates model on held-out test set
2. Computes per-size-category metrics (tiny/small/medium/large)
3. Compares v4, v6, and ablation models
4. Generates comparison tables and visualizations

Usage:
    # Evaluate single model
    python scripts/20_evaluate_stratified.py \
        --data-dir fusion_data \
        --checkpoint fusion_results_v6/checkpoint_best.pth \
        --output-dir results/v6_evaluation
    
    # Compare multiple models
    python scripts/20_evaluate_stratified.py \
        --data-dir fusion_data \
        --checkpoints fusion_results_v4/checkpoint_best.pth fusion_results_v6/checkpoint_best.pth \
        --names v4 v6 \
        --output-dir results/v4_vs_v6
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_network import AdaptiveFusionNetwork
from src.models.fusion_network_v6 import AdaptiveFusionNetworkV6
from src.utils import (
    print_header, print_step, print_success, print_warning, print_info
)


# Size thresholds (in voxels, matching training)
TINY_THRESHOLD = 500
SMALL_THRESHOLD = 1000
MEDIUM_THRESHOLD = 5000


def categorize_lesion(voxels: int) -> str:
    """Categorize lesion by volume."""
    if voxels < TINY_THRESHOLD:
        return 'tiny'
    elif voxels < SMALL_THRESHOLD:
        return 'small'
    elif voxels < MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'large'


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive segmentation metrics."""
    pred_bin = (pred > 0.5).astype(bool).flatten()
    label_bin = label.astype(bool).flatten()
    
    tp = np.sum(pred_bin & label_bin)
    fp = np.sum(pred_bin & ~label_bin)
    tn = np.sum(~pred_bin & ~label_bin)
    fn = np.sum(~pred_bin & label_bin)
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # Recall
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint (auto-detect v4 or v6)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('model_config', {})
    
    # Check if v6 (has 'use_se' or 'version' key)
    if config.get('version') == 'v6' or 'use_se' in config:
        config.pop('version', None)
        model = AdaptiveFusionNetworkV6(**config)
        version = 'v6'
    else:
        model = AdaptiveFusionNetwork(**config)
        version = 'v4'
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print_info(f"Loaded {version} model from {checkpoint_path}")
    return model, version


def evaluate_model(
    model: torch.nn.Module,
    data_dir: Path,
    device: torch.device,
    split: str = 'val',
) -> pd.DataFrame:
    """
    Evaluate model on all cases and return per-case metrics.
    
    Args:
        model: Trained fusion model
        data_dir: Path to fusion_data directory
        device: Torch device
        split: 'val' or 'test' (determines which files to use)
        
    Returns:
        DataFrame with per-case metrics
    """
    npz_files = sorted(list(data_dir.glob("*.npz")))
    
    # For validation, use last 20% of files
    if split == 'val':
        split_idx = int(0.8 * len(npz_files))
        npz_files = npz_files[split_idx:]
    
    print_info(f"Evaluating on {len(npz_files)} cases ({split} split)")
    
    results = []
    
    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc="Evaluating"):
            data = np.load(npz_file)
            
            # Load data
            predictions = np.stack([
                data.get('nnunet', np.zeros_like(data['label'])),
                data.get('swin_unetr', np.zeros_like(data['label'])),
                data.get('segresnet', np.zeros_like(data['label']))
            ], axis=0).astype(np.float32)
            
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.float32)
            
            # To tensors
            pred_tensor = torch.from_numpy(predictions).unsqueeze(0).to(device)
            img_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            
            # Inference
            output = model(pred_tensor, img_tensor)
            output_np = output[0, 0].cpu().numpy()
            
            # Compute metrics
            metrics = compute_metrics(output_np, label)
            
            # Add case info
            case_id = npz_file.stem
            label_voxels = int(np.sum(label > 0))
            category = categorize_lesion(label_voxels)
            
            results.append({
                'case_id': case_id,
                'label_voxels': label_voxels,
                'category': category,
                **metrics,
            })
    
    return pd.DataFrame(results)


def compute_stratified_summary(df: pd.DataFrame) -> Dict:
    """Compute summary statistics stratified by size category."""
    summary = {
        'overall': {
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'mean_sensitivity': df['sensitivity'].mean(),
            'mean_precision': df['precision'].mean(),
            'n_cases': len(df),
        }
    }
    
    for category in ['tiny', 'small', 'medium', 'large']:
        cat_df = df[df['category'] == category]
        if len(cat_df) == 0:
            continue
        
        summary[category] = {
            'mean_dice': cat_df['dice'].mean(),
            'std_dice': cat_df['dice'].std(),
            'mean_sensitivity': cat_df['sensitivity'].mean(),
            'mean_precision': cat_df['precision'].mean(),
            'n_cases': len(cat_df),
        }
    
    return summary


def print_comparison_table(summaries: Dict[str, Dict], model_names: List[str]):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("STRATIFIED COMPARISON")
    print("=" * 80)
    
    # Header
    header = f"{'Category':<12}"
    for name in model_names:
        header += f" | {name:^18}"
    print(header)
    print("-" * 80)
    
    # Rows
    for category in ['tiny', 'small', 'medium', 'large', 'overall']:
        row = f"{category:<12}"
        for name in model_names:
            if category in summaries[name]:
                dice = summaries[name][category]['mean_dice']
                n = summaries[name][category]['n_cases']
                row += f" | {dice:.4f} (n={n:3d})   "
            else:
                row += f" |        N/A        "
        print(row)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Stratified Evaluation")
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to fusion_data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint to evaluate')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Multiple checkpoints to compare')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Names for each checkpoint (for comparison)')
    parser.add_argument('--output-dir', type=str, default='results/stratified_eval',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'all'],
                        help='Data split to evaluate')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    
    # Determine checkpoints to evaluate
    if args.checkpoints:
        checkpoints = args.checkpoints
        names = args.names if args.names else [f"model_{i}" for i in range(len(checkpoints))]
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
        names = ['model']
    else:
        print_error("Please provide --checkpoint or --checkpoints")
        return
    
    # Evaluate each model
    all_summaries = {}
    all_results = {}
    
    for ckpt, name in zip(checkpoints, names):
        print_header(f"Evaluating: {name}")
        
        model, version = load_model(ckpt, device)
        df = evaluate_model(model, data_dir, device, args.split)
        summary = compute_stratified_summary(df)
        
        all_summaries[name] = summary
        all_results[name] = df
        
        # Save per-case results
        df.to_csv(output_dir / f"{name}_per_case.csv", index=False)
        
        # Print summary
        print(f"\n{name} Summary:")
        print(f"  Overall Dice: {summary['overall']['mean_dice']:.4f} ± {summary['overall']['std_dice']:.4f}")
        for cat in ['tiny', 'small', 'medium', 'large']:
            if cat in summary:
                print(f"  {cat.capitalize()}: {summary[cat]['mean_dice']:.4f} (n={summary[cat]['n_cases']})")
    
    # Comparison table
    if len(checkpoints) > 1:
        print_comparison_table(all_summaries, names)
    
    # Save combined summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print_success(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
