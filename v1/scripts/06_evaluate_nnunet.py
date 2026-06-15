#!/usr/bin/env python3
"""
06. nnU-Net Evaluation Script
=============================
Evaluate trained nnU-Net model on test data.

This script:
1. Runs inference on test set
2. Computes evaluation metrics (Dice, Sensitivity, Specificity, HD95)
3. Generates visualizations
4. Creates evaluation report

Usage:
    # Evaluate single fold
    python scripts/06_evaluate_nnunet.py --fold 0 --config configs/nnunet_config.yaml
    
    # Evaluate all folds (cross-validation)
    python scripts/06_evaluate_nnunet.py --all-folds
    
    # Use best configuration (ensemble of folds)
    python scripts/06_evaluate_nnunet.py --ensemble
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    
    if union == 0:
        return 1.0 if np.sum(gt) == 0 else 0.0
    
    return 2.0 * intersection / union


def compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute sensitivity (recall)."""
    tp = np.sum(pred * gt)
    fn = np.sum(gt) - tp
    
    if tp + fn == 0:
        return 1.0
    
    return tp / (tp + fn)


def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute specificity."""
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred) - np.sum(pred * gt)
    
    if tn + fp == 0:
        return 1.0
    
    return tn / (tn + fp)


def compute_hausdorff_95(pred: np.ndarray, gt: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """Compute 95th percentile Hausdorff distance."""
    try:
        from scipy.ndimage import distance_transform_edt
        
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return float('inf')
        
        # Distance transform
        pred_boundary = pred & ~(pred.astype(bool) == False)
        gt_boundary = gt & ~(gt.astype(bool) == False)
        
        # Compute distances
        pred_dist = distance_transform_edt(~pred.astype(bool), sampling=voxel_spacing)
        gt_dist = distance_transform_edt(~gt.astype(bool), sampling=voxel_spacing)
        
        # Get surface distances
        pred_surface_dist = gt_dist[pred.astype(bool)]
        gt_surface_dist = pred_dist[gt.astype(bool)]
        
        # 95th percentile
        if len(pred_surface_dist) > 0 and len(gt_surface_dist) > 0:
            hd95 = max(np.percentile(pred_surface_dist, 95), 
                      np.percentile(gt_surface_dist, 95))
            return hd95
        
        return float('inf')
    except Exception:
        return float('inf')


def run_inference(config: Dict, fold: int, output_dir: Path) -> bool:
    """
    Run nnU-Net inference on test data.
    
    Args:
        config: Configuration dictionary
        fold: Fold number
        output_dir: Output directory for predictions
        
    Returns:
        True if successful
    """
    dataset_id = config['dataset']['id']
    configuration = config['training']['configuration']
    trainer = config['training'].get('trainer', 'nnUNetTrainer')
    plans = config['training'].get('plans', 'nnUNetPlans')
    dataset_name = get_dataset_name(config)
    
    input_dir = Path(os.environ['nnUNet_raw']) / dataset_name / 'imagesTs'
    
    if not input_dir.exists():
        print_warning(f"Test images not found: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", dataset_id,
        "-c", configuration,
        "-f", str(fold),
        "-tr", trainer,
        "-p", plans
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Inference failed: {e}")
        return False


def evaluate_predictions(
    pred_dir: Path,
    gt_dir: Path,
    output_file: Path,
    voxel_spacing: Tuple[float, ...] = (1.95, 1.95, 1.95)
) -> pd.DataFrame:
    """
    Evaluate predictions against ground truth.
    
    Args:
        pred_dir: Directory containing predictions
        gt_dir: Directory containing ground truth
        output_file: Output file for metrics
        voxel_spacing: Voxel spacing for HD95 computation
        
    Returns:
        DataFrame with evaluation metrics
    """
    import nibabel as nib
    
    results = []
    
    pred_files = sorted(pred_dir.glob('*.nii.gz'))
    
    print_info(f"Evaluating {len(pred_files)} predictions...")
    
    for pred_file in pred_files:
        case_id = pred_file.stem.replace('.nii', '')
        gt_file = gt_dir / f'{case_id}.nii.gz'
        
        if not gt_file.exists():
            print_warning(f"Ground truth not found for {case_id}")
            continue
        
        # Load volumes
        pred_data = (nib.load(pred_file).get_fdata() > 0).astype(np.uint8)
        gt_data = (nib.load(gt_file).get_fdata() > 0).astype(np.uint8)
        
        # Compute metrics
        dice = compute_dice(pred_data, gt_data)
        sens = compute_sensitivity(pred_data, gt_data)
        spec = compute_specificity(pred_data, gt_data)
        hd95 = compute_hausdorff_95(pred_data, gt_data, voxel_spacing)
        
        # Volume info
        pred_volume = np.sum(pred_data) * np.prod(voxel_spacing) / 1000  # ml
        gt_volume = np.sum(gt_data) * np.prod(voxel_spacing) / 1000  # ml
        
        results.append({
            'case_id': case_id,
            'dice': dice,
            'sensitivity': sens,
            'specificity': spec,
            'hd95': hd95 if hd95 != float('inf') else np.nan,
            'pred_volume_ml': pred_volume,
            'gt_volume_ml': gt_volume,
            'volume_diff_ml': pred_volume - gt_volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(output_file, index=False)
    print_success(f"Results saved to {output_file}")
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    
    print(f"\nNumber of cases: {len(df)}")
    
    print(f"\nDice Score:")
    print(f"  Mean:   {df['dice'].mean():.4f}")
    print(f"  Std:    {df['dice'].std():.4f}")
    print(f"  Median: {df['dice'].median():.4f}")
    print(f"  Min:    {df['dice'].min():.4f}")
    print(f"  Max:    {df['dice'].max():.4f}")
    
    print(f"\nSensitivity:")
    print(f"  Mean:   {df['sensitivity'].mean():.4f}")
    
    print(f"\nSpecificity:")
    print(f"  Mean:   {df['specificity'].mean():.4f}")
    
    hd95_valid = df['hd95'].dropna()
    if len(hd95_valid) > 0:
        print(f"\nHausdorff Distance 95%:")
        print(f"  Mean:   {hd95_valid.mean():.2f} mm")
        print(f"  Median: {hd95_valid.median():.2f} mm")
    
    print(f"\nVolume Correlation:")
    if len(df) > 1:
        correlation = df['pred_volume_ml'].corr(df['gt_volume_ml'])
        print(f"  R²:     {correlation**2:.4f}")
    
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net model"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--fold', '-f',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Fold to evaluate'
    )
    parser.add_argument(
        '--all-folds',
        action='store_true',
        help='Evaluate all folds'
    )
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip inference, only compute metrics on existing predictions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory for predictions'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment
    setup_nnunet_env(config)
    
    print_header("nnU-Net Evaluation")
    
    dataset_name = get_dataset_name(config)
    results_base = Path(config['paths'].get('results', 'results'))
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Ground truth directory
    gt_dir = Path(os.environ['nnUNet_raw']) / dataset_name / 'labelsTs'
    
    if not gt_dir.exists():
        print_error(f"Ground truth directory not found: {gt_dir}")
        return
    
    folds = range(5) if args.all_folds else [args.fold]
    all_results = []
    
    for fold in folds:
        print_step(fold + 1, len(folds) if args.all_folds else 1, f"Evaluating fold {fold}")
        
        # Output directory
        pred_dir = Path(args.output_dir) if args.output_dir else results_base / f'predictions_fold{fold}'
        
        # Run inference if needed
        if not args.skip_inference:
            success = run_inference(config, fold, pred_dir)
            if not success:
                print_warning(f"Inference failed for fold {fold}, skipping...")
                continue
        
        # Check predictions exist
        if not pred_dir.exists() or len(list(pred_dir.glob('*.nii.gz'))) == 0:
            print_warning(f"No predictions found in {pred_dir}")
            continue
        
        # Evaluate
        metrics_file = results_base / f'metrics_fold{fold}.csv'
        df = evaluate_predictions(pred_dir, gt_dir, metrics_file)
        df['fold'] = fold
        all_results.append(df)
        
        print_summary(df)
    
    # Combined results
    if len(all_results) > 1:
        print_header("Cross-Validation Summary")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(results_base / 'metrics_all_folds.csv', index=False)
        
        print(f"\nCombined results across {len(all_results)} folds:")
        print_summary(combined_df)
        
        # Per-fold summary
        print("\nPer-fold Dice scores:")
        for fold in combined_df['fold'].unique():
            fold_df = combined_df[combined_df['fold'] == fold]
            print(f"  Fold {fold}: {fold_df['dice'].mean():.4f} ± {fold_df['dice'].std():.4f}")
    
    print_success("\nEvaluation complete!")
    print(f"Results saved to: {results_base}")


if __name__ == "__main__":
    main()
