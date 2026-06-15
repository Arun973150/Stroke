#!/usr/bin/env python3
"""
10. Evaluate Swin-UNETR
=======================
Evaluate trained Swin-UNETR model on test data.

This script:
1. Loads trained Swin-UNETR checkpoints
2. Runs sliding window inference on test set
3. Computes Dice, sensitivity, specificity, Hausdorff distance
4. Saves predictions as NIfTI files
5. Outputs per-case and aggregate metrics

Usage:
    python scripts/10_evaluate_swin_unetr.py --config configs/swin_unetr_config.yaml --fold 0
    
Evaluate ensemble of multiple folds:
    python scripts/10_evaluate_swin_unetr.py --config configs/swin_unetr_config.yaml --folds 0 1 2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# MONAI imports
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.models.swin_unetr import SwinUNETRWrapper
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    """
    Compute segmentation metrics.
    
    Args:
        pred: Binary prediction mask
        label: Binary ground truth mask
    
    Returns:
        Dictionary of metrics
    """
    # Flatten for metric computation
    pred_flat = pred.flatten().astype(bool)
    label_flat = label.flatten().astype(bool)
    
    # True/False positives/negatives
    tp = np.sum(pred_flat & label_flat)
    fp = np.sum(pred_flat & ~label_flat)
    tn = np.sum(~pred_flat & ~label_flat)
    fn = np.sum(~pred_flat & label_flat)
    
    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # Sensitivity (recall)
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificity
    specificity = tn / (tn + fp + 1e-8)
    
    # Precision
    precision = tp / (tp + fp + 1e-8)
    
    # Volume information
    pred_volume = np.sum(pred_flat)
    label_volume = np.sum(label_flat)
    
    return {
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'pred_volume': int(pred_volume),
        'label_volume': int(label_volume),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def evaluate_fold(
    config: dict,
    fold: int,
    checkpoint: str = "checkpoint_best.pth",
    save_predictions: bool = True,
) -> Dict[str, any]:
    """
    Evaluate a single fold on test data.
    
    Args:
        config: Configuration dictionary
        fold: Fold number
        checkpoint: Checkpoint filename to load
        save_predictions: Whether to save prediction NIfTIs
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device(f"cuda:{config['gpu']['device_id']}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = Path(config['paths']['checkpoints']) / f"fold_{fold}" / checkpoint
    
    if not checkpoint_path.exists():
        print_error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print_step(f"Loading model from {checkpoint_path}")
    model, _ = SwinUNETRWrapper.load_checkpoint(checkpoint_path, device)
    model.eval()
    
    # Load test data paths
    raw_dir = Path(config['paths']['nnunet_raw'])
    test_images_dir = raw_dir / "imagesTs"
    test_labels_dir = raw_dir / "labelsTs"
    
    if not test_images_dir.exists():
        print_error(f"Test images directory not found: {test_images_dir}")
        return None
    
    # Get test case IDs
    test_cases = sorted([f.stem.replace('_0000', '') for f in test_images_dir.glob("*_0000.nii.gz")])
    print_info(f"Found {len(test_cases)} test cases")
    
    # Output directory
    output_dir = Path(config['paths']['results']) / f"fold_{fold}" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inference settings
    roi_size = tuple(config.get('inference', {}).get('roi_size', [128, 128, 128]))
    sw_batch_size = config.get('inference', {}).get('sw_batch_size', 4)
    overlap = config.get('inference', {}).get('overlap', 0.5)
    
    post_pred = AsDiscrete(argmax=True)
    
    results = []
    
    with torch.no_grad():
        for case_id in tqdm(test_cases, desc=f"Evaluating fold {fold}"):
            # Load image channels
            channels = []
            for ch in range(3):  # DWI, ADC, FLAIR
                img_path = test_images_dir / f"{case_id}_{ch:04d}.nii.gz"
                if img_path.exists():
                    nii = nib.load(str(img_path))
                    channels.append(nii.get_fdata())
                else:
                    print_warning(f"Missing channel {ch} for {case_id}")
                    break
            
            if len(channels) != 3:
                continue
            
            # Stack channels and prepare for inference
            image = np.stack(channels, axis=0).astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(0).to(device)  # [1, 3, H, W, D]
            
            # Sliding window inference
            output = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode='gaussian',
            )
            
            # Post-process
            pred = post_pred(output[0])  # [H, W, D]
            pred_np = pred.cpu().numpy().astype(np.uint8)
            
            # Load ground truth if available
            label_path = test_labels_dir / f"{case_id}.nii.gz"
            if label_path.exists():
                label_nii = nib.load(str(label_path))
                label_np = label_nii.get_fdata().astype(np.uint8)
                
                # Compute metrics
                metrics = compute_metrics(pred_np, label_np)
                metrics['case_id'] = case_id
                results.append(metrics)
                
                print(f"{case_id}: Dice = {metrics['dice']:.4f}")
            else:
                print_warning(f"No ground truth for {case_id}")
            
            # Save prediction
            if save_predictions:
                # Use reference image for affine/header
                ref_nii = nib.load(str(test_images_dir / f"{case_id}_0000.nii.gz"))
                pred_nii = nib.Nifti1Image(pred_np, ref_nii.affine, ref_nii.header)
                nib.save(pred_nii, output_dir / f"{case_id}.nii.gz")
    
    # Aggregate results
    if results:
        df = pd.DataFrame(results)
        
        summary = {
            'fold': fold,
            'num_cases': len(results),
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'mean_sensitivity': df['sensitivity'].mean(),
            'mean_specificity': df['specificity'].mean(),
            'mean_precision': df['precision'].mean(),
        }
        
        # Save per-case results
        df.to_csv(output_dir.parent / "per_case_metrics.csv", index=False)
        
        # Save summary
        with open(output_dir.parent / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print_success(f"\nFold {fold} Results:")
        print(f"  Mean Dice: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"  Mean Sensitivity: {summary['mean_sensitivity']:.4f}")
        print(f"  Mean Specificity: {summary['mean_specificity']:.4f}")
        
        return summary
    
    return None


def evaluate_ensemble(
    config: dict,
    folds: List[int],
    checkpoint: str = "checkpoint_best.pth",
) -> Dict[str, any]:
    """
    Evaluate ensemble of multiple folds.
    
    Args:
        config: Configuration dictionary
        folds: List of fold numbers to ensemble
        checkpoint: Checkpoint filename to load
    
    Returns:
        Dictionary with ensemble evaluation results
    """
    device = torch.device(f"cuda:{config['gpu']['device_id']}" if torch.cuda.is_available() else "cpu")
    
    # Load all models
    models = []
    for fold in folds:
        checkpoint_path = Path(config['paths']['checkpoints']) / f"fold_{fold}" / checkpoint
        if checkpoint_path.exists():
            model, _ = SwinUNETRWrapper.load_checkpoint(checkpoint_path, device)
            model.eval()
            models.append(model)
            print_info(f"Loaded fold {fold}")
        else:
            print_warning(f"Checkpoint not found for fold {fold}, skipping")
    
    if not models:
        print_error("No models loaded!")
        return None
    
    print_info(f"Ensemble of {len(models)} models")
    
    # Load test data
    raw_dir = Path(config['paths']['nnunet_raw'])
    test_images_dir = raw_dir / "imagesTs"
    test_labels_dir = raw_dir / "labelsTs"
    
    test_cases = sorted([f.stem.replace('_0000', '') for f in test_images_dir.glob("*_0000.nii.gz")])
    
    # Output directory
    output_dir = Path(config['paths']['results']) / f"ensemble_{'_'.join(map(str, folds))}" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    roi_size = tuple(config.get('inference', {}).get('roi_size', [128, 128, 128]))
    sw_batch_size = config.get('inference', {}).get('sw_batch_size', 4)
    overlap = config.get('inference', {}).get('overlap', 0.5)
    
    results = []
    
    with torch.no_grad():
        for case_id in tqdm(test_cases, desc="Evaluating ensemble"):
            # Load image
            channels = []
            for ch in range(3):
                img_path = test_images_dir / f"{case_id}_{ch:04d}.nii.gz"
                if img_path.exists():
                    channels.append(nib.load(str(img_path)).get_fdata())
            
            if len(channels) != 3:
                continue
            
            image = np.stack(channels, axis=0).astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            
            # Ensemble prediction (average probabilities)
            ensemble_pred = None
            for model in models:
                output = sliding_window_inference(
                    image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode='gaussian',
                )
                
                # Softmax probabilities
                probs = torch.softmax(output, dim=1)
                
                if ensemble_pred is None:
                    ensemble_pred = probs
                else:
                    ensemble_pred += probs
            
            # Average and argmax
            ensemble_pred /= len(models)
            pred = torch.argmax(ensemble_pred[0], dim=0).cpu().numpy().astype(np.uint8)
            
            # Evaluate
            label_path = test_labels_dir / f"{case_id}.nii.gz"
            if label_path.exists():
                label = nib.load(str(label_path)).get_fdata().astype(np.uint8)
                metrics = compute_metrics(pred, label)
                metrics['case_id'] = case_id
                results.append(metrics)
                print(f"{case_id}: Dice = {metrics['dice']:.4f}")
            
            # Save prediction
            ref_nii = nib.load(str(test_images_dir / f"{case_id}_0000.nii.gz"))
            pred_nii = nib.Nifti1Image(pred, ref_nii.affine, ref_nii.header)
            nib.save(pred_nii, output_dir / f"{case_id}.nii.gz")
    
    # Aggregate
    if results:
        df = pd.DataFrame(results)
        summary = {
            'folds': folds,
            'num_models': len(models),
            'num_cases': len(results),
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'mean_sensitivity': df['sensitivity'].mean(),
            'mean_specificity': df['specificity'].mean(),
        }
        
        df.to_csv(output_dir.parent / "per_case_metrics.csv", index=False)
        
        with open(output_dir.parent / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print_success(f"\nEnsemble Results ({len(models)} models):")
        print(f"  Mean Dice: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"  Mean Sensitivity: {summary['mean_sensitivity']:.4f}")
        
        return summary
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNETR model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--fold', type=int, default=None, help='Single fold to evaluate')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='Multiple folds for ensemble')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth', help='Checkpoint filename')
    parser.add_argument('--no-save', action='store_true', help='Do not save predictions')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.folds and len(args.folds) > 1:
        # Ensemble evaluation
        results = evaluate_ensemble(
            config=config,
            folds=args.folds,
            checkpoint=args.checkpoint,
        )
    elif args.fold is not None or (args.folds and len(args.folds) == 1):
        # Single fold evaluation
        fold = args.fold if args.fold is not None else args.folds[0]
        results = evaluate_fold(
            config=config,
            fold=fold,
            checkpoint=args.checkpoint,
            save_predictions=not args.no_save,
        )
    else:
        print_error("Please specify --fold or --folds")
        return
    
    if results:
        print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
