#!/usr/bin/env python3
"""
16. Evaluate Ensemble
=====================
Comprehensive evaluation of the full ensemble pipeline.

This script:
1. Compares individual models vs simple averaging vs learned fusion
2. Computes per-case and aggregate metrics
3. Stratified analysis by lesion size
4. Generates comparison tables and visualizations

Usage:
    python scripts/16_evaluate_ensemble.py \
        --data-dir fusion_data \
        --fusion-checkpoint fusion_checkpoints/checkpoint_best.pth \
        --output-dir results/ensemble_evaluation
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
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics."""
    pred_flat = pred.flatten().astype(bool)
    label_flat = label.flatten().astype(bool)
    
    tp = np.sum(pred_flat & label_flat)
    fp = np.sum(pred_flat & ~label_flat)
    tn = np.sum(~pred_flat & ~label_flat)
    fn = np.sum(~pred_flat & label_flat)
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    # Volume in voxels
    pred_volume = np.sum(pred_flat)
    label_volume = np.sum(label_flat)
    
    return {
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'pred_volume': int(pred_volume),
        'label_volume': int(label_volume),
    }


def evaluate_individual_models(data: Dict) -> Dict[str, Dict[str, float]]:
    """Evaluate each individual model."""
    results = {}
    label = data['label']
    
    if 'nnunet' in data:
        pred = (data['nnunet'] > 0.5).astype(np.uint8)
        results['nnunet'] = compute_metrics(pred, label)
    
    if 'swin_unetr' in data:
        pred = (data['swin_unetr'] > 0.5).astype(np.uint8)
        results['swin_unetr'] = compute_metrics(pred, label)
    
    if 'segresnet' in data:
        pred = (data['segresnet'] > 0.5).astype(np.uint8)
        results['segresnet'] = compute_metrics(pred, label)
    
    return results


def evaluate_simple_average(data: Dict) -> Dict[str, float]:
    """Evaluate simple averaging of predictions."""
    predictions = []
    
    if 'nnunet' in data:
        predictions.append(data['nnunet'])
    if 'swin_unetr' in data:
        predictions.append(data['swin_unetr'])
    if 'segresnet' in data:
        predictions.append(data['segresnet'])
    
    if not predictions:
        return {}
    
    avg_pred = np.mean(predictions, axis=0)
    pred_bin = (avg_pred > 0.5).astype(np.uint8)
    
    return compute_metrics(pred_bin, data['label'])


def evaluate_learned_fusion(
    data: Dict,
    model: AdaptiveFusionNetwork,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate learned fusion network."""
    # Stack predictions in CRITICAL order: [nnunet, swin, segresnet]
    predictions = [
        data['nnunet'] if 'nnunet' in data else np.zeros_like(data['label']),
        data['swin_unetr'] if 'swin_unetr' in data else np.zeros_like(data['label']),
        data['segresnet'] if 'segresnet' in data else np.zeros_like(data['label'])
    ]
    predictions = np.stack(predictions, axis=0).astype(np.float32)
    image = data['image'].astype(np.float32)
    
    # Move to device
    pred_tensor = torch.from_numpy(predictions).unsqueeze(0).to(device)
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    
    # Run fusion
    with torch.no_grad():
        output = model(pred_tensor, image_tensor)
        fused_prob = output[0, 0].cpu().numpy()
    
    pred_bin = (fused_prob > 0.5).astype(np.uint8)
    metrics = compute_metrics(pred_bin, data['label'])
    
    return metrics, fused_prob


def categorize_lesion_size(volume: int, voxel_spacing_ml: float = 0.001) -> str:
    """Categorize lesion by size."""
    volume_ml = volume * voxel_spacing_ml
    
    if volume_ml < 5:
        return 'small'
    elif volume_ml < 50:
        return 'medium'
    else:
        return 'large'


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with pre-computed predictions')
    parser.add_argument('--fusion-checkpoint', type=str, default=None,
                        help='Fusion network checkpoint')
    parser.add_argument('--output-dir', type=str, default='results/ensemble_evaluation',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fusion model if provided
    fusion_model = None
    if args.fusion_checkpoint and Path(args.fusion_checkpoint).exists():
        print_step("Loading fusion network...")
        fusion_model, _ = AdaptiveFusionNetwork.load_checkpoint(
            args.fusion_checkpoint, device
        )
        fusion_model.eval()
        print_success("Fusion network loaded!")
    
    # Get all data files
    data_dir = Path(args.data_dir)
    npz_files = sorted(list(data_dir.glob("*.npz")))
    print_info(f"Found {len(npz_files)} cases")
    
    # Evaluate each case
    results = []
    
    for npz_file in tqdm(npz_files, desc="Evaluating"):
        case_id = npz_file.stem
        data = dict(np.load(npz_file))
        
        if 'label' not in data:
            continue
        
        case_results = {'case_id': case_id}
        
        # Lesion size category
        label_volume = data['label'].sum()
        case_results['label_volume'] = int(label_volume)
        case_results['size_category'] = categorize_lesion_size(label_volume)
        
        # Individual models
        individual = evaluate_individual_models(data)
        for model_name, metrics in individual.items():
            for metric_name, value in metrics.items():
                case_results[f'{model_name}_{metric_name}'] = value
        
        # Simple average
        avg_metrics = evaluate_simple_average(data)
        for metric_name, value in avg_metrics.items():
            case_results[f'simple_avg_{metric_name}'] = value
        
        # Learned fusion
        if fusion_model is not None:
            fusion_metrics, _ = evaluate_learned_fusion(data, fusion_model, device)
            for metric_name, value in fusion_metrics.items():
                case_results[f'learned_fusion_{metric_name}'] = value
        
        results.append(case_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save per-case results
    df.to_csv(output_dir / "per_case_results.csv", index=False)
    
    # Aggregate results
    print_header("Overall Results")
    
    summary = {}
    
    # Compute means for each method
    methods = ['nnunet', 'swin_unetr', 'segresnet', 'simple_avg']
    if fusion_model is not None:
        methods.append('learned_fusion')
    
    for method in methods:
        dice_col = f'{method}_dice'
        if dice_col in df.columns:
            mean_dice = df[dice_col].mean()
            std_dice = df[dice_col].std()
            summary[method] = {
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'mean_sensitivity': df[f'{method}_sensitivity'].mean() if f'{method}_sensitivity' in df.columns else None,
                'mean_specificity': df[f'{method}_specificity'].mean() if f'{method}_specificity' in df.columns else None,
            }
            print(f"{method:20s}: Dice = {mean_dice:.4f} ± {std_dice:.4f}")
    
    # Stratified results
    print_header("Stratified by Lesion Size")
    
    stratified = {}
    for size in ['small', 'medium', 'large']:
        size_df = df[df['size_category'] == size]
        if len(size_df) == 0:
            continue
        
        stratified[size] = {'num_cases': len(size_df)}
        print(f"\n{size.upper()} lesions (n={len(size_df)}):")
        
        for method in methods:
            dice_col = f'{method}_dice'
            if dice_col in size_df.columns:
                mean_dice = size_df[dice_col].mean()
                stratified[size][method] = mean_dice
                print(f"  {method:20s}: {mean_dice:.4f}")
    
    # Save summary
    summary_output = {
        'overall': summary,
        'stratified': stratified,
        'num_cases': len(df),
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary_output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    # Improvement analysis
    print_header("Improvement Analysis")
    
    if 'learned_fusion_dice' in df.columns:
        # vs simple average
        improvement_vs_avg = df['learned_fusion_dice'] - df['simple_avg_dice']
        print(f"Learned fusion vs Simple average: {improvement_vs_avg.mean():.4f} ± {improvement_vs_avg.std():.4f}")
        
        # vs best individual
        individual_dice = df[['nnunet_dice', 'swin_unetr_dice', 'segresnet_dice']].max(axis=1)
        improvement_vs_best = df['learned_fusion_dice'] - individual_dice
        print(f"Learned fusion vs Best individual: {improvement_vs_best.mean():.4f} ± {improvement_vs_best.std():.4f}")
    
    print_success(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
