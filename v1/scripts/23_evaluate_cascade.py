#!/usr/bin/env python3
"""
23. Evaluate Cascaded nnU-Net
=============================
Evaluate cascade model and compare with baseline.

Usage:
    python scripts/23_evaluate_cascade.py --data-dir fusion_data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils import print_header, print_info, print_success
except ImportError:
    def print_header(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")
    def print_info(msg): print(f"[INFO] {msg}")
    def print_success(msg): print(f"[OK] {msg}")


# Size thresholds (voxels)
TINY_THRESHOLD = 500
SMALL_THRESHOLD = 1000
MEDIUM_THRESHOLD = 5000


def compute_dice(pred: np.ndarray, label: np.ndarray) -> float:
    """Compute Dice score."""
    pred_bin = (pred > 0.5).astype(float)
    intersection = np.sum(pred_bin * label)
    union = np.sum(pred_bin) + np.sum(label)
    if union == 0:
        return 1.0 if np.sum(label) == 0 else 0.0
    return 2 * intersection / union


def categorize_lesion(label: np.ndarray) -> str:
    """Categorize lesion by size."""
    voxels = np.sum(label > 0)
    if voxels < TINY_THRESHOLD:
        return 'tiny'
    elif voxels < SMALL_THRESHOLD:
        return 'small'
    elif voxels < MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'large'


def run_cascade_inference(
    image: np.ndarray,
    lowres_model_path: str,
    cascade_model_path: str,
    device: torch.device
) -> np.ndarray:
    """
    Run cascade inference.
    
    Note: This is a simplified version. For production, use nnUNetv2_predict.
    """
    # For proper cascade inference, use nnU-Net's built-in predictor:
    # nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET -c 3d_cascade_fullres
    
    # This function is a placeholder for custom integration
    raise NotImplementedError(
        "Use nnUNetv2_predict for cascade inference:\n"
        "nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_cascade_fullres -f all"
    )


def evaluate_predictions(data_dir: str, predictions_dir: str):
    """Evaluate pre-computed predictions."""
    print_header("Evaluating Cascade Predictions")
    
    data_path = Path(data_dir)
    pred_path = Path(predictions_dir)
    
    if not pred_path.exists():
        print(f"Predictions directory not found: {predictions_dir}")
        print("\nGenerate predictions first using:")
        print("  nnUNetv2_predict \\")
        print("    -i /path/to/test_images \\")
        print("    -o /path/to/predictions \\")
        print("    -d 001 \\")
        print("    -c 3d_cascade_fullres \\")
        print("    -f all")
        return
    
    # Find prediction files
    pred_files = sorted(pred_path.glob("*.nii.gz"))
    print_info(f"Found {len(pred_files)} predictions")
    
    # Compute metrics by category
    results = {'tiny': [], 'small': [], 'medium': [], 'large': []}
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        # Load prediction
        import nibabel as nib
        pred_nib = nib.load(pred_file)
        pred = pred_nib.get_fdata()
        
        # Find corresponding label
        case_id = pred_file.stem.replace('.nii', '')
        label_file = data_path / f"{case_id}_label.nii.gz"
        
        if not label_file.exists():
            # Try NPZ format
            npz_file = data_path / f"{case_id}.npz"
            if npz_file.exists():
                data = np.load(npz_file)
                label = data['label']
            else:
                continue
        else:
            label_nib = nib.load(label_file)
            label = label_nib.get_fdata()
        
        # Compute Dice
        dice = compute_dice(pred, label)
        category = categorize_lesion(label)
        results[category].append(dice)
    
    # Print results
    print_header("Cascade Model Results")
    
    all_dice = []
    for cat in ['tiny', 'small', 'medium', 'large']:
        if results[cat]:
            mean_dice = np.mean(results[cat])
            std_dice = np.std(results[cat])
            n = len(results[cat])
            print(f"  {cat:8s}: {mean_dice:.4f} +/- {std_dice:.4f} (n={n})")
            all_dice.extend(results[cat])
    
    if all_dice:
        print(f"\n  Overall: {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    
    print_success("Evaluation complete")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cascade nnU-Net")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='fusion_data',
        help='Directory with ground truth labels'
    )
    parser.add_argument(
        '--predictions-dir',
        type=str,
        help='Directory with cascade predictions'
    )
    
    args = parser.parse_args()
    
    if args.predictions_dir:
        evaluate_predictions(args.data_dir, args.predictions_dir)
    else:
        print("Cascade nnU-Net Evaluation")
        print("=" * 50)
        print()
        print("Step 1: Generate predictions using nnU-Net:")
        print()
        print("  # For cascade model:")
        print("  nnUNetv2_predict \\")
        print("    -i /path/to/test_images \\")
        print("    -o cascade_predictions \\")
        print("    -d 001 \\")
        print("    -c 3d_cascade_fullres \\")
        print("    -f all")
        print()
        print("Step 2: Evaluate predictions:")
        print()
        print("  python scripts/23_evaluate_cascade.py \\")
        print("    --data-dir fusion_data \\")
        print("    --predictions-dir cascade_predictions")


if __name__ == "__main__":
    main()
