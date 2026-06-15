#!/usr/bin/env python3
"""
14. Generate Base Model Predictions
====================================
Pre-compute predictions from all base models for fusion training.

This script:
1. Loads nnU-Net, Swin-UNETR, and SegResNet checkpoints
2. Runs inference on validation/training data
3. Saves soft predictions (probabilities) as NPY files
4. These will be used to train the fusion network

Usage:
    python scripts/14_generate_base_predictions.py \
        --nnunet-predictions trained_models/predictions_ensemble \
        --swin-config configs/swin_unetr_config.yaml \
        --segresnet-config configs/segresnet_config.yaml \
        --output-dir fusion_data
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

# MONAI imports
from monai.inferers import sliding_window_inference

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.models.swin_unetr import SwinUNETRWrapper
from src.models.segresnet import SegResNetWrapper
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_nnunet_predictions(
    pred_dir: Path,
    case_ids: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load nnU-Net ensemble predictions.
    
    Args:
        pred_dir: Directory with nnU-Net predictions (*.nii.gz)
        case_ids: List of case IDs to load
    
    Returns:
        Dictionary mapping case_id to prediction array
    """
    predictions = {}
    
    for case_id in case_ids:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        if pred_path.exists():
            nii = nib.load(str(pred_path))
            pred = nii.get_fdata().astype(np.float32)
            predictions[case_id] = pred
        else:
            print_warning(f"nnU-Net prediction not found: {pred_path}")
    
    return predictions


def generate_model_predictions(
    model,
    image: torch.Tensor,
    roi_size: tuple = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    tta: bool = False,
) -> np.ndarray:
    """
    Generate soft predictions from a model, optionally with TTA.
    
    Args:
        model: Model to use for inference
        image: Input tensor [1, C, H, W, D]
        roi_size: Sliding window ROI size
        sw_batch_size: Sliding window batch size
        overlap: Overlap between windows
        tta: Whether to use Test-Time Augmentation (8-fold flip/rotate)
    
    Returns:
        Probability map [H, W, D]
    """
    model.eval()
    
    def _get_prob(img):
        with torch.no_grad():
            output = sliding_window_inference(
                img,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode='gaussian',
            )
            return torch.softmax(output, dim=1)[:, 1]  # Return lesion class prob

    with torch.no_grad():
        if not tta:
            lesion_prob = _get_prob(image)
        else:
            # TTA: 8 transformations (flips in 3 dims)
            # This is a standard medical imaging TTA set
            tta_probs = []
            
            # Original
            tta_probs.append(_get_prob(image))
            
            # Flips
            for dims in [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]:
                flipped_img = torch.flip(image, dims=dims)
                flipped_prob = _get_prob(flipped_img)
                # Flip prediction back
                tta_probs.append(torch.flip(flipped_prob, dims=[d-1 for d in dims]))
            
            lesion_prob = torch.mean(torch.stack(tta_probs), dim=0)
        
        lesion_prob = lesion_prob[0].cpu().numpy()
    
    return lesion_prob


def main():
    parser = argparse.ArgumentParser(description="Generate base model predictions for fusion")
    parser.add_argument('--nnunet-predictions', type=str, required=True,
                        help='Directory with nnU-Net predictions')
    parser.add_argument('--swin-config', type=str, default='configs/swin_unetr_config.yaml',
                        help='Swin-UNETR config file')
    parser.add_argument('--swin-checkpoint', type=str, default=None,
                        help='Swin-UNETR checkpoint (default: best from config)')
    parser.add_argument('--swin-folds', type=int, nargs='+', default=[0],
                        help='Swin-UNETR folds to use')
    parser.add_argument('--segresnet-config', type=str, default='configs/segresnet_config.yaml',
                        help='SegResNet config file')
    parser.add_argument('--segresnet-checkpoint', type=str, default=None,
                        help='SegResNet checkpoint (default: best from config)')
    parser.add_argument('--segresnet-folds', type=int, nargs='+', default=[0],
                        help='SegResNet folds to use')
    parser.add_argument('--output-dir', type=str, default='fusion_data',
                        help='Output directory for predictions')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Data split to process')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    if args.tta:
        print_info("TTA enabled (8-fold flips)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    swin_config = load_config(args.swin_config) if Path(args.swin_config).exists() else None
    segresnet_config = load_config(args.segresnet_config) if Path(args.segresnet_config).exists() else None
    
    # Get data paths from config
    if swin_config:
        raw_dir = Path(swin_config['paths']['nnunet_raw'])
    elif segresnet_config:
        raw_dir = Path(segresnet_config['paths']['nnunet_raw'])
    else:
        print_error("No valid config found!")
        return
    
    # Determine image directory based on split
    if args.split == 'test':
        images_dir = raw_dir / "imagesTs"
        labels_dir = raw_dir / "labelsTs"
    else:
        images_dir = raw_dir / "imagesTr"
        labels_dir = raw_dir / "labelsTr"
    
    if not images_dir.exists():
        print_error(f"Images directory not found: {images_dir}")
        return
    
    # Get case IDs
    case_ids = sorted([f.name.replace('_0000.nii.gz', '') for f in images_dir.glob("*_0000.nii.gz")])
    print_info(f"Found {len(case_ids)} cases")
    
    # Load nnU-Net predictions
    print_step("Loading nnU-Net predictions...")
    nnunet_preds = load_nnunet_predictions(Path(args.nnunet_predictions), case_ids)
    print_success(f"Loaded {len(nnunet_preds)} nnU-Net predictions")
    
    # Load Swin-UNETR models
    swin_models = []
    if swin_config:
        print_step("Loading Swin-UNETR models...")
        for fold in args.swin_folds:
            checkpoint_path = (Path(swin_config['paths']['checkpoints']) / 
                             f"fold_{fold}" / "checkpoint_best.pth")
            if args.swin_checkpoint:
                checkpoint_path = Path(args.swin_checkpoint)
            
            if checkpoint_path.exists():
                model, _ = SwinUNETRWrapper.load_checkpoint(checkpoint_path, device)
                model.eval()
                swin_models.append(model)
            else:
                print_warning(f"Swin-UNETR checkpoint not found: {checkpoint_path}")
        print_success(f"Loaded {len(swin_models)} Swin-UNETR models")
    
    # Load SegResNet models
    segresnet_models = []
    if segresnet_config:
        print_step("Loading SegResNet models...")
        for fold in args.segresnet_folds:
            checkpoint_path = (Path(segresnet_config['paths']['checkpoints']) / 
                             f"fold_{fold}" / "checkpoint_best.pth")
            if args.segresnet_checkpoint:
                checkpoint_path = Path(args.segresnet_checkpoint)
            
            if checkpoint_path.exists():
                model, _ = SegResNetWrapper.load_checkpoint(checkpoint_path, device)
                model.eval()
                segresnet_models.append(model)
            else:
                print_warning(f"SegResNet checkpoint not found: {checkpoint_path}")
        print_success(f"Loaded {len(segresnet_models)} SegResNet models")
    
    # Inference settings
    roi_size = (128, 128, 128)
    sw_batch_size = 4
    overlap = 0.5
    
    # Process each case
    print_header("Generating Predictions")
    
    for case_id in tqdm(case_ids, desc="Processing cases"):
        # Skip if nnU-Net prediction is missing
        if case_id not in nnunet_preds:
            continue
        
        # Load image
        channels = []
        for ch in range(3):
            img_path = images_dir / f"{case_id}_{ch:04d}.nii.gz"
            if img_path.exists():
                nii = nib.load(str(img_path))
                channels.append(nii.get_fdata().astype(np.float32))
        
        if len(channels) != 3:
            print_warning(f"Missing channels for {case_id}")
            continue
        
        image = np.stack(channels, axis=0)
        
        # Normalize
        for c in range(3):
            mask = image[c] > 0
            if mask.sum() > 0:
                mean = image[c][mask].mean()
                std = image[c][mask].std()
                if std > 0:
                    image[c] = (image[c] - mean) / std
        
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        
        # Get predictions from each model type
        predictions = {}
        
        # nnU-Net (already binary, convert to soft)
        nnunet_pred = nnunet_preds[case_id]
        predictions['nnunet'] = nnunet_pred.astype(np.float32)
        
        # Swin-UNETR ensemble
        if swin_models:
            swin_probs = []
            for model in swin_models:
                prob = generate_model_predictions(
                    model, image_tensor, roi_size, sw_batch_size, overlap, args.tta
                )
                swin_probs.append(prob)
            predictions['swin_unetr'] = np.mean(swin_probs, axis=0)
        
        # SegResNet ensemble
        if segresnet_models:
            segresnet_probs = []
            for model in segresnet_models:
                prob = generate_model_predictions(
                    model, image_tensor, roi_size, sw_batch_size, overlap
                )
                segresnet_probs.append(prob)
            predictions['segresnet'] = np.mean(segresnet_probs, axis=0)
        
        # Load ground truth if available
        label_path = labels_dir / f"{case_id}.nii.gz"
        if label_path.exists():
            label = nib.load(str(label_path)).get_fdata().astype(np.float32)
            predictions['label'] = label
        
        # Save image for fusion network context
        predictions['image'] = image
        
        # Save as NPY
        case_output = output_dir / f"{case_id}.npz"
        np.savez_compressed(case_output, **predictions)
    
    # Save metadata
    metadata = {
        'num_cases': len(case_ids),
        'models': {
            'nnunet': 'ensemble',
            'swin_unetr': {
                'folds': args.swin_folds,
                'num_models': len(swin_models),
            },
            'segresnet': {
                'folds': args.segresnet_folds,
                'num_models': len(segresnet_models),
            },
        },
        'split': args.split,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_success(f"\nPredictions saved to: {output_dir}")
    print_info(f"Total cases processed: {len(case_ids)}")


if __name__ == "__main__":
    main()
