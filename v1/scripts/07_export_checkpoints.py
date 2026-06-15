#!/usr/bin/env python3
"""
07. Export Checkpoints Script
=============================
Export best nnU-Net checkpoints for use in ensemble pipeline.

This script:
1. Finds best checkpoints from each fold
2. Copies them to organized checkpoint directory
3. Creates model info JSON with metadata
4. Prepares for ensemble/fusion training

Usage:
    python scripts/07_export_checkpoints.py --config configs/nnunet_config.yaml
    
    # Specify output directory
    python scripts/07_export_checkpoints.py --output checkpoints/nnunet/
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def get_fold_info(results_path: Path, fold: int) -> Optional[Dict]:
    """
    Get information about a trained fold.
    
    Args:
        results_path: Path to training results
        fold: Fold number
        
    Returns:
        Dictionary with fold info or None if not found
    """
    fold_path = results_path / f"fold_{fold}"
    
    if not fold_path.exists():
        return None
    
    info = {
        'fold': fold,
        'path': str(fold_path),
        'checkpoint_final': (fold_path / 'checkpoint_final.pth').exists(),
        'checkpoint_best': (fold_path / 'checkpoint_best.pth').exists(),
        'completed': False,
        'best_metric': None
    }
    
    # Check if training completed
    if info['checkpoint_final']:
        info['completed'] = True
        
        # Get file info
        ckpt_path = fold_path / 'checkpoint_best.pth'
        if ckpt_path.exists():
            stat = ckpt_path.stat()
            info['checkpoint_size_mb'] = stat.st_size / (1024**2)
            info['checkpoint_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    # Try to get best metric from debug.json
    debug_file = fold_path / 'debug.json'
    if debug_file.exists():
        try:
            with open(debug_file, 'r') as f:
                debug = json.load(f)
                info['best_metric'] = debug.get('best_mean_dice')
                info['final_epoch'] = debug.get('current_epoch')
        except Exception:
            pass
    
    return info


def export_checkpoints(config: Dict, output_dir: Path) -> Dict:
    """
    Export best checkpoints from all folds.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for exported checkpoints
        
    Returns:
        Export summary dictionary
    """
    print_header("Export nnU-Net Checkpoints")
    
    dataset_name = get_dataset_name(config)
    trainer = config['training'].get('trainer', 'nnUNetTrainer')
    plans = config['training'].get('plans', 'nnUNetPlans')
    configuration = config['training']['configuration']
    
    results_path = Path(os.environ['nnUNet_results']) / dataset_name / \
                   f"{trainer}__{plans}__{configuration}"
    
    print_info(f"Source: {results_path}")
    print_info(f"Target: {output_dir}")
    
    if not results_path.exists():
        print_error(f"Results directory not found: {results_path}")
        return {}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather fold information
    print_step(1, 3, "Gathering fold information...")
    
    folds_info = []
    completed_folds = []
    
    for fold in range(5):
        info = get_fold_info(results_path, fold)
        
        if info is None:
            print_info(f"Fold {fold}: Not found")
        elif not info['completed']:
            print_warning(f"Fold {fold}: Not completed")
        else:
            metric_str = f" (dice: {info['best_metric']:.4f})" if info.get('best_metric') else ""
            print_success(f"Fold {fold}: Complete{metric_str}")
            completed_folds.append(fold)
            folds_info.append(info)
    
    if not completed_folds:
        print_error("No completed folds found!")
        return {}
    
    # Export checkpoints
    print_step(2, 3, f"Exporting {len(completed_folds)} checkpoints...")
    
    exported_files = []
    
    for fold in completed_folds:
        src_best = results_path / f"fold_{fold}" / "checkpoint_best.pth"
        src_final = results_path / f"fold_{fold}" / "checkpoint_final.pth"
        
        if src_best.exists():
            dst = output_dir / f"best_model_fold{fold}.pth"
            shutil.copy(src_best, dst)
            size_mb = dst.stat().st_size / (1024**2)
            print_success(f"Exported fold {fold}: {size_mb:.1f} MB")
            exported_files.append(str(dst))
    
    # Copy plans and dataset info
    plans_file = Path(os.environ['nnUNet_preprocessed']) / dataset_name / "nnUNetPlans.json"
    if plans_file.exists():
        shutil.copy(plans_file, output_dir / "nnUNetPlans.json")
        print_success("Copied nnUNetPlans.json")
    
    # Create model info
    print_step(3, 3, "Creating model metadata...")
    
    model_info = {
        'model_type': 'nnunet_v2',
        'dataset_name': dataset_name,
        'configuration': configuration,
        'trainer': trainer,
        'plans': plans,
        'export_date': datetime.now().isoformat(),
        'folds': folds_info,
        'completed_folds': completed_folds,
        'exported_files': exported_files,
        'average_dice': None
    }
    
    # Calculate average dice if available
    dice_scores = [f.get('best_metric') for f in folds_info if f.get('best_metric')]
    if dice_scores:
        model_info['average_dice'] = sum(dice_scores) / len(dice_scores)
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print_success("Created model_info.json")
    
    # Summary
    print_header("Export Summary")
    
    print(f"\nExported {len(completed_folds)} folds to: {output_dir}")
    print(f"\nFiles created:")
    for f in exported_files:
        print(f"  - {Path(f).name}")
    print(f"  - nnUNetPlans.json")
    print(f"  - model_info.json")
    
    if model_info.get('average_dice'):
        print(f"\nAverage validation Dice: {model_info['average_dice']:.4f}")
    
    print()
    print_info("Next steps:")
    print_info("  1. Train Swin-UNETR: python scripts/07_train_swin_unetr.py")
    print_info("  2. Train SegResNet: python scripts/08_train_segresnet.py")
    print_info("  3. Create ensemble: python scripts/10_generate_ensemble_predictions.py")
    
    return model_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export nnU-Net checkpoints"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for checkpoints'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment
    setup_nnunet_env(config)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['paths'].get('checkpoints', 'checkpoints')) / 'nnunet'
    
    # Export
    export_checkpoints(config, output_dir)


if __name__ == "__main__":
    main()
