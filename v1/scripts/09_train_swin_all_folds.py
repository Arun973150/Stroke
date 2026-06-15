#!/usr/bin/env python3
"""
09. Train Swin-UNETR All Folds
==============================
Train Swin-UNETR model across multiple folds for cross-validation.

Usage:
    python scripts/09_train_swin_all_folds.py --config configs/swin_unetr_config.yaml
    
Train specific folds:
    python scripts/09_train_swin_all_folds.py --config configs/swin_unetr_config.yaml --start-fold 1
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import training function using importlib (filename starts with number)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_swin_unetr", 
    PROJECT_ROOT / "scripts" / "08_train_swin_unetr.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

train_fold = train_module.train_fold
load_config = train_module.load_config

from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def train_all_folds(
    config: dict,
    start_fold: int = 0,
    num_folds: int = 5,
    skip_completed: bool = True,
) -> dict:
    """
    Train Swin-UNETR across all folds.
    
    Args:
        config: Configuration dictionary
        start_fold: Starting fold number
        num_folds: Total number of folds
        skip_completed: Skip folds that already have checkpoints
    
    Returns:
        Dictionary with results for each fold
    """
    results = {}
    total_start_time = time.time()
    
    checkpoint_base = Path(config['paths']['checkpoints'])
    
    print_header("Swin-UNETR Multi-Fold Training")
    print_info(f"Training folds {start_fold} to {num_folds - 1}")
    
    for fold in range(start_fold, num_folds):
        fold_checkpoint_dir = checkpoint_base / f"fold_{fold}"
        best_checkpoint = fold_checkpoint_dir / "checkpoint_best.pth"
        final_checkpoint = fold_checkpoint_dir / "checkpoint_final.pth"
        
        # Check if fold is already complete
        if skip_completed and final_checkpoint.exists():
            print_info(f"Fold {fold}: Already complete, skipping")
            results[fold] = {'status': 'skipped', 'reason': 'already_complete'}
            continue
        
        # Check if we should resume
        resume_path = None
        if best_checkpoint.exists():
            print_info(f"Fold {fold}: Found existing checkpoint, will resume")
            resume_path = str(best_checkpoint)
        
        print_header(f"Training Fold {fold}/{num_folds - 1}")
        fold_start_time = time.time()
        
        try:
            fold_results = train_fold(
                config=config,
                fold=fold,
                resume=resume_path,
            )
            
            fold_time = time.time() - fold_start_time
            fold_results['time'] = fold_time
            fold_results['status'] = 'complete'
            results[fold] = fold_results
            
            print_success(f"Fold {fold} complete! Dice: {fold_results['best_dice']:.4f}, "
                         f"Time: {fold_time / 3600:.2f}h")
            
        except Exception as e:
            print_error(f"Fold {fold} failed: {e}")
            results[fold] = {'status': 'failed', 'error': str(e)}
            
            # Continue with next fold
            continue
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print_header("Training Summary")
    print("=" * 60)
    
    completed_folds = [f for f, r in results.items() if r.get('status') == 'complete']
    failed_folds = [f for f, r in results.items() if r.get('status') == 'failed']
    skipped_folds = [f for f, r in results.items() if r.get('status') == 'skipped']
    
    if completed_folds:
        dice_scores = [results[f]['best_dice'] for f in completed_folds]
        mean_dice = sum(dice_scores) / len(dice_scores)
        print_success(f"Completed folds: {completed_folds}")
        print_info(f"Mean Dice: {mean_dice:.4f}")
        print_info(f"Best Dice: {max(dice_scores):.4f} (fold {completed_folds[dice_scores.index(max(dice_scores))]})")
    
    if skipped_folds:
        print_info(f"Skipped folds: {skipped_folds}")
    
    if failed_folds:
        print_error(f"Failed folds: {failed_folds}")
    
    print_info(f"Total time: {total_time / 3600:.2f} hours")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Swin-UNETR across all folds")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--start-fold', type=int, default=0, help='Starting fold number')
    parser.add_argument('--num-folds', type=int, default=5, help='Total number of folds')
    parser.add_argument('--no-skip', action='store_true', help='Do not skip completed folds')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train all folds
    results = train_all_folds(
        config=config,
        start_fold=args.start_fold,
        num_folds=args.num_folds,
        skip_completed=not args.no_skip,
    )
    
    print("\nFinal Results:")
    for fold, result in results.items():
        print(f"  Fold {fold}: {result}")


if __name__ == "__main__":
    main()
