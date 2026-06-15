#!/usr/bin/env python3
"""
05. Train All nnU-Net Folds Script
==================================
Train all 5 folds of nnU-Net sequentially for complete cross-validation.

This script:
1. Trains folds 0-4 sequentially
2. Automatically resumes from checkpoints
3. Provides progress summary
4. Estimates remaining time

Usage:
    python scripts/05_train_all_folds.py --config configs/nnunet_config.yaml
    
    # Start from specific fold
    python scripts/05_train_all_folds.py --start-fold 2
    
    # Skip completed folds
    python scripts/05_train_all_folds.py --skip-completed
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_step, print_success, print_warning, print_error, print_info
)

# Import training functions using importlib (since filename starts with number)
import importlib.util
spec = importlib.util.spec_from_file_location("train_nnunet", PROJECT_ROOT / "scripts" / "04_train_nnunet.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

train_fold = train_module.train_fold
check_training_status = train_module.check_training_status
get_gpu_info = train_module.get_gpu_info


def train_all_folds(
    config: dict,
    start_fold: int = 0,
    skip_completed: bool = True
) -> dict:
    """
    Train all 5 folds sequentially.
    
    Args:
        config: Configuration dictionary
        start_fold: Fold to start from
        skip_completed: Whether to skip already completed folds
        
    Returns:
        Dictionary with training results
    """
    print_header("nnU-Net 5-Fold Cross-Validation Training")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print_info(f"GPU: {gpu_info['name']} ({gpu_info['memory_total']:.1f} GB)")
    else:
        print_warning("No GPU detected!")
    
    print_info(f"Dataset: {get_dataset_name(config)}")
    print_info(f"Configuration: {config['training']['configuration']}")
    print_info(f"Starting from fold: {start_fold}")
    
    # Check current status
    print()
    print_info("Checking current status...")
    
    folds_to_train = []
    fold_times = []
    
    for fold in range(5):
        status = check_training_status(config, fold)
        
        if status['checkpoint_final']:
            if skip_completed:
                print_success(f"Fold {fold}: Already complete (skipping)")
            else:
                folds_to_train.append(fold)
        elif fold >= start_fold:
            folds_to_train.append(fold)
            if status['checkpoint_latest']:
                print_warning(f"Fold {fold}: In progress (will resume)")
            else:
                print_info(f"Fold {fold}: Not started")
    
    if not folds_to_train:
        print_success("\nAll folds already complete!")
        return {'completed': list(range(5)), 'failed': []}
    
    print()
    print_info(f"Folds to train: {folds_to_train}")
    
    # Estimate time
    estimated_per_fold = 8 * 3600  # 8 hours per fold (conservative)
    total_estimated = len(folds_to_train) * estimated_per_fold
    print_info(f"Estimated total time: ~{total_estimated // 3600} hours")
    
    # Train each fold
    results = {'completed': [], 'failed': [], 'times': {}}
    overall_start = time.time()
    
    for i, fold in enumerate(folds_to_train):
        fold_start = time.time()
        
        print()
        print("=" * 70)
        print(f"TRAINING FOLD {fold} ({i + 1}/{len(folds_to_train)})")
        print("=" * 70)
        
        # Check if should continue from checkpoint
        status = check_training_status(config, fold)
        continue_training = status['checkpoint_latest'] and not status['checkpoint_final']
        
        success = train_fold(
            config=config,
            fold=fold,
            continue_training=continue_training,
            max_retries=5
        )
        
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        results['times'][fold] = fold_time
        
        if success:
            results['completed'].append(fold)
            print_success(f"Fold {fold} completed in {timedelta(seconds=int(fold_time))}")
            
            # Update ETA
            if i < len(folds_to_train) - 1:
                avg_time = sum(fold_times) / len(fold_times)
                remaining = len(folds_to_train) - i - 1
                eta = avg_time * remaining
                print_info(f"ETA for remaining {remaining} folds: ~{eta // 3600:.1f} hours")
        else:
            results['failed'].append(fold)
            print_error(f"Fold {fold} failed!")
    
    # Summary
    total_time = time.time() - overall_start
    
    print()
    print_header("Training Summary")
    
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")
    print(f"Completed folds: {results['completed']}")
    print(f"Failed folds: {results['failed']}")
    
    for fold, fold_time in results['times'].items():
        print(f"  Fold {fold}: {timedelta(seconds=int(fold_time))}")
    
    if not results['failed']:
        print_success("\n✓ All folds completed successfully!")
        print_info("\nNext steps:")
        print_info("  1. Run evaluation: python scripts/06_evaluate_nnunet.py")
        print_info("  2. Export checkpoints: python scripts/07_export_checkpoints.py")
    else:
        print_warning(f"\n⚠ {len(results['failed'])} folds failed")
        print_info("Check logs and retry failed folds individually")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train all 5 folds of nnU-Net"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-fold',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Fold to start from'
    )
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        default=True,
        help='Skip already completed folds'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip completed folds (retrain all)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment variables
    setup_nnunet_env(config)
    
    # Run training
    skip_completed = not args.no_skip
    
    results = train_all_folds(
        config=config,
        start_fold=args.start_fold,
        skip_completed=skip_completed
    )
    
    # Exit with appropriate code
    if results['failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
