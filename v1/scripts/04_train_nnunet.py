#!/usr/bin/env python3
"""
04. nnU-Net Training Script
===========================
Train nnU-Net model for stroke lesion segmentation.

This script:
1. Trains a single fold of nnU-Net
2. Supports resume from checkpoint
3. Provides progress monitoring
4. Saves best checkpoints

Usage:
    # Train single fold
    python scripts/04_train_nnunet.py --fold 0 --config configs/nnunet_config.yaml
    
    # Continue from checkpoint
    python scripts/04_train_nnunet.py --fold 0 --continue
    
    # Validation only
    python scripts/04_train_nnunet.py --fold 0 --val-only

Note: For training all 5 folds, use 05_train_all_folds.py
"""

import argparse
import json
import os
import subprocess
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


def get_gpu_info() -> dict:
    """Get GPU information using nvidia-smi."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_version': torch.version.cuda
            }
    except Exception:
        pass
    
    return {'available': False}


def check_training_status(config: dict, fold: int) -> dict:
    """
    Check the status of training for a specific fold.
    
    Args:
        config: Configuration dictionary
        fold: Fold number
        
    Returns:
        Status dictionary
    """
    dataset_name = get_dataset_name(config)
    trainer = config['training'].get('trainer', 'nnUNetTrainer')
    plans = config['training'].get('plans', 'nnUNetPlans')
    configuration = config['training']['configuration']
    
    results_path = Path(os.environ['nnUNet_results']) / dataset_name / \
                   f"{trainer}__{plans}__{configuration}" / f"fold_{fold}"
    
    status = {
        'fold': fold,
        'results_path': str(results_path),
        'exists': results_path.exists(),
        'checkpoint_final': False,
        'checkpoint_best': False,
        'checkpoint_latest': False,
        'latest_epoch': None
    }
    
    if results_path.exists():
        # Check for checkpoints
        status['checkpoint_final'] = (results_path / 'checkpoint_final.pth').exists()
        status['checkpoint_best'] = (results_path / 'checkpoint_best.pth').exists()
        status['checkpoint_latest'] = (results_path / 'checkpoint_latest.pth').exists()
        
        # Try to get epoch info
        debug_json = results_path / 'debug.json'
        if debug_json.exists():
            try:
                with open(debug_json, 'r') as f:
                    debug = json.load(f)
                    status['latest_epoch'] = debug.get('current_epoch')
            except Exception:
                pass
    
    return status


def train_fold(config: dict, fold: int, continue_training: bool = False, 
               val_only: bool = False, max_retries: int = 3) -> bool:
    """
    Train a single fold of nnU-Net.
    
    Args:
        config: Configuration dictionary
        fold: Fold number (0-4)
        continue_training: Whether to continue from checkpoint
        val_only: Only run validation
        max_retries: Maximum retry attempts on failure
        
    Returns:
        True if training completed successfully
    """
    dataset_id = config['dataset']['id']
    dataset_name = get_dataset_name(config)
    configuration = config['training']['configuration']
    trainer = config['training'].get('trainer', 'nnUNetTrainer')
    plans = config['training'].get('plans', 'nnUNetPlans')
    save_npz = config['training'].get('save_npz', True)
    
    print_header(f"nnU-Net Training - Fold {fold}")
    
    # Display GPU info
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print_info(f"GPU: {gpu_info['name']}")
        print_info(f"Memory: {gpu_info['memory_total']:.1f} GB")
        print_info(f"CUDA: {gpu_info['cuda_version']}")
    else:
        print_warning("No GPU detected - training will be very slow!")
    
    print()
    print_info(f"Dataset: {dataset_name}")
    print_info(f"Configuration: {configuration}")
    print_info(f"Trainer: {trainer}")
    print_info(f"Plans: {plans}")
    
    # Check current status
    status = check_training_status(config, fold)
    
    if status['checkpoint_final'] and not continue_training and not val_only:
        print_success(f"Fold {fold} already complete!")
        return True
    
    if status['checkpoint_latest'] and not continue_training:
        print_info(f"Found checkpoint at epoch {status.get('latest_epoch', 'unknown')}")
        print_info("Use --continue to resume training")
    
    # Build command
    cmd = [
        "nnUNetv2_train",
        dataset_id,
        configuration,
        str(fold),
        "-tr", trainer,
        "-p", plans
    ]
    
    if save_npz:
        cmd.append("--npz")
    
    if val_only:
        cmd.append("--val")
        print_info("Running validation only")
    elif continue_training and status['checkpoint_latest']:
        cmd.append("--c")
        print_info("Continuing from checkpoint")
    
    # Results path for monitoring
    results_path = Path(os.environ['nnUNet_results']) / dataset_name / \
                   f"{trainer}__{plans}__{configuration}" / f"fold_{fold}"
    
    print()
    print("-" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    print()
    
    # Run training with retry logic
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            print_step(attempt + 1, max_retries, f"Training attempt {attempt + 1}")
            
            # Run training
            result = subprocess.run(cmd, check=True)
            
            # Check if completed
            if (results_path / 'checkpoint_final.pth').exists():
                elapsed = time.time() - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                
                print()
                print_header("Training Complete!")
                print_success(f"Fold {fold} finished in {elapsed_str}")
                print_success(f"Results saved to: {results_path}")
                
                return True
                
        except subprocess.CalledProcessError as e:
            print_warning(f"Training interrupted: {e}")
            
            if attempt < max_retries - 1:
                print_info("Will retry with checkpoint resume...")
                cmd = [c for c in cmd if c != "--c"]  # Remove existing --c
                cmd.append("--c")  # Add resume flag
                time.sleep(10)
            else:
                print_error(f"Training failed after {max_retries} attempts")
                return False
                
        except KeyboardInterrupt:
            print_warning("\nTraining manually interrupted")
            print_info("Progress saved. Run with --continue to resume.")
            return False
    
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train nnU-Net for stroke segmentation"
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
        required=True,
        choices=[0, 1, 2, 3, 4],
        help='Fold to train (0-4)'
    )
    parser.add_argument(
        '--continue', '-cont',
        dest='continue_training',
        action='store_true',
        help='Continue training from checkpoint'
    )
    parser.add_argument(
        '--val-only', '--val',
        action='store_true',
        help='Only run validation (no training)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts on failure'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment variables
    setup_nnunet_env(config)
    
    # Verify environment
    print_info("Verifying environment...")
    for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        if var not in os.environ:
            print_error(f"{var} not set!")
            sys.exit(1)
    print_success("Environment OK")
    
    # Run training
    success = train_fold(
        config=config,
        fold=args.fold,
        continue_training=args.continue_training,
        val_only=args.val_only,
        max_retries=args.max_retries
    )
    
    # Show status of all folds
    print()
    print_header("Training Status (All Folds)")
    
    for fold in range(5):
        status = check_training_status(config, fold)
        
        if status['checkpoint_final']:
            print_success(f"Fold {fold}: Complete")
        elif status['checkpoint_latest']:
            epoch = status.get('latest_epoch', '?')
            print_warning(f"Fold {fold}: In progress (epoch {epoch})")
        else:
            print_info(f"Fold {fold}: Not started")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
