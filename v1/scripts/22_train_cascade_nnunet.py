#!/usr/bin/env python3
"""
22. Train Cascaded nnU-Net for Tiny Lesions
============================================
Two-stage cascaded nnU-Net specifically designed to improve tiny lesion detection.

Architecture:
    Stage 1 (3d_lowres): Low-resolution model at ~5mm spacing
        - Finds candidate lesion regions
        - Provides global context
        
    Stage 2 (3d_cascade_fullres): Full-resolution model at native spacing
        - Uses low-res predictions as additional input channel
        - Refines segmentation at full resolution
        - Focuses compute on candidate regions

Expected Improvement:
    - Tiny lesions: 0.33 -> 0.40-0.45 Dice (+20-35%)
    - Better localization of small lesions

Usage:
    # Step 1: Preprocess for cascade (only once)
    python scripts/22_train_cascade_nnunet.py --preprocess
    
    # Step 2: Train Stage 1 (low-res) - all folds
    python scripts/22_train_cascade_nnunet.py --stage 1 --fold 0
    python scripts/22_train_cascade_nnunet.py --stage 1 --fold 1
    ...
    
    # Step 3: Train Stage 2 (cascade fullres) - all folds
    python scripts/22_train_cascade_nnunet.py --stage 2 --fold 0
    ...
    
    # Or train all at once:
    python scripts/22_train_cascade_nnunet.py --train-all

Requirements:
    - Dataset must already be set up in nnU-Net format (script 02)
    - Requires A100 GPU (40GB) for cascade training
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils import (
        print_header, print_step, print_success, print_warning, print_error, print_info
    )
except ImportError:
    # Fallback for Lambda Labs
    def print_header(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")
    def print_step(curr, total, msg): print(f"[{curr}/{total}] {msg}")
    def print_success(msg): print(f"[OK] {msg}")
    def print_warning(msg): print(f"[WARN] {msg}")
    def print_error(msg): print(f"[ERROR] {msg}")
    def print_info(msg): print(f"[INFO] {msg}")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_nnunet_env(config: dict):
    """Set up nnU-Net environment variables."""
    paths = config['paths']
    os.environ['nnUNet_raw'] = paths['nnunet_raw']
    os.environ['nnUNet_preprocessed'] = paths['nnunet_preprocessed']
    os.environ['nnUNet_results'] = paths['nnunet_results']
    
    print_info(f"nnUNet_raw: {paths['nnunet_raw']}")
    print_info(f"nnUNet_preprocessed: {paths['nnunet_preprocessed']}")
    print_info(f"nnUNet_results: {paths['nnunet_results']}")


def run_preprocessing(config: dict):
    """
    Run nnU-Net preprocessing for cascade configuration.
    
    This generates plans for both 3d_lowres and 3d_cascade_fullres.
    """
    print_header("Preprocessing for Cascade nnU-Net")
    
    dataset_id = config['dataset']['id']
    
    # Run plan and preprocess for cascade
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        "-c", "3d_lowres", "3d_cascade_fullres",  # Both configurations
        "--verify_dataset_integrity"
    ]
    
    print_info(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print_success("Cascade preprocessing complete!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Preprocessing failed: {e}")
        return False


def train_stage(config: dict, stage: int, fold: int, continue_training: bool = False):
    """
    Train a single stage and fold.
    
    Args:
        config: Configuration dictionary
        stage: 1 (lowres) or 2 (cascade_fullres)
        fold: Fold number (0-4)
        continue_training: Resume from checkpoint
    """
    dataset_id = config['dataset']['id']
    
    if stage == 1:
        configuration = "3d_lowres"
        stage_name = "Low-Resolution (Stage 1)"
    else:
        configuration = "3d_cascade_fullres"
        stage_name = "Cascade Full-Resolution (Stage 2)"
    
    print_header(f"Training {stage_name} - Fold {fold}")
    
    trainer = config['cascade'][f'stage{stage}']['trainer']
    plans = config['cascade'][f'stage{stage}']['plans']
    
    print_info(f"Configuration: {configuration}")
    print_info(f"Trainer: {trainer}")
    print_info(f"Plans: {plans}")
    print_info(f"Fold: {fold}")
    
    # Build command
    cmd = [
        "nnUNetv2_train",
        dataset_id,
        configuration,
        str(fold),
        "-tr", trainer,
        "-p", plans,
        "--npz"  # Save softmax outputs
    ]
    
    if continue_training:
        cmd.append("--c")
        print_info("Continuing from checkpoint")
    
    print()
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print()
        print_success(f"Stage {stage} Fold {fold} complete in {elapsed_str}")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print_warning("Training interrupted. Use --continue to resume.")
        return False


def train_all(config: dict):
    """Train all stages and folds sequentially."""
    folds = config['training']['folds']
    
    print_header("Training Complete Cascade Pipeline")
    print_info(f"Training folds: {folds}")
    print_info("This will take ~24-48 hours on A100")
    print()
    
    # Stage 1: Train all low-res folds
    print_header("STAGE 1: Low-Resolution Models")
    for fold in folds:
        success = train_stage(config, stage=1, fold=fold)
        if not success:
            print_error(f"Stage 1 Fold {fold} failed!")
            return False
    
    print_success("Stage 1 Complete - All low-res folds trained")
    print()
    
    # Stage 2: Train all cascade fullres folds
    print_header("STAGE 2: Cascade Full-Resolution Models")
    for fold in folds:
        success = train_stage(config, stage=2, fold=fold)
        if not success:
            print_error(f"Stage 2 Fold {fold} failed!")
            return False
    
    print_success("Stage 2 Complete - All cascade folds trained")
    print()
    
    print_header("CASCADE TRAINING COMPLETE!")
    print_success("All 10 models (5 lowres + 5 cascade) trained successfully")
    
    return True


def check_status(config: dict):
    """Check training status for all stages and folds."""
    print_header("Cascade Training Status")
    
    dataset_id = config['dataset']['id']
    results_base = Path(os.environ.get('nnUNet_results', '/home/ubuntu/nnUNet/nnUNet_results'))
    dataset_name = f"Dataset{dataset_id}_ISLES2022"
    
    for stage, cfg_name in [(1, "3d_lowres"), (2, "3d_cascade_fullres")]:
        print(f"\nStage {stage} ({cfg_name}):")
        print("-" * 40)
        
        for fold in range(5):
            trainer = config['cascade'][f'stage{stage}']['trainer']
            plans = config['cascade'][f'stage{stage}']['plans']
            
            fold_path = results_base / dataset_name / f"{trainer}__{plans}__{cfg_name}" / f"fold_{fold}"
            
            if (fold_path / "checkpoint_final.pth").exists():
                print_success(f"  Fold {fold}: Complete")
            elif (fold_path / "checkpoint_latest.pth").exists():
                print_warning(f"  Fold {fold}: In progress")
            else:
                print_info(f"  Fold {fold}: Not started")


def main():
    parser = argparse.ArgumentParser(
        description="Train Cascaded nnU-Net for tiny lesion improvement"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_cascade_config.yaml',
        help='Path to cascade configuration file'
    )
    
    # Actions
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run preprocessing for cascade configuration'
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        help='Stage to train (1=lowres, 2=cascade_fullres)'
    )
    parser.add_argument(
        '--fold', '-f',
        type=int,
        choices=[0, 1, 2, 3, 4],
        help='Fold to train'
    )
    parser.add_argument(
        '--train-all',
        action='store_true',
        help='Train all stages and folds sequentially'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check training status'
    )
    parser.add_argument(
        '--continue', '-cont',
        dest='continue_training',
        action='store_true',
        help='Continue from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    setup_nnunet_env(config)
    
    # Execute action
    if args.preprocess:
        run_preprocessing(config)
    elif args.status:
        check_status(config)
    elif args.train_all:
        train_all(config)
    elif args.stage and args.fold is not None:
        train_stage(config, args.stage, args.fold, args.continue_training)
    else:
        parser.print_help()
        print()
        print_info("Quick start:")
        print("  1. python scripts/22_train_cascade_nnunet.py --preprocess")
        print("  2. python scripts/22_train_cascade_nnunet.py --train-all")
        print()
        print("  Or use screen for long training:")
        print("  screen -S cascade")
        print("  python scripts/22_train_cascade_nnunet.py --train-all")
        print("  (Ctrl+A, D to detach)")


if __name__ == "__main__":
    main()
