#!/usr/bin/env python3
"""
03. nnU-Net Planning and Preprocessing Script
==============================================
Runs nnU-Net's automatic dataset analysis, planning, and preprocessing.

This script:
1. Verifies dataset integrity
2. Analyzes dataset properties (spacing, intensity, etc.)
3. Determines optimal network configuration
4. Preprocesses all data for training

Usage:
    python scripts/03_plan_and_preprocess.py --config configs/nnunet_config.yaml
    python scripts/03_plan_and_preprocess.py --dataset-id 001

Note: This is typically run once before training.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def run_nnunet_plan_and_preprocess(config: dict) -> bool:
    """
    Run nnU-Net planning and preprocessing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful
    """
    print_header("nnU-Net Planning & Preprocessing")
    
    dataset_id = config['dataset']['id']
    dataset_name = get_dataset_name(config)
    
    print_info(f"Dataset ID: {dataset_id}")
    print_info(f"Dataset Name: {dataset_name}")
    
    # Verify environment is set up
    required_env_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for var in required_env_vars:
        if var not in os.environ:
            print_error(f"Environment variable {var} not set!")
            print_info("Run setup_environment.sh first or call setup_nnunet_env()")
            return False
        print_success(f"{var}: {os.environ[var]}")
    
    # Verify dataset exists
    dataset_path = Path(os.environ['nnUNet_raw']) / dataset_name
    if not dataset_path.exists():
        print_error(f"Dataset not found: {dataset_path}")
        print_info("Run 02_setup_nnunet_dataset.py first")
        return False
    
    # ---------------------------------------------------------------------
    # Step 1: Run planning and preprocessing
    # ---------------------------------------------------------------------
    print_step(1, 2, "Running nnU-Net planning and preprocessing...")
    print_info("This may take 30-60 minutes depending on dataset size...")
    print()
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        "--verify_dataset_integrity"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print_success("Planning and preprocessing complete!")
    except subprocess.CalledProcessError as e:
        print_error(f"Planning failed with error: {e}")
        return False
    except FileNotFoundError:
        print_error("nnUNetv2_plan_and_preprocess command not found!")
        print_info("Make sure nnU-Net v2 is installed: pip install nnunetv2")
        return False
    
    # ---------------------------------------------------------------------
    # Step 2: Display configuration
    # ---------------------------------------------------------------------
    print_step(2, 2, "Reviewing generated configuration...")
    
    display_nnunet_plans(config)
    
    print_header("Planning Complete")
    print(f"\nNext step: Start training:")
    print(f"  python scripts/04_train_nnunet.py --fold 0 --config configs/nnunet_config.yaml")
    print(f"\nOr train all folds:")
    print(f"  python scripts/05_train_all_folds.py --config configs/nnunet_config.yaml")
    
    return True


def display_nnunet_plans(config: dict) -> None:
    """
    Display nnU-Net plans configuration.
    
    Args:
        config: Configuration dictionary
    """
    dataset_name = get_dataset_name(config)
    plans_file = Path(os.environ['nnUNet_preprocessed']) / dataset_name / "nnUNetPlans.json"
    
    print("\n" + "-" * 50)
    print("nnU-Net Configuration")
    print("-" * 50)
    
    if not plans_file.exists():
        print_warning(f"Plans file not found: {plans_file}")
        return
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    print(f"\nAvailable configurations: {list(plans.get('configurations', {}).keys())}")
    
    if '3d_fullres' in plans.get('configurations', {}):
        cfg = plans['configurations']['3d_fullres']
        arch = cfg.get('architecture', {})
        
        print(f"\n3D Full Resolution Settings:")
        print(f"  Patch size: {cfg.get('patch_size', 'N/A')}")
        print(f"  Batch size: {cfg.get('batch_size', 'N/A')}")
        print(f"  Network class: {arch.get('network_class_name', 'N/A')}")
        
        arch_kwargs = arch.get('arch_kwargs', {})
        print(f"  Conv type: {arch_kwargs.get('conv_op', 'N/A')}")
        print(f"  Features per stage: {arch_kwargs.get('features_per_stage', 'N/A')}")
        print(f"  Number of stages: {arch_kwargs.get('n_stages', 'N/A')}")
        
        batch_size = cfg.get('batch_size', 0)
        if batch_size == 1:
            print_warning(f"Batch size is 1 (memory constrained)")
        else:
            print_success(f"Batch size is {batch_size}")
    
    # Display fingerprint info
    fingerprint_file = Path(os.environ['nnUNet_preprocessed']) / dataset_name / "dataset_fingerprint.json"
    
    if fingerprint_file.exists():
        with open(fingerprint_file, 'r') as f:
            fingerprint = json.load(f)
        
        print(f"\nDataset Fingerprint:")
        print(f"  Median image size: {fingerprint.get('median_image_size_in_voxels', 'N/A')}")
        print(f"  Median spacing: {fingerprint.get('median_spacing', 'N/A')}")
        print(f"  Foreground intensity mean: {fingerprint.get('foreground_intensity_properties_per_channel', {}).get('0', {}).get('mean', 'N/A')}")
    
    print_success("Plans look good!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run nnU-Net planning and preprocessing"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset-id',
        type=str,
        help='Override dataset ID'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override dataset ID if provided
    if args.dataset_id:
        config['dataset']['id'] = args.dataset_id
    
    # Set up environment variables
    setup_nnunet_env(config)
    
    # Run planning
    success = run_nnunet_plan_and_preprocess(config)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
