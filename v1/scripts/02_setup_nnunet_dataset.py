#!/usr/bin/env python3
"""
02. nnU-Net Dataset Setup Script
================================
Organizes preprocessed data into nnU-Net v2 expected structure.

This script:
1. Copies preprocessed data to nnUNet_raw
2. Creates proper Dataset folder structure
3. Validates dataset integrity
4. Verifies dataset.json configuration

Usage:
    python scripts/02_setup_nnunet_dataset.py --config configs/nnunet_config.yaml
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def setup_nnunet_dataset(config: dict) -> None:
    """
    Set up nnU-Net dataset structure from preprocessed data.
    
    Args:
        config: Configuration dictionary
    """
    print_header("nnU-Net Dataset Setup")
    
    # Get paths
    preprocessed_path = Path(config['paths']['preprocessed']) / 'nnunet_format'
    nnunet_raw = Path(config['paths']['nnunet_raw'])
    
    dataset_name = get_dataset_name(config)
    dataset_path = nnunet_raw / dataset_name
    
    print_info(f"Source: {preprocessed_path}")
    print_info(f"Target: {dataset_path}")
    print_info(f"Dataset: {dataset_name}")
    
    # Verify source exists
    if not preprocessed_path.exists():
        print_error(f"Preprocessed data not found: {preprocessed_path}")
        print_info("Run 01_preprocess_isles.py first")
        return
    
    # ---------------------------------------------------------------------
    # Step 1: Create dataset directory
    # ---------------------------------------------------------------------
    print_step(1, 4, "Creating dataset directory...")
    
    dataset_path.mkdir(parents=True, exist_ok=True)
    print_success(f"Created {dataset_path}")
    
    # ---------------------------------------------------------------------
    # Step 2: Copy data folders
    # ---------------------------------------------------------------------
    print_step(2, 4, "Copying data folders...")
    
    folders_to_copy = ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']
    
    for folder in folders_to_copy:
        src = preprocessed_path / folder
        dst = dataset_path / folder
        
        if src.exists():
            # Remove existing if present
            if dst.exists():
                shutil.rmtree(dst)
            
            shutil.copytree(src, dst)
            file_count = len(list(dst.glob('*.nii.gz')))
            print_success(f"{folder}: {file_count} files copied")
        else:
            print_warning(f"{folder} not found in source (may be empty)")
    
    # ---------------------------------------------------------------------
    # Step 3: Copy dataset.json
    # ---------------------------------------------------------------------
    print_step(3, 4, "Setting up dataset.json...")
    
    dataset_json_src = preprocessed_path / 'dataset.json'
    dataset_json_dst = dataset_path / 'dataset.json'
    
    if dataset_json_src.exists():
        shutil.copy(dataset_json_src, dataset_json_dst)
        print_success("dataset.json copied")
    else:
        print_error("dataset.json not found!")
        print_info("Creating default dataset.json...")
        
        # Create default
        dataset_info = {
            "channel_names": {
                "0": "DWI",
                "1": "ADC",
                "2": "FLAIR"
            },
            "labels": {
                "background": 0,
                "lesion": 1
            },
            "numTraining": 0,
            "numTest": 0,
            "file_ending": ".nii.gz"
        }
        
        with open(dataset_json_dst, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    # ---------------------------------------------------------------------
    # Step 4: Verify dataset
    # ---------------------------------------------------------------------
    print_step(4, 4, "Verifying dataset...")
    
    verify_nnunet_dataset(dataset_path, dataset_json_dst)
    
    print_header("Setup Complete")
    print(f"\nDataset ready at: {dataset_path}")
    print(f"\nNext step: Run nnU-Net planning and preprocessing:")
    print(f"  python scripts/03_plan_and_preprocess.py --config configs/nnunet_config.yaml")


def verify_nnunet_dataset(dataset_path: Path, dataset_json_path: Path) -> bool:
    """
    Verify nnU-Net dataset structure and integrity.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_json_path: Path to dataset.json
        
    Returns:
        True if verification passed
    """
    print("\n" + "-" * 50)
    print("Dataset Verification")
    print("-" * 50)
    
    imagesTr_path = dataset_path / 'imagesTr'
    labelsTr_path = dataset_path / 'labelsTr'
    imagesTs_path = dataset_path / 'imagesTs'
    
    # Count training cases
    train_cases = set()
    if imagesTr_path.exists():
        for f in imagesTr_path.glob('*_0000.nii.gz'):
            case_id = f.name.replace('_0000.nii.gz', '')
            train_cases.add(case_id)
    
    # Count test cases
    test_cases = set()
    if imagesTs_path.exists():
        for f in imagesTs_path.glob('*_0000.nii.gz'):
            case_id = f.name.replace('_0000.nii.gz', '')
            test_cases.add(case_id)
    
    print(f"\nCase counts:")
    print(f"  Training: {len(train_cases)} cases")
    print(f"  Test: {len(test_cases)} cases")
    print(f"  Total: {len(train_cases) + len(test_cases)} cases")
    
    # Verify modality completeness
    print(f"\nVerifying modality completeness (first 3 cases):")
    all_complete = True
    
    for case_id in sorted(train_cases)[:3]:
        dwi = (imagesTr_path / f'{case_id}_0000.nii.gz').exists()
        adc = (imagesTr_path / f'{case_id}_0001.nii.gz').exists()
        flair = (imagesTr_path / f'{case_id}_0002.nii.gz').exists()
        label = (labelsTr_path / f'{case_id}.nii.gz').exists()
        
        complete = all([dwi, adc, flair, label])
        status = "✓" if complete else "✗"
        print(f"  {status} {case_id}: DWI={dwi}, ADC={adc}, FLAIR={flair}, Label={label}")
        
        if not complete:
            all_complete = False
    
    # Verify dataset.json
    print(f"\nVerifying dataset.json:")
    
    with open(dataset_json_path, 'r') as f:
        ds_json = json.load(f)
    
    print(f"  numTraining: {ds_json.get('numTraining', 'MISSING')} (found: {len(train_cases)})")
    print(f"  numTest: {ds_json.get('numTest', 'MISSING')} (found: {len(test_cases)})")
    print(f"  file_ending: {ds_json.get('file_ending', 'MISSING')}")
    
    # Update if counts don't match
    needs_update = False
    if ds_json.get('numTraining') != len(train_cases):
        print_warning("Training count mismatch - updating...")
        ds_json['numTraining'] = len(train_cases)
        needs_update = True
    
    if ds_json.get('numTest') != len(test_cases):
        print_warning("Test count mismatch - updating...")
        ds_json['numTest'] = len(test_cases)
        needs_update = True
    
    if needs_update:
        with open(dataset_json_path, 'w') as f:
            json.dump(ds_json, f, indent=2)
        print_success("dataset.json updated")
    else:
        print_success("dataset.json is correct")
    
    if all_complete and len(train_cases) > 0:
        print_success("\nDataset verification passed!")
        return True
    else:
        print_error("\nDataset verification failed!")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up nnU-Net dataset structure"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment variables
    setup_nnunet_env(config)
    
    # Run setup
    setup_nnunet_dataset(config)


if __name__ == "__main__":
    main()
