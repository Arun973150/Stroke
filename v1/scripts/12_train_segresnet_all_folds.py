#!/usr/bin/env python3
"""
12. Train SegResNet All Folds
=============================
Sequential training of SegResNet across all 5 folds.

Usage:
    python scripts/12_train_segresnet_all_folds.py --config configs/segresnet_config.yaml
    
Train specific folds:
    python scripts/12_train_segresnet_all_folds.py --config configs/segresnet_config.yaml --folds 0 1 2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import importlib.util

def import_train_script(script_path):
    """Import the numbered training script dynamically."""
    spec = importlib.util.spec_from_file_location("train_segresnet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train SegResNet on all folds")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Folds to train (default: all 5)')
    parser.add_argument('--epochs', type=int, default=None, help='Override max epochs')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Results storage
    all_results = []
    
    print("=" * 60)
    print(f"SegResNet Multi-Fold Training")
    print(f"Folds: {args.folds}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    for fold in args.folds:
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold}")
        print(f"{'='*60}\n")
        
        try:
            # Dynamically import the numbered training script
            script_path = PROJECT_ROOT / "scripts" / "11_train_segresnet.py"
            train_module = import_train_script(str(script_path))
            train_fold = train_module.train_fold
            
            results = train_fold(
                config=config,
                fold=fold,
                max_epochs=args.epochs,
            )
            
            results['fold'] = fold
            results['status'] = 'success'
            all_results.append(results)
            
            print(f"\nFold {fold} completed: Dice = {results['best_dice']:.4f}")
            
        except Exception as e:
            print(f"\nFold {fold} FAILED: {str(e)}")
            all_results.append({
                'fold': fold,
                'status': 'failed',
                'error': str(e),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    successful = [r for r in all_results if r['status'] == 'success']
    
    if successful:
        dice_scores = [r['best_dice'] for r in successful]
        mean_dice = sum(dice_scores) / len(dice_scores)
        
        print(f"Successful folds: {len(successful)}/{len(args.folds)}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Best Dice: {max(dice_scores):.4f} (Fold {successful[dice_scores.index(max(dice_scores))]['fold']})")
        
        for r in all_results:
            if r['status'] == 'success':
                print(f"  Fold {r['fold']}: Dice = {r['best_dice']:.4f}")
            else:
                print(f"  Fold {r['fold']}: FAILED")
    else:
        print("All folds failed!")
    
    # Save results
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "all_folds_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
