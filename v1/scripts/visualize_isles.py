#!/usr/bin/env python3
"""
ISLES-2022 Visualization Script - Overlay Predictions on MRI
=============================================================
Creates publication-quality overlays of stroke lesion predictions on DWI MRI.

Usage:
    # Visualize a single case with MRI background
    python scripts/visualize_isles.py --case 0221
    
    # Grid view of a case 
    python scripts/visualize_isles.py --case 0240 --grid
    
    # Visualize multiple best/worst cases
    python scripts/visualize_isles.py --best 5
    python scripts/visualize_isles.py --worst 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resample_volume(volume: np.ndarray, target_shape: Tuple[int, ...], order: int = 1) -> np.ndarray:
    """Resample volume to target shape using scipy zoom."""
    if volume.shape == target_shape:
        return volume
    
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

# Default paths
ISLES_DIR = PROJECT_ROOT / "ISLES-2022" / "ISLES-2022"
PRED_DIR = PROJECT_ROOT / "trained_models" / "predictions_ensemble"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"


def load_nifti(filepath: Path) -> Optional[np.ndarray]:
    """Load a NIfTI file and return data."""
    try:
        import nibabel as nib
        img = nib.load(filepath)
        return img.get_fdata()
    except Exception as e:
        print(f"  Warning: Could not load {filepath.name}: {e}")
        return None


def get_isles_paths(case_num: str, isles_dir: Path) -> Dict[str, Optional[Path]]:
    """Get paths to DWI, ADC and ground truth mask for a case."""
    subject = f"sub-strokecase{case_num}"
    
    # DWI path
    dwi_path = isles_dir / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_dwi.nii.gz"
    
    # ADC path  
    adc_path = isles_dir / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_adc.nii.gz"
    
    # Ground truth mask
    gt_path = isles_dir / "derivatives" / subject / "ses-0001" / f"{subject}_ses-0001_msk.nii.gz"
    
    return {
        'dwi': dwi_path if dwi_path.exists() else None,
        'adc': adc_path if adc_path.exists() else None,
        'gt': gt_path if gt_path.exists() else None
    }


def normalize_intensity(image: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image intensity to [0, 1]."""
    low = np.percentile(image, p_low)
    high = np.percentile(image, p_high)
    if high - low > 0:
        return np.clip((image - low) / (high - low), 0, 1)
    return np.zeros_like(image)


def create_overlay(
    mri_slice: np.ndarray,
    pred_slice: np.ndarray,
    gt_slice: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """Create RGB overlay with prediction and optional ground truth."""
    mri_norm = normalize_intensity(mri_slice)
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    
    # Prediction overlay (red)
    pred_mask = pred_slice > 0
    if np.any(pred_mask):
        rgb[pred_mask, 0] = np.clip(rgb[pred_mask, 0] * (1-alpha) + 1.0 * alpha, 0, 1)
        rgb[pred_mask, 1] = np.clip(rgb[pred_mask, 1] * (1-alpha) + 0.2 * alpha, 0, 1)
        rgb[pred_mask, 2] = np.clip(rgb[pred_mask, 2] * (1-alpha) + 0.2 * alpha, 0, 1)
    
    # Ground truth overlay (green for FN, yellow for TP)
    if gt_slice is not None:
        gt_mask = gt_slice > 0
        overlap = pred_mask & gt_mask  # True Positive (yellow)
        fn_mask = gt_mask & ~pred_mask  # False Negative (green)
        
        if np.any(overlap):
            rgb[overlap, 0] = np.clip(rgb[overlap, 0] * (1-alpha) + 1.0 * alpha, 0, 1)
            rgb[overlap, 1] = np.clip(rgb[overlap, 1] * (1-alpha) + 0.9 * alpha, 0, 1)
            rgb[overlap, 2] = np.clip(rgb[overlap, 2] * (1-alpha) + 0.0 * alpha, 0, 1)
        
        if np.any(fn_mask):
            rgb[fn_mask, 0] = np.clip(rgb[fn_mask, 0] * (1-alpha) + 0.2 * alpha, 0, 1)
            rgb[fn_mask, 1] = np.clip(rgb[fn_mask, 1] * (1-alpha) + 1.0 * alpha, 0, 1)
            rgb[fn_mask, 2] = np.clip(rgb[fn_mask, 2] * (1-alpha) + 0.2 * alpha, 0, 1)
    
    return rgb


def find_lesion_slices(volume: np.ndarray, n: int = 5) -> List[int]:
    """Find slices with most lesion content."""
    if volume is None or volume.ndim != 3:
        return []
    
    sums = [np.sum(volume[:, :, i] > 0) for i in range(volume.shape[2])]
    if max(sums) == 0:
        mid = volume.shape[2] // 2
        return list(range(max(0, mid-n//2), min(volume.shape[2], mid+n//2+1)))
    
    return sorted(np.argsort(sums)[::-1][:n])


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    if union == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0
    return 2.0 * intersection / union


def visualize_case(
    case_num: str,
    isles_dir: Path = ISLES_DIR,
    pred_dir: Path = PRED_DIR,
    output_dir: Path = OUTPUT_DIR,
    n_slices: int = 5,
    grid: bool = False
) -> Optional[Path]:
    """Visualize a single case with MRI background."""
    
    case_id = f"case_{case_num}"
    print(f"\n{'='*60}")
    print(f"  Visualizing: {case_id}")
    print(f"{'='*60}")
    
    # Load prediction
    pred_path = pred_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        print(f"  Error: Prediction not found: {pred_path}")
        return None
    
    pred_data = load_nifti(pred_path)
    if pred_data is None:
        return None
    print(f"  Prediction shape: {pred_data.shape}")
    
    # Get ISLES paths
    paths = get_isles_paths(case_num, isles_dir)
    
    # Load DWI
    dwi_data = load_nifti(paths['dwi']) if paths['dwi'] else None
    if dwi_data is not None:
        print(f"  Loaded DWI: {paths['dwi'].name}")
    else:
        print(f"  Warning: No DWI found, using blank background")
        dwi_data = np.zeros_like(pred_data)
    
    # Load ground truth
    gt_data = load_nifti(paths['gt']) if paths['gt'] else None
    if gt_data is not None:
        print(f"  Loaded GT mask: {paths['gt'].name}")
    
    # Handle shape mismatch by resampling to prediction shape
    if dwi_data.shape != pred_data.shape:
        print(f"  Resampling DWI from {dwi_data.shape} to {pred_data.shape}")
        dwi_data = resample_volume(dwi_data, pred_data.shape, order=1)
    
    if gt_data is not None and gt_data.shape != pred_data.shape:
        print(f"  Resampling GT from {gt_data.shape} to {pred_data.shape}")
        gt_data = resample_volume(gt_data, pred_data.shape, order=0)  # Nearest neighbor for masks
    
    # Compute Dice AFTER resampling
    dice = None
    if gt_data is not None:
        dice = compute_dice(pred_data, gt_data)
        print(f"  Dice Score: {dice:.4f} ({dice*100:.1f}%)")
    
    # Find slices to visualize
    search_vol = gt_data if gt_data is not None else pred_data
    slices = find_lesion_slices(search_vol, n_slices if not grid else 9)
    print(f"  Visualizing slices: {slices}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if grid:
        # Create 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'{case_id} - Stroke Lesion Segmentation\nDice: {dice*100:.1f}%' if gt_data is not None else f'{case_id} - Stroke Lesion Segmentation', 
                     fontsize=14, fontweight='bold')
        
        for idx in range(9):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if idx < len(slices):
                s = slices[idx]
                overlay = create_overlay(
                    dwi_data[:, :, s],
                    pred_data[:, :, s],
                    gt_data[:, :, s] if gt_data is not None else None
                )
                ax.imshow(np.rot90(overlay), aspect='auto')
                ax.set_title(f'Slice {s}', fontsize=10)
            ax.axis('off')
        
        output_file = output_dir / f"{case_id}_grid.png"
    else:
        # Single row of slices
        n_cols = len(slices)
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3, 4))
        if n_cols == 1:
            axes = [axes]
        
        title = f'{case_id} - Stroke Lesion Segmentation'
        if gt_data is not None:
            title += f'\nDice: {dice*100:.1f}%'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        for idx, s in enumerate(slices):
            ax = axes[idx]
            overlay = create_overlay(
                dwi_data[:, :, s],
                pred_data[:, :, s],
                gt_data[:, :, s] if gt_data is not None else None
            )
            ax.imshow(np.rot90(overlay), aspect='auto')
            ax.set_title(f'Slice {s}', fontsize=10)
            ax.axis('off')
        
        output_file = output_dir / f"{case_id}_overlay.png"
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Prediction (FP)'),
    ]
    if gt_data is not None:
        legend_elements.extend([
            mpatches.Patch(facecolor='yellow', alpha=0.7, edgecolor='black', label='True Positive'),
            mpatches.Patch(facecolor='green', alpha=0.7, edgecolor='black', label='False Negative (missed)'),
        ])
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12 if not grid else 0.08)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file}")
    return output_file


def load_summary(pred_dir: Path) -> Dict:
    """Load evaluation summary JSON."""
    summary_path = pred_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="ISLES-2022 Visualization with MRI Overlay")
    parser.add_argument('--case', type=str, help='Case number to visualize (e.g., 0221)')
    parser.add_argument('--isles-dir', type=str, default=str(ISLES_DIR), help='ISLES-2022 data directory')
    parser.add_argument('--pred-dir', type=str, default=str(PRED_DIR), help='Predictions directory')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')
    parser.add_argument('--n-slices', type=int, default=5, help='Number of slices')
    parser.add_argument('--grid', action='store_true', help='Create 3x3 grid')
    parser.add_argument('--best', type=int, help='Visualize N best Dice cases')
    parser.add_argument('--worst', type=int, help='Visualize N worst Dice cases')
    parser.add_argument('--all', action='store_true', help='Visualize all cases')
    
    args = parser.parse_args()
    
    isles_dir = Path(args.isles_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    
    if not pred_dir.exists():
        print(f"Error: Predictions directory not found: {pred_dir}")
        print("Tip: Run 'Expand-Archive -Path trained_models\\predictions_ensemble.zip -DestinationPath trained_models'")
        return
    
    # Determine cases to visualize
    if args.case:
        cases = [args.case]
    elif args.best or args.worst:
        summary = load_summary(pred_dir)
        if 'metric_per_case' in summary:
            case_dice = []
            for item in summary['metric_per_case']:
                case_file = Path(item['prediction_file']).stem.replace('.nii', '')
                dice = item['metrics']['1'].get('Dice', 0)
                if not np.isnan(dice):
                    case_num = case_file.replace('case_', '')
                    case_dice.append((case_num, dice))
            
            case_dice.sort(key=lambda x: x[1], reverse=True)
            
            if args.best:
                cases = [c[0] for c in case_dice[:args.best]]
                print(f"Best {args.best} cases by Dice:")
            else:
                cases = [c[0] for c in case_dice[-args.worst:]]
                print(f"Worst {args.worst} cases by Dice:")
            
            for c, d in (case_dice[:args.best] if args.best else case_dice[-args.worst:]):
                print(f"  case_{c}: {d*100:.1f}%")
        else:
            print("No summary.json found, using first 5 cases")
            cases = ['0213', '0214', '0215', '0216', '0217']
    elif args.all:
        cases = [f.stem.replace('case_', '').replace('.nii', '') 
                 for f in sorted(pred_dir.glob('case_*.nii.gz'))]
    else:
        # Default: first 5 test cases
        cases = ['0221', '0240', '0246', '0237', '0213']
    
    print(f"\nProcessing {len(cases)} case(s)...")
    print(f"ISLES data: {isles_dir}")
    print(f"Predictions: {pred_dir}")
    print(f"Output: {output_dir}")
    
    for case_num in cases:
        visualize_case(case_num, isles_dir, pred_dir, output_dir, args.n_slices, args.grid)
    
    print(f"\n✅ Visualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
