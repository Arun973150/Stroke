#!/usr/bin/env python3
"""
Visualization Script - Overlay Stroke Predictions on MRI Slices
================================================================
Generates overlaid visualizations of predicted stroke lesions on original MRI.

Usage:
    # Visualize a single case
    python scripts/visualize_predictions.py --case case_0221 --output results/visualizations
    
    # Visualize all predictions in a folder
    python scripts/visualize_predictions.py --pred-dir trained_models/predictions_ensemble --output results/visualizations
    
    # Visualize specific slice
    python scripts/visualize_predictions.py --case case_0221 --slice 50
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_nifti(filepath: Path) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI file and return data with header info."""
    try:
        import nibabel as nib
        img = nib.load(filepath)
        data = img.get_fdata()
        header = {
            'shape': img.shape,
            'affine': img.affine,
            'spacing': img.header.get_zooms()
        }
        return data, header
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def normalize_intensity(image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99) -> np.ndarray:
    """Normalize image intensity to [0, 1] range with percentile clipping."""
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    
    image_clipped = np.clip(image, p_low, p_high)
    
    if p_high - p_low > 0:
        image_normalized = (image_clipped - p_low) / (p_high - p_low)
    else:
        image_normalized = np.zeros_like(image_clipped)
    
    return image_normalized


def create_overlay_image(
    mri_slice: np.ndarray,
    pred_slice: np.ndarray,
    gt_slice: Optional[np.ndarray] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """Create an overlay image with prediction (and optionally ground truth)."""
    # Normalize MRI to [0, 1]
    mri_norm = normalize_intensity(mri_slice)
    
    # Create RGB image from grayscale MRI
    rgb_image = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    
    # Add prediction overlay (red)
    pred_mask = pred_slice > 0
    if np.any(pred_mask):
        rgb_image[pred_mask, 0] = rgb_image[pred_mask, 0] * (1 - alpha) + 1.0 * alpha  # Red
        rgb_image[pred_mask, 1] = rgb_image[pred_mask, 1] * (1 - alpha) + 0.2 * alpha
        rgb_image[pred_mask, 2] = rgb_image[pred_mask, 2] * (1 - alpha) + 0.2 * alpha
    
    # Add ground truth overlay (green) if provided
    if gt_slice is not None:
        gt_mask = gt_slice > 0
        if np.any(gt_mask):
            # Overlap (yellow) where both pred and gt
            overlap = pred_mask & gt_mask
            only_gt = gt_mask & ~pred_mask
            
            rgb_image[only_gt, 0] = rgb_image[only_gt, 0] * (1 - alpha) + 0.2 * alpha
            rgb_image[only_gt, 1] = rgb_image[only_gt, 1] * (1 - alpha) + 1.0 * alpha  # Green
            rgb_image[only_gt, 2] = rgb_image[only_gt, 2] * (1 - alpha) + 0.2 * alpha
            
            # Yellow for overlap
            rgb_image[overlap, 0] = rgb_image[overlap, 0] * (1 - alpha) + 1.0 * alpha
            rgb_image[overlap, 1] = rgb_image[overlap, 1] * (1 - alpha) + 1.0 * alpha
            rgb_image[overlap, 2] = rgb_image[overlap, 2] * (1 - alpha) + 0.0 * alpha
    
    return np.clip(rgb_image, 0, 1)


def find_best_slices(volume: np.ndarray, n_slices: int = 5) -> List[int]:
    """Find slices with the most lesion content."""
    if volume.ndim != 3:
        return [volume.shape[-1] // 2]
    
    # Sum over each axial slice
    slice_sums = []
    for i in range(volume.shape[2]):
        slice_sums.append(np.sum(volume[:, :, i] > 0))
    
    # Get indices of top N slices
    if max(slice_sums) == 0:
        # No lesion found, return middle slices
        mid = volume.shape[2] // 2
        return list(range(max(0, mid - n_slices // 2), min(volume.shape[2], mid + n_slices // 2 + 1)))
    
    sorted_indices = np.argsort(slice_sums)[::-1]
    return sorted(sorted_indices[:n_slices])


def visualize_case(
    case_id: str,
    pred_path: Path,
    mri_dir: Optional[Path] = None,
    gt_dir: Optional[Path] = None,
    output_dir: Path = Path('results/visualizations'),
    specific_slice: Optional[int] = None,
    n_slices: int = 5,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """Visualize a single case with prediction overlay."""
    
    print(f"\n{'='*50}")
    print(f"Visualizing: {case_id}")
    print(f"{'='*50}")
    
    # Load prediction
    pred_data, pred_header = load_nifti(pred_path)
    if pred_data is None:
        print(f"Failed to load prediction: {pred_path}")
        return
    
    print(f"Prediction shape: {pred_data.shape}")
    
    # Load MRI if available
    mri_data = None
    if mri_dir and mri_dir.exists():
        # Try different naming conventions
        mri_patterns = [
            mri_dir / f"{case_id}_0000.nii.gz",  # nnU-Net format (DWI)
            mri_dir / f"{case_id}.nii.gz",
        ]
        for mri_path in mri_patterns:
            if mri_path.exists():
                mri_data, _ = load_nifti(mri_path)
                print(f"Loaded MRI: {mri_path.name}")
                break
    
    if mri_data is None:
        print("Warning: No MRI found, using prediction intensity for background")
        mri_data = np.zeros_like(pred_data)
    
    # Load ground truth if available
    gt_data = None
    if gt_dir and gt_dir.exists():
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if gt_path.exists():
            gt_data, _ = load_nifti(gt_path)
            print(f"Loaded ground truth: {gt_path.name}")
    
    # Determine slices to visualize
    if specific_slice is not None:
        slices = [specific_slice]
    else:
        # Find slices with most lesion content
        search_vol = gt_data if gt_data is not None else pred_data
        slices = find_best_slices(search_vol, n_slices)
    
    print(f"Visualizing slices: {slices}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    n_cols = len(slices)
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0], figsize[1]))
    if n_cols == 1:
        axes = [axes]
    
    fig.suptitle(f'{case_id} - Stroke Lesion Segmentation', fontsize=14, fontweight='bold')
    
    for idx, slice_idx in enumerate(slices):
        ax = axes[idx]
        
        # Get slices
        mri_slice = mri_data[:, :, slice_idx] if mri_data.ndim == 3 else mri_data
        pred_slice = pred_data[:, :, slice_idx] if pred_data.ndim == 3 else pred_data
        gt_slice = gt_data[:, :, slice_idx] if gt_data is not None and gt_data.ndim == 3 else None
        
        # Create overlay
        overlay = create_overlay_image(mri_slice, pred_slice, gt_slice)
        
        # Display
        ax.imshow(np.rot90(overlay), aspect='auto')
        ax.set_title(f'Slice {slice_idx}', fontsize=10)
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.6, label='Prediction'),
    ]
    if gt_data is not None:
        legend_elements.extend([
            mpatches.Patch(facecolor='green', alpha=0.6, label='Ground Truth'),
            mpatches.Patch(facecolor='yellow', alpha=0.6, label='Overlap (TP)')
        ])
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    output_path = output_dir / f"{case_id}_overlay.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def visualize_grid(
    case_id: str,
    pred_path: Path,
    mri_dir: Optional[Path] = None,
    gt_dir: Optional[Path] = None,
    output_dir: Path = Path('results/visualizations'),
    n_slices: int = 9
) -> None:
    """Create a 3x3 grid visualization of multiple slices."""
    
    print(f"\n{'='*50}")
    print(f"Creating grid visualization: {case_id}")
    print(f"{'='*50}")
    
    # Load volumes
    pred_data, _ = load_nifti(pred_path)
    if pred_data is None:
        return
    
    mri_data = None
    if mri_dir and mri_dir.exists():
        mri_path = mri_dir / f"{case_id}_0000.nii.gz"
        if mri_path.exists():
            mri_data, _ = load_nifti(mri_path)
    
    gt_data = None
    if gt_dir and gt_dir.exists():
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if gt_path.exists():
            gt_data, _ = load_nifti(gt_path)
    
    if mri_data is None:
        mri_data = np.zeros_like(pred_data)
    
    # Get evenly spaced slices through the volume
    total_slices = pred_data.shape[2]
    slice_indices = np.linspace(total_slices * 0.2, total_slices * 0.8, n_slices, dtype=int)
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'{case_id} - Multi-Slice View', fontsize=14, fontweight='bold')
    
    for idx, slice_idx in enumerate(slice_indices):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        mri_slice = mri_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        gt_slice = gt_data[:, :, slice_idx] if gt_data is not None else None
        
        overlay = create_overlay_image(mri_slice, pred_slice, gt_slice)
        
        ax.imshow(np.rot90(overlay), aspect='auto')
        ax.set_title(f'Slice {slice_idx}', fontsize=9)
        ax.axis('off')
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor='red', alpha=0.6, label='Prediction')]
    if gt_data is not None:
        legend_elements.extend([
            mpatches.Patch(facecolor='green', alpha=0.6, label='Ground Truth'),
            mpatches.Patch(facecolor='yellow', alpha=0.6, label='Overlap')
        ])
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize stroke segmentation predictions")
    parser.add_argument('--case', type=str, help='Specific case ID to visualize (e.g., case_0221)')
    parser.add_argument('--pred-dir', type=str, default='trained_models/predictions_ensemble',
                       help='Directory containing prediction .nii.gz files')
    parser.add_argument('--mri-dir', type=str, help='Directory containing original MRI images')
    parser.add_argument('--gt-dir', type=str, help='Directory containing ground truth labels')
    parser.add_argument('--output', type=str, default='results/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--slice', type=int, help='Specific slice index to visualize')
    parser.add_argument('--n-slices', type=int, default=5, help='Number of slices to visualize')
    parser.add_argument('--grid', action='store_true', help='Create 3x3 grid visualization')
    parser.add_argument('--all', action='store_true', help='Visualize all cases in pred-dir')
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    mri_dir = Path(args.mri_dir) if args.mri_dir else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    
    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {pred_dir}")
        print("Tip: Unzip predictions_ensemble.zip first")
        return
    
    # Get list of cases to visualize
    if args.case:
        cases = [args.case]
    elif args.all:
        cases = [f.stem.replace('.nii', '') for f in pred_dir.glob('*.nii.gz')]
        cases = sorted(set(cases))
    else:
        # Default: visualize first 5 cases
        cases = [f.stem.replace('.nii', '') for f in sorted(pred_dir.glob('*.nii.gz'))[:5]]
    
    print(f"\nFound {len(cases)} case(s) to visualize")
    print(f"Output directory: {output_dir}")
    
    for case_id in cases:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            print(f"Prediction not found: {pred_path}")
            continue
        
        if args.grid:
            visualize_grid(case_id, pred_path, mri_dir, gt_dir, output_dir)
        else:
            visualize_case(
                case_id, pred_path, mri_dir, gt_dir, output_dir,
                specific_slice=args.slice, n_slices=args.n_slices
            )
    
    print(f"\n✅ Visualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
