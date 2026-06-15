#!/usr/bin/env python3
"""
Lambda Labs - Full Prediction Pipeline
=======================================
Runs inference on NEW test data using all trained models + fusion.

DATA FORMAT:
  Your test NIfTI files should be placed in ONE of these formats:

  FORMAT 1 - Raw ISLES format (separate folders per patient):
    test_data/
    ├── sub-strokecase0001/
    │   └── ses-0001/
    │       ├── sub-strokecase0001_ses-0001_dwi.nii.gz
    │       ├── sub-strokecase0001_ses-0001_adc.nii.gz
    │       └── sub-strokecase0001_ses-0001_flair.nii.gz
    └── sub-strokecase0002/
        └── ...

  FORMAT 2 - nnU-Net format (already organized):
    test_data/
    ├── case_0001_0000.nii.gz   (DWI  = channel 0)
    ├── case_0001_0001.nii.gz   (ADC  = channel 1)
    ├── case_0001_0002.nii.gz   (FLAIR = channel 2)
    ├── case_0002_0000.nii.gz
    └── ...

  FORMAT 3 - Simple triplets (DWI/ADC/FLAIR named per case):
    test_data/
    ├── patient001_DWI.nii.gz
    ├── patient001_ADC.nii.gz
    ├── patient001_FLAIR.nii.gz
    ├── patient002_DWI.nii.gz
    └── ...

USAGE:
    # Format 1 (ISLES raw)
    python lambda_predict.py --input-dir /path/to/test_data --format isles

    # Format 2 (nnU-Net)
    python lambda_predict.py --input-dir /path/to/test_data --format nnunet

    # Format 3 (simple triplets)
    python lambda_predict.py --input-dir /path/to/test_data --format triplet

    # Single case
    python lambda_predict.py --dwi scan_dwi.nii.gz --adc scan_adc.nii.gz --flair scan_flair.nii.gz

    # With ground truth for evaluation
    python lambda_predict.py --input-dir /path/to/test_data --format nnunet --gt-dir /path/to/labels

    # Skip nnU-Net (faster, uses 10 models instead of 15)
    python lambda_predict.py --input-dir /path/to/test_data --format nnunet --no-nnunet
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# MONAI imports
from monai.inferers import sliding_window_inference

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segresnet import SegResNetWrapper
from src.models.swin_unetr import SwinUNETRWrapper
from src.models.fusion_network import AdaptiveFusionNetwork

try:
    from src.models.nnunet_wrapper import nnUNetWrapper
    HAS_NNUNET = True
except ImportError:
    HAS_NNUNET = False
    print("[INFO] nnU-Net wrapper not available, will skip nnU-Net folds")


# ═══════════════════════════════════════════════════
#  DATA DISCOVERY - Find your test scans
# ═══════════════════════════════════════════════════

def discover_isles_format(input_dir: Path) -> List[Dict[str, Path]]:
    """Discover test cases in ISLES raw format (sub-strokecaseXXXX/ses-0001/)."""
    cases = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith('sub-'):
            continue
        ses_dir = subdir / 'ses-0001'
        if not ses_dir.exists():
            ses_dir = subdir  # Try flat structure

        dwi = list(ses_dir.glob('*dwi*')) or list(ses_dir.glob('*DWI*'))
        adc = list(ses_dir.glob('*adc*')) or list(ses_dir.glob('*ADC*'))
        flair = list(ses_dir.glob('*flair*')) or list(ses_dir.glob('*FLAIR*'))

        if dwi and adc and flair:
            cases.append({
                'case_id': subdir.name,
                'dwi': dwi[0],
                'adc': adc[0],
                'flair': flair[0],
            })
    return cases


def discover_nnunet_format(input_dir: Path) -> List[Dict[str, Path]]:
    """Discover test cases in nnU-Net format (case_XXXX_000{0,1,2}.nii.gz)."""
    cases = []
    dwi_files = sorted(input_dir.glob('*_0000.nii.gz'))

    for dwi_path in dwi_files:
        base = dwi_path.name.replace('_0000.nii.gz', '')
        adc_path = input_dir / f'{base}_0001.nii.gz'
        flair_path = input_dir / f'{base}_0002.nii.gz'

        if adc_path.exists() and flair_path.exists():
            cases.append({
                'case_id': base,
                'dwi': dwi_path,
                'adc': adc_path,
                'flair': flair_path,
            })
    return cases


def discover_triplet_format(input_dir: Path) -> List[Dict[str, Path]]:
    """Discover test cases as simple triplets (*_DWI.nii.gz, *_ADC.nii.gz, *_FLAIR.nii.gz)."""
    cases = []
    dwi_files = sorted(list(input_dir.glob('*DWI*')) + list(input_dir.glob('*dwi*')))

    for dwi_path in dwi_files:
        name = dwi_path.stem.replace('.nii', '')
        # Try to find matching ADC and FLAIR
        base = name.replace('_DWI', '').replace('_dwi', '').replace('DWI', '').replace('dwi', '')

        adc_candidates = list(input_dir.glob(f'{base}*ADC*')) + list(input_dir.glob(f'{base}*adc*'))
        flair_candidates = list(input_dir.glob(f'{base}*FLAIR*')) + list(input_dir.glob(f'{base}*flair*'))

        if adc_candidates and flair_candidates:
            cases.append({
                'case_id': base.rstrip('_'),
                'dwi': dwi_path,
                'adc': adc_candidates[0],
                'flair': flair_candidates[0],
            })
    return cases


# ═══════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════

def preprocess_case(dwi_path: Path, adc_path: Path, flair_path: Path, device: torch.device) -> Tuple[torch.Tensor, nib.Nifti1Image]:
    """
    Load and preprocess a single case from raw NIfTI files.

    Returns:
        image_tensor: [1, 3, H, W, D] normalized tensor on device
        dwi_nii: original DWI NIfTI (for saving output with correct affine)
    """
    # Load NIfTI files
    dwi_nii = nib.load(str(dwi_path))
    adc_nii = nib.load(str(adc_path))
    flair_nii = nib.load(str(flair_path))

    dwi = dwi_nii.get_fdata().astype(np.float32)
    adc = adc_nii.get_fdata().astype(np.float32)
    flair = flair_nii.get_fdata().astype(np.float32)

    # Handle shape mismatches - resize ADC/FLAIR to match DWI
    target_shape = dwi.shape
    if adc.shape != target_shape:
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_shape, adc.shape)]
        adc = zoom(adc, factors, order=1)
    if flair.shape != target_shape:
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_shape, flair.shape)]
        flair = zoom(flair, factors, order=1)

    # Stack channels: [3, H, W, D]
    image = np.stack([dwi, adc, flair], axis=0)

    # Per-channel z-score normalization (non-zero voxels only)
    for c in range(3):
        mask = image[c] > 0
        if mask.sum() > 0:
            mean = image[c][mask].mean()
            std = image[c][mask].std()
            if std > 0:
                image[c] = (image[c] - mean) / std

    # To tensor: [1, 3, H, W, D]
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    return image_tensor, dwi_nii


# ═══════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════

def load_all_models(project_dir: Path, device: torch.device, use_nnunet: bool = True):
    """Load all trained model checkpoints."""
    models = {
        'nnunet': [],
        'swin_unetr': [],
        'segresnet': [],
        'fusion': None,
    }

    # ── nnU-Net (5 folds) ──
    if use_nnunet and HAS_NNUNET:
        nn_dir = project_dir / 'trained_models' / 'nnunet'
        for fold in range(5):
            ckpt = nn_dir / f'fold_{fold}' / 'checkpoint_final.pth'
            if ckpt.exists():
                try:
                    model, _ = nnUNetWrapper.load_checkpoint(ckpt, device=device)
                    model.eval()
                    models['nnunet'].append(model)
                except Exception as e:
                    print(f"  [WARN] nnU-Net fold {fold} failed: {e}")
        print(f"  nnU-Net:    {len(models['nnunet'])} folds loaded")
    else:
        print(f"  nnU-Net:    skipped {'(--no-nnunet)' if not use_nnunet else '(not installed)'}")

    # ── SegResNet (5 folds) ──
    seg_dir = project_dir / 'trained_models' / 'segresnet'
    for fold in range(5):
        ckpt = seg_dir / f'fold_{fold}' / 'checkpoint_best.pth'
        if ckpt.exists():
            model, _ = SegResNetWrapper.load_checkpoint(ckpt, device=device)
            model.eval()
            models['segresnet'].append(model)
    print(f"  SegResNet:  {len(models['segresnet'])} folds loaded")

    # ── Swin-UNETR (5 folds) ──
    swin_dir = project_dir / 'trained_models' / 'swin_unetr'
    for fold in range(5):
        ckpt = swin_dir / f'fold_{fold}' / 'checkpoint_best.pth'
        if ckpt.exists():
            model, _ = SwinUNETRWrapper.load_checkpoint(ckpt, device=device)
            model.eval()
            models['swin_unetr'].append(model)
    print(f"  Swin-UNETR: {len(models['swin_unetr'])} folds loaded")

    # ── Fusion Network ──
    fusion_ckpt = project_dir / 'checkpoints' / 'fusion_v6_best.pth'
    if not fusion_ckpt.exists():
        fusion_ckpt = project_dir / 'trained_models' / 'fusion_network_v4_specialist.pth'

    fusion_net = AdaptiveFusionNetwork(in_channels=8, use_residual=True).to(device)
    if fusion_ckpt.exists():
        ckpt_data = torch.load(fusion_ckpt, map_location=device, weights_only=False)
        fusion_net.load_state_dict(ckpt_data['model_state_dict'])
        print(f"  Fusion:     loaded from {fusion_ckpt.name}")
    else:
        print(f"  Fusion:     [WARN] no checkpoint found, using random weights")
    fusion_net.eval()
    models['fusion'] = fusion_net

    total = len(models['nnunet']) + len(models['segresnet']) + len(models['swin_unetr']) + 1
    print(f"\n  Total models: {total}")
    return models


# ═══════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════

@torch.no_grad()
def run_ensemble_inference(
    image: torch.Tensor,
    models: dict,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Run full ensemble inference on a single preprocessed image.

    Returns dict with keys: 'nnunet', 'swin_unetr', 'segresnet', 'fusion', 'final'
    All values are numpy arrays of shape [H, W, D] with probability values.
    """
    results = {}

    # ── nnU-Net ──
    if models['nnunet']:
        preds = []
        for m in models['nnunet']:
            out = sliding_window_inference(image, roi_size, sw_batch_size, m)
            preds.append(torch.softmax(out, dim=1)[:, 1:])
        nn_avg = torch.mean(torch.stack(preds), dim=0)
        results['nnunet'] = nn_avg
    else:
        nn_avg = None

    # ── Swin-UNETR ──
    if models['swin_unetr']:
        preds = []
        for m in models['swin_unetr']:
            out = sliding_window_inference(image, roi_size, sw_batch_size, m)
            preds.append(torch.softmax(out, dim=1)[:, 1:])
        swin_avg = torch.mean(torch.stack(preds), dim=0)
        results['swin_unetr'] = swin_avg
    else:
        swin_avg = torch.zeros(1, 1, *image.shape[2:], device=image.device)

    # ── SegResNet ──
    if models['segresnet']:
        preds = []
        for m in models['segresnet']:
            out = sliding_window_inference(image, roi_size, sw_batch_size, m)
            preds.append(torch.softmax(out, dim=1)[:, 1:])
        seg_avg = torch.mean(torch.stack(preds), dim=0)
        results['segresnet'] = seg_avg
    else:
        seg_avg = torch.zeros(1, 1, *image.shape[2:], device=image.device)

    # ── Fusion ──
    # nnU-Net fallback if not available
    nnunet_pred = nn_avg if nn_avg is not None else (swin_avg + seg_avg) / 2.0

    fusion_inputs = torch.cat([nnunet_pred, swin_avg, seg_avg], dim=1)
    fused = models['fusion'](fusion_inputs, image)
    results['fusion'] = fused

    # Convert everything to numpy
    np_results = {}
    for key, val in results.items():
        np_results[key] = val.squeeze().cpu().numpy()

    # Final binary prediction
    np_results['final_binary'] = (np_results['fusion'] > 0.5).astype(np.uint8)

    return np_results


# ═══════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics between prediction and ground truth."""
    pred_flat = pred.flatten().astype(bool)
    label_flat = label.flatten().astype(bool)

    tp = np.sum(pred_flat & label_flat)
    fp = np.sum(pred_flat & ~label_flat)
    tn = np.sum(~pred_flat & ~label_flat)
    fn = np.sum(~pred_flat & label_flat)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision_val = tp / (tp + fp + 1e-8)

    return {
        'dice': round(float(dice), 4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'precision': round(float(precision_val), 4),
        'pred_volume_voxels': int(np.sum(pred_flat)),
        'label_volume_voxels': int(np.sum(label_flat)),
    }


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def normalize_display(img, pct_lo=1, pct_hi=99):
    lo = np.percentile(img, pct_lo)
    hi = np.percentile(img, pct_hi)
    if hi - lo > 0:
        return np.clip((img - lo) / (hi - lo), 0, 1)
    return np.zeros_like(img)


def visualize_case(
    case_id: str,
    image_np: np.ndarray,       # [3, H, W, D]
    predictions: Dict[str, np.ndarray],  # each [H, W, D]
    output_dir: Path,
    gt: Optional[np.ndarray] = None,
):
    """Generate visualization for a single case."""
    # Find best slice (most predicted lesion)
    fusion_pred = predictions.get('fusion', predictions.get('final_binary', np.zeros(image_np.shape[1:])))
    slice_sums = [np.sum(fusion_pred[:, :, i] > 0.3) for i in range(fusion_pred.shape[2])]
    best_z = int(np.argmax(slice_sums)) if max(slice_sums) > 0 else fusion_pred.shape[2] // 2

    z = best_z
    alpha = 0.5

    # ─── Panel 1: Multi-model comparison ───
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))

    # Row 1: Input modalities
    for i, name in enumerate(['DWI', 'ADC', 'FLAIR']):
        ax = axes[0, i]
        ax.imshow(np.rot90(normalize_display(image_np[i, :, :, z])), cmap='gray')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.axis('off')

    if gt is not None:
        ax = axes[0, 3]
        mri_norm = normalize_display(image_np[0, :, :, z])
        rgb = np.stack([mri_norm]*3, axis=-1)
        gt_mask = gt[:, :, z] > 0
        rgb[gt_mask, 0] *= (1 - alpha); rgb[gt_mask, 0] += 0.2 * alpha
        rgb[gt_mask, 1] *= (1 - alpha); rgb[gt_mask, 1] += 1.0 * alpha
        rgb[gt_mask, 2] *= (1 - alpha); rgb[gt_mask, 2] += 0.2 * alpha
        ax.imshow(np.rot90(np.clip(rgb, 0, 1)))
        ax.set_title('Ground Truth', fontsize=13, fontweight='bold', color='green')
        ax.axis('off')

    # Row 2: Model predictions
    model_list = [
        ('nnunet', 'nnU-Net', (1, 0.2, 0.2)),
        ('swin_unetr', 'Swin-UNETR', (0.2, 0.5, 1)),
        ('segresnet', 'SegResNet', (0.2, 0.8, 0.3)),
        ('fusion', 'Fusion', (0.6, 0.3, 0.9)),
    ]

    for i, (key, name, color) in enumerate(model_list[:n_cols]):
        ax = axes[1, i]
        if key in predictions:
            pred = predictions[key]
            pred_bin = (pred > 0.5).astype(bool)
            mri_norm = normalize_display(image_np[0, :, :, z])
            rgb = np.stack([mri_norm]*3, axis=-1)

            pred_mask = pred_bin[:, :, z]
            if gt is not None:
                gt_mask = gt[:, :, z] > 0
                overlap = pred_mask & gt_mask
                only_pred = pred_mask & ~gt_mask
                only_gt = gt_mask & ~pred_mask

                rgb[overlap, 0] = rgb[overlap, 0]*(1-alpha) + 1.0*alpha
                rgb[overlap, 1] = rgb[overlap, 1]*(1-alpha) + 1.0*alpha
                rgb[overlap, 2] = rgb[overlap, 2]*(1-alpha) + 0.0*alpha
                rgb[only_pred, 0] = rgb[only_pred, 0]*(1-alpha) + 1.0*alpha
                rgb[only_pred, 1] = rgb[only_pred, 1]*(1-alpha) + 0.2*alpha
                rgb[only_pred, 2] = rgb[only_pred, 2]*(1-alpha) + 0.2*alpha
                rgb[only_gt, 0] = rgb[only_gt, 0]*(1-alpha) + 0.2*alpha
                rgb[only_gt, 1] = rgb[only_gt, 1]*(1-alpha) + 1.0*alpha
                rgb[only_gt, 2] = rgb[only_gt, 2]*(1-alpha) + 0.2*alpha

                dice = compute_metrics(pred_bin, gt)['dice']
                ax.set_title(f'{name}\nDice={dice:.3f}', fontsize=12, fontweight='bold')
            else:
                for c_idx in range(3):
                    rgb[pred_mask, c_idx] = rgb[pred_mask, c_idx]*(1-alpha) + color[c_idx]*alpha
                ax.set_title(name, fontsize=12, fontweight='bold')

            ax.imshow(np.rot90(np.clip(rgb, 0, 1)))
        else:
            ax.text(0.5, 0.5, f'{name}\nN/A', ha='center', va='center', fontsize=14)
            ax.set_title(name, fontsize=12)
        ax.axis('off')

    # Legend
    if gt is not None:
        legend_elements = [
            mpatches.Patch(facecolor='yellow', alpha=0.8, label='True Positive'),
            mpatches.Patch(facecolor='red', alpha=0.8, label='False Positive'),
            mpatches.Patch(facecolor='green', alpha=0.8, label='False Negative'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)

    fig.suptitle(f'{case_id} — Slice {z}', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # ─── Panel 2: Probability heatmaps ───
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    heatmaps = [
        ('nnunet', 'nnU-Net Prob', 'Reds'),
        ('swin_unetr', 'Swin-UNETR Prob', 'Blues'),
        ('segresnet', 'SegResNet Prob', 'Greens'),
        ('fusion', 'Fusion Prob', 'Purples'),
    ]
    for i, (key, title, cmap) in enumerate(heatmaps):
        ax = axes[i]
        if key in predictions:
            im = ax.imshow(np.rot90(predictions[key][:, :, z]), cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f'{case_id} — Probability Maps — Slice {z}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_heatmaps.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Visualizations saved for {case_id}")


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel Stroke - Lambda Labs Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (choose one)
    group = parser.add_argument_group("Input - choose ONE option")
    group.add_argument('--input-dir', type=str, help='Directory containing test NIfTI files')
    group.add_argument('--dwi', type=str, help='Single case: path to DWI NIfTI')
    parser.add_argument('--adc', type=str, help='Single case: path to ADC NIfTI')
    parser.add_argument('--flair', type=str, help='Single case: path to FLAIR NIfTI')

    # Format
    parser.add_argument('--format', type=str, default='nnunet',
                        choices=['isles', 'nnunet', 'triplet'],
                        help='Input data format (default: nnunet)')

    # Ground truth (optional)
    parser.add_argument('--gt-dir', type=str, default=None,
                        help='Directory with ground truth labels (case_XXXX.nii.gz)')

    # Output
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for predictions')
    parser.add_argument('--vis-dir', type=str, default='test_visualizations',
                        help='Output directory for visualizations')

    # Model options
    parser.add_argument('--project-dir', type=str, default=str(PROJECT_ROOT),
                        help='Project root directory')
    parser.add_argument('--no-nnunet', action='store_true',
                        help='Skip nnU-Net (faster, uses 10 models)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Skip visualization generation')

    args = parser.parse_args()

    # ─── Validate inputs ───
    if not args.input_dir and not args.dwi:
        parser.error("Provide either --input-dir or --dwi/--adc/--flair")
    if args.dwi and (not args.adc or not args.flair):
        parser.error("Single case mode requires --dwi, --adc, and --flair")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_dir = Path(args.project_dir)
    output_dir = Path(args.output_dir)
    vis_dir = Path(args.vis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Sentinel Stroke - Prediction Pipeline")
    print("=" * 60)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # ─── Discover test cases ───
    if args.dwi:
        # Single case mode
        cases = [{
            'case_id': Path(args.dwi).stem.replace('_dwi', '').replace('_DWI', ''),
            'dwi': Path(args.dwi),
            'adc': Path(args.adc),
            'flair': Path(args.flair),
        }]
    else:
        input_dir = Path(args.input_dir)
        if args.format == 'isles':
            cases = discover_isles_format(input_dir)
        elif args.format == 'nnunet':
            cases = discover_nnunet_format(input_dir)
        else:
            cases = discover_triplet_format(input_dir)

    if not cases:
        print(f"[ERROR] No test cases found in {args.input_dir}")
        print(f"  Format: {args.format}")
        print(f"  Check the --format flag and directory structure.")
        sys.exit(1)

    print(f"  Found {len(cases)} test case(s):")
    for c in cases[:5]:
        print(f"    {c['case_id']}")
    if len(cases) > 5:
        print(f"    ... and {len(cases) - 5} more")
    print()

    # ─── Load models ───
    print("Loading models...")
    models = load_all_models(project_dir, device, use_nnunet=not args.no_nnunet)
    print()

    # ─── Run inference ───
    print("=" * 60)
    print("  Running Inference")
    print("=" * 60)

    all_metrics = []
    start_time = time.time()

    for case_info in tqdm(cases, desc="Processing"):
        case_id = case_info['case_id']

        # Preprocess
        image_tensor, dwi_nii = preprocess_case(
            case_info['dwi'], case_info['adc'], case_info['flair'], device
        )
        print(f"\n  {case_id}: shape={list(image_tensor.shape)}")

        # Run ensemble
        predictions = run_ensemble_inference(image_tensor, models)

        # Save NIfTI prediction
        result_nii = nib.Nifti1Image(
            predictions['final_binary'],
            dwi_nii.affine,
            dwi_nii.header
        )
        nib.save(result_nii, str(output_dir / f'{case_id}_prediction.nii.gz'))

        # Save probability map too
        prob_nii = nib.Nifti1Image(
            predictions['fusion'].astype(np.float32),
            dwi_nii.affine,
            dwi_nii.header
        )
        nib.save(prob_nii, str(output_dir / f'{case_id}_probability.nii.gz'))

        # Load ground truth if available
        gt = None
        if args.gt_dir:
            gt_path = Path(args.gt_dir) / f'{case_id}.nii.gz'
            if gt_path.exists():
                gt = nib.load(str(gt_path)).get_fdata().astype(np.float32)

        # Compute metrics if GT available
        if gt is not None:
            case_metrics = {'case_id': case_id}
            for model_key in ['nnunet', 'swin_unetr', 'segresnet', 'fusion']:
                if model_key in predictions:
                    pred_bin = (predictions[model_key] > 0.5).astype(np.uint8)
                    m = compute_metrics(pred_bin, gt)
                    for k, v in m.items():
                        case_metrics[f'{model_key}_{k}'] = v
            all_metrics.append(case_metrics)
            print(f"    Fusion Dice: {case_metrics.get('fusion_dice', 'N/A')}")

        # Visualize
        if not args.no_vis:
            image_np = image_tensor.squeeze(0).cpu().numpy()  # [3, H, W, D]
            visualize_case(case_id, image_np, predictions, vis_dir, gt=gt)

        # Free GPU memory
        del image_tensor
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    # ─── Print summary ───
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Cases processed: {len(cases)}")
    print(f"  Total time:      {elapsed:.1f}s ({elapsed/len(cases):.1f}s per case)")
    print(f"  Predictions:     {output_dir}/")
    print(f"  Visualizations:  {vis_dir}/")

    if all_metrics:
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        df.to_csv(output_dir / 'metrics.csv', index=False)

        print(f"\n  {'Method':<20} {'Dice':>8} {'Sensitivity':>12} {'Precision':>10}")
        print("  " + "-" * 52)
        for method in ['nnunet', 'swin_unetr', 'segresnet', 'fusion']:
            col = f'{method}_dice'
            if col in df.columns:
                d = df[col].mean()
                s = df[f'{method}_sensitivity'].mean()
                p = df[f'{method}_precision'].mean()
                print(f"  {method:<20} {d:>8.4f} {s:>12.4f} {p:>10.4f}")
        print("  " + "-" * 52)
        print(f"\n  Metrics saved to: {output_dir / 'metrics.csv'}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
