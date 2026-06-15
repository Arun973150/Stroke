"""
Visualize Problem Cases
========================
Creates side-by-side slice views for each zero-sensitivity case:
  - DWI/TRACE slice
  - ADC slice
  - Ground truth overlay on DWI
  - Prediction overlay on DWI
  - Prediction overlay on ADC

Helps diagnose:
  - Mask misalignment (GT is on wrong region)
  - Model blindness (no signal in GT region)
  - Image quality issues (ADC collapse, motion, etc)

Usage:
  python scripts/visualize_problem_cases.py --debug-dir debug_cases --out-dir debug_cases/viz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_informative_slices(mask, prob, n_slices=5):
    """Pick slices that show: GT lesion, prediction, and middle brain."""
    slices = set()

    # Slices through GT
    if mask.any():
        z_gt = np.where(mask.any(axis=(1, 2)))[0]
        slices.update(np.linspace(z_gt.min(), z_gt.max(), 3, dtype=int).tolist())

    # Slices through high-prob predictions
    if prob is not None and prob.max() > 0.1:
        pred_mask = prob > 0.1
        if pred_mask.any():
            z_pred = np.where(pred_mask.any(axis=(1, 2)))[0]
            slices.update(np.linspace(z_pred.min(), z_pred.max(), 3, dtype=int).tolist())

    # Middle slice if nothing else
    if not slices:
        slices.add(mask.shape[0] // 2)

    return sorted(slices)[:n_slices]


def make_overlay(base, mask, color="red", alpha=0.5):
    """Overlay a binary/prob mask on a grayscale base image."""
    # Normalize base to 0-1
    b = base.astype(np.float32)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    rgb = np.stack([b, b, b], axis=-1)

    if mask.sum() == 0:
        return rgb

    # Color lookup
    colors = {"red": [1, 0, 0], "green": [0, 1, 0], "yellow": [1, 1, 0]}
    c = np.array(colors[color])

    m = mask.astype(np.float32)
    if m.max() > 1:
        m = m / m.max()

    for ch in range(3):
        rgb[..., ch] = rgb[..., ch] * (1 - alpha * m) + c[ch] * alpha * m

    return np.clip(rgb, 0, 1)


def visualize_case(sid, npz_path, prob_path, out_dir, spacing=(1, 1, 1)):
    """Create a visualization figure for one case."""
    data = np.load(npz_path)
    image = data["image"]  # (3, D, H, W)
    mask = data["mask"].astype(np.uint8)

    prob = np.load(prob_path) if prob_path and prob_path.exists() else None

    dwi = image[0]  # TRACE/DWI
    adc = image[1]  # ADC

    # Stats
    vol_ml = mask.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
    max_prob_overall = prob.max() if prob is not None else 0
    max_prob_in_gt = prob[mask > 0].max() if (prob is not None and mask.any()) else 0

    print(f"\n{'=' * 60}")
    print(f"{sid}")
    print(f"{'=' * 60}")
    print(f"  Volume (GT):        {vol_ml:.2f} ml")
    print(f"  Shape:              {mask.shape}")
    print(f"  Max prob overall:   {max_prob_overall:.4f}")
    print(f"  Max prob in GT:     {max_prob_in_gt:.4f}")
    print(f"  DWI range:          [{dwi.min():.2f}, {dwi.max():.2f}] mean={dwi.mean():.2f}")
    print(f"  ADC range:          [{adc.min():.2f}, {adc.max():.2f}] mean={adc.mean():.2f}")

    # Where is the highest prediction vs the GT?
    if prob is not None and prob.max() > 0:
        pred_max_coord = np.unravel_index(prob.argmax(), prob.shape)
        print(f"  Max prob location:  z={pred_max_coord[0]} y={pred_max_coord[1]} x={pred_max_coord[2]}")
        if mask.any():
            gt_coords = np.array(np.where(mask > 0))
            gt_center = gt_coords.mean(axis=1)
            print(f"  GT center:          z={gt_center[0]:.0f} y={gt_center[1]:.0f} x={gt_center[2]:.0f}")
            dist = np.sqrt(sum((pred_max_coord[i] - gt_center[i])**2 for i in range(3)))
            print(f"  Distance (vox):     {dist:.1f}")
            if dist > 20:
                print(f"  *** MASK MISALIGNMENT SUSPECTED ***")

    # Pick informative slices
    slices = get_informative_slices(mask, prob)
    n = len(slices)

    # 5 columns: DWI, ADC, GT on DWI, Pred on DWI, Pred on ADC
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{sid}  |  vol={vol_ml:.2f}ml  |  max_prob_overall={max_prob_overall:.3f}  |  "
        f"max_prob_in_GT={max_prob_in_gt:.3f}",
        fontsize=14, fontweight="bold"
    )

    for i, z in enumerate(slices):
        dwi_slice = dwi[z]
        adc_slice = adc[z]
        mask_slice = mask[z]
        prob_slice = prob[z] if prob is not None else np.zeros_like(mask_slice)

        # Col 0: DWI
        axes[i, 0].imshow(dwi_slice, cmap="gray")
        axes[i, 0].set_title(f"DWI — slice {z}")
        axes[i, 0].axis("off")

        # Col 1: ADC
        axes[i, 1].imshow(adc_slice, cmap="gray")
        axes[i, 1].set_title(f"ADC — slice {z}")
        axes[i, 1].axis("off")

        # Col 2: GT overlay on DWI (red)
        axes[i, 2].imshow(make_overlay(dwi_slice, mask_slice, "red", 0.6))
        gt_on_slice = mask_slice.sum()
        axes[i, 2].set_title(f"GT on DWI (red) — {gt_on_slice} vox")
        axes[i, 2].axis("off")

        # Col 3: Prediction overlay on DWI (yellow)
        pred_binary = (prob_slice > 0.1).astype(np.uint8)
        axes[i, 3].imshow(make_overlay(dwi_slice, pred_binary, "yellow", 0.6))
        pred_vox = pred_binary.sum()
        axes[i, 3].set_title(f"Pred>0.1 on DWI (yellow) — {pred_vox} vox")
        axes[i, 3].axis("off")

        # Col 4: Prediction heatmap on ADC
        axes[i, 4].imshow(adc_slice, cmap="gray")
        if prob_slice.max() > 0.01:
            axes[i, 4].imshow(prob_slice, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[i, 4].set_title(f"Prob heatmap on ADC")
        axes[i, 4].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"{sid}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", default="debug_cases")
    parser.add_argument("--out-dir", default="debug_cases/viz")
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(debug_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} cases")

    for npz in npz_files:
        sid = npz.stem
        prob_path = debug_dir / f"{sid}_prob.npy"
        visualize_case(sid, npz, prob_path, out_dir)

    print(f"\n{'=' * 60}")
    print(f"Done. Outputs in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
