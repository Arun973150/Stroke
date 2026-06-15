"""
Sensitivity Sweep: Find settings that push sensitivity to 90%+
================================================================
Reads saved probability maps from cascade inference and sweeps
aggressive threshold/postprocessing combinations to maximize sensitivity.

No GPU needed — just reads .npy files.

Usage:
  python scripts/20_sensitivity_sweep.py --config configs/soop_config.yaml
  python scripts/20_sensitivity_sweep.py --config configs/soop_config.yaml --pred-dir /path/to/predictions
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")


def compute_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1, 1, 1)) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = (pred & gt).sum()
    fn = (gt & ~pred).sum()
    fp = (pred & ~gt).sum()
    tn = (~pred & ~gt).sum()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)

    voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    gt_vol = gt.sum() * voxel_vol_ml
    pred_vol = pred.sum() * voxel_vol_ml

    return {
        "dice": dice,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "gt_vol_ml": gt_vol,
        "pred_vol_ml": pred_vol,
    }


def classify_volume_bin(volume_ml):
    if volume_ml <= 0:
        return "none"
    elif volume_ml < 1:
        return "tiny"
    elif volume_ml < 5:
        return "small"
    elif volume_ml < 50:
        return "medium"
    else:
        return "large"


def postprocess_highsens(prob: np.ndarray, threshold: float,
                          min_component_size: int,
                          max_prob_threshold: float,
                          min_volume_ml: float,
                          spacing: tuple,
                          use_dilation: bool = False) -> np.ndarray:
    """Postprocess with configurable aggressiveness."""
    # FP suppression: low confidence
    if max_prob_threshold > 0 and prob.max() < max_prob_threshold:
        return np.zeros_like(prob, dtype=np.uint8)

    binary = (prob > threshold).astype(np.uint8)

    # Optional dilation to capture lesion edges
    if use_dilation:
        binary = ndimage.binary_dilation(binary, structure=np.ones((3, 3, 3)),
                                          iterations=1).astype(np.uint8)

    # Volume suppression
    if min_volume_ml > 0:
        voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
        total_vol_ml = binary.sum() * voxel_vol_ml
        if total_vol_ml < min_volume_ml:
            return np.zeros_like(prob, dtype=np.uint8)

    # Remove small components
    if min_component_size > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            if (labeled == i).sum() < min_component_size:
                binary[labeled == i] = 0

    # Morphological closing
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)),
                                     iterations=1).astype(np.uint8)
    return binary


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep to find 90%+ settings")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    spacing = tuple(config["preprocessing"]["target_spacing"])

    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        Path(config["paths"]["checkpoints"]).parent / "predictions" / args.split
    )

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    # Load all probability maps and ground truths
    print(f"Loading {len(subject_ids)} subjects from {pred_dir}...")
    subjects = []
    for sid in tqdm(subject_ids, desc="Loading"):
        prob_path = pred_dir / f"{sid}_prob.npy"
        if not prob_path.exists():
            continue

        gt = np.load(preproc_dir / f"{sid}.npz")["mask"]
        prob = np.load(prob_path)
        gt_vol = gt.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
        vol_bin = classify_volume_bin(gt_vol)

        subjects.append({
            "sid": sid,
            "prob": prob,
            "gt": gt.astype(bool),
            "gt_vol_ml": gt_vol,
            "vol_bin": vol_bin,
        })

    print(f"Loaded {len(subjects)} subjects with probability maps\n")

    # Count by bin
    bin_counts = {}
    for s in subjects:
        b = s["vol_bin"]
        bin_counts[b] = bin_counts.get(b, 0) + 1
    for b, c in sorted(bin_counts.items()):
        print(f"  {b}: {c}")

    # ── Sweep configurations ──────────────────────────────────────────────
    # Aggressive configs targeting 90%+ sensitivity
    configs = [
        # (name, threshold, min_comp, max_prob_thresh, min_vol_ml, use_dilation)
        ("current",           0.35, 5, 0.40, 0.20, False),
        ("low_thresh_0.25",   0.25, 5, 0.40, 0.20, False),
        ("low_thresh_0.20",   0.20, 5, 0.40, 0.20, False),
        ("low_thresh_0.15",   0.15, 5, 0.40, 0.20, False),
        ("low_thresh_0.10",   0.10, 5, 0.40, 0.20, False),
        ("no_fp_suppr_0.20",  0.20, 3, 0.00, 0.00, False),
        ("no_fp_suppr_0.15",  0.15, 3, 0.00, 0.00, False),
        ("no_fp_suppr_0.10",  0.10, 3, 0.00, 0.00, False),
        ("aggressive_0.15",   0.15, 1, 0.00, 0.00, False),
        ("aggressive_0.10",   0.10, 1, 0.00, 0.00, False),
        ("dilated_0.20",      0.20, 3, 0.00, 0.00, True),
        ("dilated_0.15",      0.15, 3, 0.00, 0.00, True),
        ("ultra_sens_0.08",   0.08, 1, 0.00, 0.00, False),
        ("ultra_dilated_0.10",0.10, 1, 0.00, 0.00, True),
    ]

    print(f"\n{'=' * 120}")
    print(f"SENSITIVITY SWEEP — {len(configs)} configurations")
    print(f"{'=' * 120}")
    print(f"{'Config':<25} {'Dice':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} "
          f"{'FP%':>6} {'Tiny':>8} {'Small':>8} {'Med':>8} {'Large':>8} {'NOTE':>12}")
    print(f"{'-' * 120}")

    best_sens90 = None

    for name, thresh, min_comp, max_prob, min_vol, use_dil in configs:
        all_metrics = []
        bin_metrics = {"tiny": [], "small": [], "medium": [], "large": [], "none": []}
        fp_count = 0
        no_lesion_count = 0

        for s in subjects:
            pred = postprocess_highsens(
                s["prob"], thresh, min_comp, max_prob, min_vol, spacing, use_dil
            )
            m = compute_metrics(pred, s["gt"], spacing)
            m["vol_bin"] = s["vol_bin"]
            all_metrics.append(m)
            bin_metrics[s["vol_bin"]].append(m)

            if s["vol_bin"] == "none":
                no_lesion_count += 1
                if pred.sum() > 0:
                    fp_count += 1

        # Aggregate — only lesion-positive cases for Dice/Sens
        pos_metrics = [m for m in all_metrics if m["gt_vol_ml"] > 0]
        avg_dice = np.mean([m["dice"] for m in pos_metrics])
        avg_sens = np.mean([m["sensitivity"] for m in pos_metrics])
        avg_spec = np.mean([m["specificity"] for m in pos_metrics])
        avg_prec = np.mean([m["precision"] for m in pos_metrics])
        fp_pct = fp_count / max(no_lesion_count, 1) * 100

        # Per-bin Dice
        bin_dice = {}
        for b in ["tiny", "small", "medium", "large"]:
            if bin_metrics[b]:
                bin_dice[b] = np.mean([m["dice"] for m in bin_metrics[b]])
            else:
                bin_dice[b] = 0.0

        note = ""
        if avg_sens >= 0.90:
            note = "<-- 90%+"
            if best_sens90 is None or avg_dice > best_sens90["dice"]:
                best_sens90 = {"name": name, "dice": avg_dice, "sens": avg_sens,
                               "thresh": thresh, "min_comp": min_comp,
                               "max_prob": max_prob, "min_vol": min_vol,
                               "use_dil": use_dil, "fp_pct": fp_pct}

        print(f"{name:<25} {avg_dice:>8.4f} {avg_sens:>8.4f} {avg_spec:>8.4f} {avg_prec:>8.4f} "
              f"{fp_pct:>5.1f}% {bin_dice['tiny']:>8.4f} {bin_dice['small']:>8.4f} "
              f"{bin_dice['medium']:>8.4f} {bin_dice['large']:>8.4f} {note:>12}")

    print(f"\n{'=' * 120}")

    if best_sens90:
        print(f"\nBEST CONFIG WITH SENSITIVITY >= 90%:")
        print(f"  Name:           {best_sens90['name']}")
        print(f"  Dice:           {best_sens90['dice']:.4f}")
        print(f"  Sensitivity:    {best_sens90['sens']:.4f}")
        print(f"  FP rate:        {best_sens90['fp_pct']:.1f}%")
        print(f"  Settings:")
        print(f"    threshold:          {best_sens90['thresh']}")
        print(f"    min_component_size: {best_sens90['min_comp']}")
        print(f"    max_prob_threshold: {best_sens90['max_prob']}")
        print(f"    min_volume_ml:      {best_sens90['min_vol']}")
        print(f"    use_dilation:       {best_sens90['use_dil']}")
        print(f"\n  To apply: update postprocess() call in 10_cascade_inference.py")
    else:
        print(f"\nWARNING: No config reached 90% sensitivity!")
        print(f"This means the models themselves need improvement:")
        print(f"  1. Retrain with Tversky loss (alpha=0.7, beta=0.3)")
        print(f"  2. Add Path B (full-brain) for cases Stage 1 misses")
        print(f"  3. Lower Stage 1 detection_threshold from 0.1 to 0.05")

    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
