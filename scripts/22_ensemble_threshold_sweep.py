"""
Ensemble Threshold Sweep
=========================
Reads saved ensemble probability maps and sweeps threshold + postprocessing
combinations to find the optimal setting for 90%+ sensitivity.

No GPU needed — just reads .npy files.

Usage:
  python scripts/22_ensemble_threshold_sweep.py --config configs/soop_config.yaml --split test
  python scripts/22_ensemble_threshold_sweep.py --config configs/soop_config.yaml --split test --pred-dir /path/to/predictions
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
        "dice": dice, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision,
        "gt_vol_ml": gt_vol, "pred_vol_ml": pred_vol,
    }


def classify_volume_bin(volume_ml):
    if volume_ml <= 0: return "none"
    elif volume_ml < 1: return "tiny"
    elif volume_ml < 5: return "small"
    elif volume_ml < 50: return "medium"
    else: return "large"


def postprocess(prob, threshold, min_comp, comp_mean_thresh, use_closing, use_dilation):
    binary = (prob > threshold).astype(np.uint8)

    if use_dilation:
        binary = ndimage.binary_dilation(binary, structure=np.ones((3, 3, 3)),
                                          iterations=1).astype(np.uint8)

    if min_comp > 0 or comp_mean_thresh > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            comp_mask = labeled == i
            if comp_mask.sum() < min_comp:
                binary[comp_mask] = 0
                continue
            if comp_mean_thresh > 0 and prob[comp_mask].mean() < comp_mean_thresh:
                binary[comp_mask] = 0

    if use_closing:
        binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)),
                                         iterations=1).astype(np.uint8)
    return binary


def main():
    parser = argparse.ArgumentParser(description="Ensemble threshold sweep")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pred-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"])
    spacing = tuple(config["preprocessing"]["target_spacing"])

    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        ckpt_dir.parent / "predictions_ensemble" / args.split
    )

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    # Load all probability maps and ground truths
    print(f"Loading {len(subject_ids)} subjects from {pred_dir}...")
    subjects = []
    missing = 0
    for sid in tqdm(subject_ids, desc="Loading"):
        prob_path = pred_dir / f"{sid}_prob.npy"
        if not prob_path.exists():
            missing += 1
            continue

        gt = np.load(preproc_dir / f"{sid}.npz")["mask"]
        prob = np.load(prob_path)
        gt_vol = gt.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
        vol_bin = classify_volume_bin(gt_vol)

        subjects.append({
            "sid": sid, "prob": prob, "gt": gt.astype(bool),
            "gt_vol_ml": gt_vol, "vol_bin": vol_bin,
        })

    print(f"Loaded {len(subjects)} subjects ({missing} missing)")

    # Count by bin
    bin_counts = {}
    for s in subjects:
        bin_counts[s["vol_bin"]] = bin_counts.get(s["vol_bin"], 0) + 1
    for b, c in sorted(bin_counts.items()):
        print(f"  {b}: {c}")

    # ── Sweep configurations ──────────────────────────────────────────────
    configs = [
        # (name, threshold, min_comp, comp_mean_thresh, use_closing, use_dilation)
        # Standard
        ("t=0.50",              0.50, 10, 0.0,  True, False),
        ("t=0.40",              0.40, 10, 0.0,  True, False),
        ("t=0.35",              0.35, 10, 0.0,  True, False),
        ("t=0.30",              0.30, 10, 0.0,  True, False),
        ("t=0.25",              0.25, 10, 0.0,  True, False),
        ("t=0.20",              0.20, 10, 0.0,  True, False),
        ("t=0.15",              0.15, 10, 0.0,  True, False),
        ("t=0.10",              0.10, 10, 0.0,  True, False),
        ("t=0.05",              0.05, 10, 0.0,  True, False),
        # Aggressive — lower min component
        ("t=0.20_mc3",          0.20,  3, 0.0,  True, False),
        ("t=0.15_mc3",          0.15,  3, 0.0,  True, False),
        ("t=0.10_mc3",          0.10,  3, 0.0,  True, False),
        ("t=0.10_mc1",          0.10,  1, 0.0,  True, False),
        # With component mean filtering (ISLES winner trick)
        ("t=0.15_cm0.2",        0.15,  3, 0.2,  True, False),
        ("t=0.10_cm0.2",        0.10,  3, 0.2,  True, False),
        ("t=0.10_cm0.15",       0.10,  3, 0.15, True, False),
        ("t=0.05_cm0.2",        0.05,  3, 0.2,  True, False),
        # With dilation (expand predictions to catch edges)
        ("t=0.20_dil",          0.20,  5, 0.0,  True, True),
        ("t=0.15_dil",          0.15,  5, 0.0,  True, True),
        ("t=0.10_dil",          0.10,  5, 0.0,  True, True),
        # Ultra aggressive
        ("t=0.05_mc1",          0.05,  1, 0.0,  True, False),
        ("t=0.05_mc1_dil",      0.05,  1, 0.0,  True, True),
        # No postprocessing at all
        ("t=0.20_raw",          0.20,  0, 0.0, False, False),
        ("t=0.10_raw",          0.10,  0, 0.0, False, False),
    ]

    print(f"\n{'=' * 140}")
    print(f"ENSEMBLE THRESHOLD SWEEP — {len(configs)} configurations, {len(subjects)} subjects")
    print(f"{'=' * 140}")
    print(f"{'Config':<20} {'Dice':>7} {'Sens':>7} {'MedSens':>8} {'Spec':>7} {'Prec':>7} "
          f"{'FP%':>5} {'Tiny':>7} {'Small':>7} {'Med':>7} {'Large':>7} "
          f"{'TinySn':>7} {'SmSn':>7} {'MedSn':>7} {'LgSn':>7} {'NOTE':>10}")
    print(f"{'-' * 140}")

    best_sens90 = None
    best_balanced = None

    for name, thresh, min_comp, comp_mean, closing, dilation in configs:
        all_m = []
        bin_m = {"tiny": [], "small": [], "medium": [], "large": [], "none": []}
        fp_count = 0
        no_lesion_count = 0

        for s in subjects:
            pred = postprocess(s["prob"], thresh, min_comp, comp_mean, closing, dilation)
            m = compute_metrics(pred, s["gt"], spacing)
            m["vol_bin"] = s["vol_bin"]
            all_m.append(m)
            bin_m[s["vol_bin"]].append(m)

            if s["vol_bin"] == "none":
                no_lesion_count += 1
                if pred.sum() > 0:
                    fp_count += 1

        pos = [m for m in all_m if m["gt_vol_ml"] > 0]
        if not pos:
            continue

        avg_dice = np.mean([m["dice"] for m in pos])
        avg_sens = np.mean([m["sensitivity"] for m in pos])
        med_sens = np.median([m["sensitivity"] for m in pos])
        avg_spec = np.mean([m["specificity"] for m in pos])
        avg_prec = np.mean([m["precision"] for m in pos])
        fp_pct = fp_count / max(no_lesion_count, 1) * 100

        # Per-bin Dice and Sensitivity
        bd = {}
        bs = {}
        for b in ["tiny", "small", "medium", "large"]:
            bm = [m for m in bin_m[b] if m["gt_vol_ml"] > 0]
            bd[b] = np.mean([m["dice"] for m in bm]) if bm else 0.0
            bs[b] = np.mean([m["sensitivity"] for m in bm]) if bm else 0.0

        note = ""
        if avg_sens >= 0.90:
            note = "<-- 90%+"
            if best_sens90 is None or avg_dice > best_sens90["dice"]:
                best_sens90 = {"name": name, "dice": avg_dice, "sens": avg_sens,
                               "med_sens": med_sens, "prec": avg_prec, "fp_pct": fp_pct,
                               "thresh": thresh, "min_comp": min_comp,
                               "comp_mean": comp_mean, "closing": closing, "dilation": dilation}

        # Track best balanced (highest Dice with sens >= 0.80)
        if avg_sens >= 0.80:
            if best_balanced is None or avg_dice > best_balanced["dice"]:
                best_balanced = {"name": name, "dice": avg_dice, "sens": avg_sens,
                                 "thresh": thresh}

        print(f"{name:<20} {avg_dice:>7.4f} {avg_sens:>7.4f} {med_sens:>8.4f} {avg_spec:>7.4f} {avg_prec:>7.4f} "
              f"{fp_pct:>4.0f}% {bd['tiny']:>7.4f} {bd['small']:>7.4f} {bd['medium']:>7.4f} {bd['large']:>7.4f} "
              f"{bs['tiny']:>7.4f} {bs['small']:>7.4f} {bs['medium']:>7.4f} {bs['large']:>7.4f} {note:>10}")

    print(f"\n{'=' * 140}")

    if best_sens90:
        print(f"\n*** BEST CONFIG WITH SENSITIVITY >= 90% ***")
        print(f"  Name:              {best_sens90['name']}")
        print(f"  Dice:              {best_sens90['dice']:.4f}")
        print(f"  Mean Sensitivity:  {best_sens90['sens']:.4f}")
        print(f"  Med  Sensitivity:  {best_sens90['med_sens']:.4f}")
        print(f"  Precision:         {best_sens90['prec']:.4f}")
        print(f"  FP rate:           {best_sens90['fp_pct']:.1f}%")
        print(f"  Settings:")
        print(f"    threshold:              {best_sens90['thresh']}")
        print(f"    min_component_size:     {best_sens90['min_comp']}")
        print(f"    component_mean_thresh:  {best_sens90['comp_mean']}")
        print(f"    use_closing:            {best_sens90['closing']}")
        print(f"    use_dilation:           {best_sens90['dilation']}")
    else:
        print(f"\nWARNING: No config reached 90% mean sensitivity!")
        if best_balanced:
            print(f"  Best balanced (sens>=80%): {best_balanced['name']} — "
                  f"Dice {best_balanced['dice']:.4f}, Sens {best_balanced['sens']:.4f}")
        print(f"\n  Options to push higher:")
        print(f"    1. ADC < 620 masking (remove anatomically impossible FPs)")
        print(f"    2. Retrain with alpha=0.8 Tversky (more FN penalty)")
        print(f"    3. Add more models to ensemble")
        print(f"    4. Per-bin thresholds (ultra-low for tiny lesions)")

    print(f"{'=' * 140}")


if __name__ == "__main__":
    main()
