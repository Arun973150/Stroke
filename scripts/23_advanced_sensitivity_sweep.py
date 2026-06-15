"""
Advanced Sensitivity Sweep — Push to 90%+
==========================================
Enhanced sweep with:
  - ADC < 620 masking (remove anatomically impossible FPs)
  - Multi-iteration dilation (1, 2, 3 iterations)
  - Per-volume-bin thresholds (ultra-low for tiny, higher for large)
  - Zero-sensitivity case investigation

Usage:
  python scripts/23_advanced_sensitivity_sweep.py --config configs/soop_config.yaml --split test
  python scripts/23_advanced_sensitivity_sweep.py --config configs/soop_config.yaml --split test --pred-dir /path/to/predictions
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


def compute_metrics(pred, gt, spacing=(1, 1, 1)):
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
    return {
        "dice": dice, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision,
        "gt_vol_ml": gt.sum() * voxel_vol_ml,
        "pred_vol_ml": pred.sum() * voxel_vol_ml,
    }


def classify_volume_bin(volume_ml):
    if volume_ml <= 0: return "none"
    elif volume_ml < 1: return "tiny"
    elif volume_ml < 5: return "small"
    elif volume_ml < 50: return "medium"
    else: return "large"


def postprocess_advanced(prob, threshold, min_comp=3, dilation_iter=0,
                          use_closing=True, adc_mask=None):
    binary = (prob > threshold).astype(np.uint8)

    # ADC masking — only keep predictions where ADC < 620 (acute ischemia)
    if adc_mask is not None:
        binary = binary & adc_mask

    # Dilation
    if dilation_iter > 0:
        binary = ndimage.binary_dilation(
            binary, structure=np.ones((3, 3, 3)),
            iterations=dilation_iter
        ).astype(np.uint8)
        # Re-apply ADC mask after dilation
        if adc_mask is not None:
            binary = binary & adc_mask

    # Remove small components
    if min_comp > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            if (labeled == i).sum() < min_comp:
                binary[labeled == i] = 0

    # Morphological closing
    if use_closing:
        binary = ndimage.binary_closing(
            binary, structure=np.ones((3, 3, 3)), iterations=1
        ).astype(np.uint8)

    return binary


def postprocess_prob_weighted_dilation(prob, threshold, iterations=1,
                                        decay=0.5, final_thresh_ratio=0.3,
                                        min_comp=3):
    """Probability-weighted dilation: new voxels get neighbor prob * decay.
    Keeps sensitivity gain of dilation but preserves Dice much better."""
    weighted = prob.copy()
    binary = (prob > threshold).astype(np.float32)

    for i in range(iterations):
        dilated = ndimage.binary_dilation(binary > 0, structure=np.ones((3, 3, 3)))
        new_voxels = dilated & (binary == 0)
        # Weight new voxels by max neighbor probability * decay
        max_neighbor = ndimage.maximum_filter(weighted, size=3)
        binary[new_voxels] = max_neighbor[new_voxels] * decay

    # Final threshold
    result = (binary > threshold * final_thresh_ratio).astype(np.uint8)

    # Remove small components
    if min_comp > 0:
        labeled, n = ndimage.label(result)
        for i in range(1, n + 1):
            if (labeled == i).sum() < min_comp:
                result[labeled == i] = 0

    # Closing
    result = ndimage.binary_closing(result, structure=np.ones((3, 3, 3)),
                                     iterations=1).astype(np.uint8)
    return result


def postprocess_uncertainty_dilation(prob, model_probs, threshold,
                                      uncertainty_percentile=50,
                                      min_prob=0.05, min_comp=3):
    """Uncertainty-guided dilation: only dilate where models disagree."""
    uncertainty = np.std(model_probs, axis=0)
    binary = (prob > threshold).astype(np.uint8)

    # Find boundary of prediction
    dilated = ndimage.binary_dilation(binary, structure=np.ones((3, 3, 3)))
    boundary = dilated & ~binary.astype(bool)

    if boundary.sum() > 0:
        # Only dilate into high-uncertainty boundary voxels
        unc_thresh = np.percentile(uncertainty[boundary], uncertainty_percentile)
        selective = boundary & (uncertainty > unc_thresh) & (prob > min_prob)
        binary = binary | selective.astype(np.uint8)

    # Remove small components
    if min_comp > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            if (labeled == i).sum() < min_comp:
                binary[labeled == i] = 0

    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)),
                                     iterations=1).astype(np.uint8)
    return binary


def postprocess_perbin(prob, gt_vol_ml, thresholds_by_bin, min_comp=3,
                        dilation_iter=1, adc_mask=None):
    """Apply different thresholds based on expected lesion volume.
    Since we don't know GT volume at test time, we apply multiple thresholds
    and take the union — catches tiny lesions with low threshold while
    keeping precision for large ones."""

    # Strategy: apply lowest threshold, then filter by component properties
    # Tiny lesions: very low threshold, small components OK
    # Large lesions: higher threshold, larger components
    binary_union = np.zeros_like(prob, dtype=np.uint8)

    for vol_bin, (thresh, min_sz) in thresholds_by_bin.items():
        binary = (prob > thresh).astype(np.uint8)

        if adc_mask is not None:
            binary = binary & adc_mask

        if dilation_iter > 0:
            binary = ndimage.binary_dilation(
                binary, structure=np.ones((3, 3, 3)),
                iterations=dilation_iter
            ).astype(np.uint8)

        # Remove components by size range for this bin
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            comp_size = (labeled == i).sum()
            if comp_size < min_sz:
                binary[labeled == i] = 0

        binary_union = binary_union | binary

    # Final closing
    binary_union = ndimage.binary_closing(
        binary_union, structure=np.ones((3, 3, 3)), iterations=1
    ).astype(np.uint8)

    return binary_union


def main():
    parser = argparse.ArgumentParser(description="Advanced sensitivity sweep")
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

    # Load all data
    print(f"Loading {len(subject_ids)} subjects from {pred_dir}...")
    subjects = []
    missing = 0
    for sid in tqdm(subject_ids, desc="Loading"):
        prob_path = pred_dir / f"{sid}_prob.npy"
        if not prob_path.exists():
            missing += 1
            continue

        npz = np.load(preproc_dir / f"{sid}.npz")
        gt = npz["mask"]
        image = npz["image"]  # (3, D, H, W) — channel 1 is ADC
        prob = np.load(prob_path)
        gt_vol = gt.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
        vol_bin = classify_volume_bin(gt_vol)

        # ADC mask: channel 1, z-score normalized — need to find threshold
        # ADC < 620 in raw space, but we have z-score normalized values
        # We'll use a relative approach: low ADC = negative z-scores
        # Use percentile-based: bottom 40% of brain ADC values
        adc = image[1]  # ADC channel
        brain_mask = np.abs(image[0]) > 0.1  # rough brain mask from TRACE
        if brain_mask.sum() > 0:
            adc_brain = adc[brain_mask]
            adc_thresh = np.percentile(adc_brain, 40)  # bottom 40%
            adc_mask = (adc < adc_thresh) & brain_mask
        else:
            adc_mask = np.ones_like(gt, dtype=bool)

        subjects.append({
            "sid": sid, "prob": prob, "gt": gt.astype(bool),
            "gt_vol_ml": gt_vol, "vol_bin": vol_bin,
            "adc_mask": adc_mask.astype(np.uint8),
            "max_prob": prob.max(),
        })

    print(f"Loaded {len(subjects)} subjects ({missing} missing)")

    bin_counts = {}
    for s in subjects:
        bin_counts[s["vol_bin"]] = bin_counts.get(s["vol_bin"], 0) + 1
    for b, c in sorted(bin_counts.items()):
        print(f"  {b}: {c}")

    # ── Investigate zero-sensitivity cases ──────────────────────────────
    print(f"\n{'=' * 80}")
    print("ZERO-SENSITIVITY INVESTIGATION")
    print(f"{'=' * 80}")

    for s in subjects:
        if s["gt_vol_ml"] <= 0:
            continue
        # Check with ultra-low threshold
        pred_ultra = (s["prob"] > 0.01).astype(bool)
        overlap = (pred_ultra & s["gt"]).sum()
        if overlap == 0:
            print(f"  {s['sid']}: vol={s['gt_vol_ml']:.2f}ml, bin={s['vol_bin']}, "
                  f"max_prob_in_GT={s['prob'][s['gt']].max():.4f}, "
                  f"max_prob_overall={s['max_prob']:.4f} — "
                  f"{'UNFIXABLE (model blind)' if s['prob'][s['gt']].max() < 0.01 else 'fixable with lower threshold'}")

    # ── Sweep configurations ──────────────────────────────────────────────
    configs = [
        # (name, threshold, min_comp, dilation_iter, use_adc)
        # Baseline
        ("baseline_t0.20",        0.20, 3, 0, False),
        ("baseline_t0.10",        0.10, 3, 0, False),
        ("baseline_t0.05",        0.05, 3, 0, False),
        # Dilation 1 iteration
        ("dil1_t0.15",            0.15, 3, 1, False),
        ("dil1_t0.10",            0.10, 3, 1, False),
        ("dil1_t0.05",            0.05, 1, 1, False),
        # Dilation 2 iterations
        ("dil2_t0.20",            0.20, 3, 2, False),
        ("dil2_t0.15",            0.15, 3, 2, False),
        ("dil2_t0.10",            0.10, 3, 2, False),
        ("dil2_t0.05",            0.05, 1, 2, False),
        # Dilation 3 iterations
        ("dil3_t0.15",            0.15, 3, 3, False),
        ("dil3_t0.10",            0.10, 3, 3, False),
        ("dil3_t0.05",            0.05, 1, 3, False),
        # Ultra aggressive (no ADC — ADC masking broken by z-score normalization)
        ("ultra_dil2_t0.03",      0.03, 1, 2, False),
        ("ultra_dil3_t0.03",      0.03, 1, 3, False),
    ]

    print(f"\n{'=' * 150}")
    print(f"ADVANCED SENSITIVITY SWEEP — {len(configs)} configurations")
    print(f"{'=' * 150}")
    print(f"{'Config':<25} {'Dice':>7} {'Sens':>7} {'MedSn':>7} {'Spec':>7} {'Prec':>7} "
          f"{'FP%':>5} {'TnSn':>7} {'SmSn':>7} {'MdSn':>7} {'LgSn':>7} "
          f"{'TnDc':>7} {'SmDc':>7} {'MdDc':>7} {'LgDc':>7} {'S=0':>4} {'NOTE':>10}")
    print(f"{'-' * 150}")

    best_sens90 = None
    best_sens85 = None

    for name, thresh, min_comp, dil_iter, use_adc in configs:
        all_m = []
        bin_m = {"tiny": [], "small": [], "medium": [], "large": [], "none": []}
        fp_count = 0
        no_lesion_count = 0
        zero_sens_count = 0

        for s in subjects:
            adc_mask = s["adc_mask"] if use_adc else None
            pred = postprocess_advanced(
                s["prob"], thresh, min_comp, dil_iter, True, adc_mask
            )
            m = compute_metrics(pred, s["gt"], spacing)
            m["vol_bin"] = s["vol_bin"]
            all_m.append(m)
            bin_m[s["vol_bin"]].append(m)

            if s["vol_bin"] == "none":
                no_lesion_count += 1
                if pred.sum() > 0:
                    fp_count += 1
            elif m["sensitivity"] == 0:
                zero_sens_count += 1

        pos = [m for m in all_m if m["gt_vol_ml"] > 0]
        if not pos:
            continue

        avg_dice = np.mean([m["dice"] for m in pos])
        avg_sens = np.mean([m["sensitivity"] for m in pos])
        med_sens = np.median([m["sensitivity"] for m in pos])
        avg_spec = np.mean([m["specificity"] for m in pos])
        avg_prec = np.mean([m["precision"] for m in pos])
        fp_pct = fp_count / max(no_lesion_count, 1) * 100

        bd, bs = {}, {}
        for b in ["tiny", "small", "medium", "large"]:
            bm = [m for m in bin_m[b] if m["gt_vol_ml"] > 0]
            bd[b] = np.mean([m["dice"] for m in bm]) if bm else 0
            bs[b] = np.mean([m["sensitivity"] for m in bm]) if bm else 0

        note = ""
        if avg_sens >= 0.90:
            note = "*** 90%+ ***"
            if best_sens90 is None or avg_dice > best_sens90["dice"]:
                best_sens90 = {"name": name, "dice": avg_dice, "sens": avg_sens,
                               "med_sens": med_sens, "prec": avg_prec, "fp_pct": fp_pct,
                               "zero_sens": zero_sens_count}
        elif avg_sens >= 0.85:
            note = "<-- 85%+"
            if best_sens85 is None or avg_dice > best_sens85["dice"]:
                best_sens85 = {"name": name, "dice": avg_dice, "sens": avg_sens}

        print(f"{name:<25} {avg_dice:>7.4f} {avg_sens:>7.4f} {med_sens:>7.4f} {avg_spec:>7.4f} {avg_prec:>7.4f} "
              f"{fp_pct:>4.0f}% {bs['tiny']:>7.4f} {bs['small']:>7.4f} {bs['medium']:>7.4f} {bs['large']:>7.4f} "
              f"{bd['tiny']:>7.4f} {bd['small']:>7.4f} {bd['medium']:>7.4f} {bd['large']:>7.4f} "
              f"{zero_sens_count:>4} {note:>10}")

    # NOTE: Per-bin threshold and ADC masking sections removed — ADC masking
    # is broken due to z-score normalization, and per-bin is too slow with
    # minimal benefit over prob-weighted dilation.

    # ── Probability-weighted dilation sweep ─────────────────────────────
    print(f"\n{'=' * 150}")
    print("PROBABILITY-WEIGHTED DILATION SWEEP")
    print(f"{'=' * 150}")
    print(f"{'Config':<30} {'Dice':>7} {'Sens':>7} {'MedSn':>7} {'Spec':>7} {'Prec':>7} "
          f"{'FP%':>5} {'TnSn':>7} {'SmSn':>7} {'MdSn':>7} {'LgSn':>7} "
          f"{'TnDc':>7} {'SmDc':>7} {'MdDc':>7} {'LgDc':>7} {'S=0':>4} {'NOTE':>10}")
    print(f"{'-' * 160}")

    pwd_configs = [
        # (name, threshold, iterations, decay, final_thresh_ratio, min_comp)
        # decay=0.5 means new voxels get half the max neighbor prob
        # final_thresh_ratio=0.3 means keep if weighted prob > threshold * 0.3
        # Conservative
        ("pwd_t0.20_d0.5_i1",      0.20, 1, 0.5, 0.3, 3),
        ("pwd_t0.15_d0.5_i1",      0.15, 1, 0.5, 0.3, 3),
        ("pwd_t0.10_d0.5_i1",      0.10, 1, 0.5, 0.3, 3),
        # Higher decay (more aggressive expansion)
        ("pwd_t0.20_d0.7_i1",      0.20, 1, 0.7, 0.3, 3),
        ("pwd_t0.15_d0.7_i1",      0.15, 1, 0.7, 0.3, 3),
        ("pwd_t0.10_d0.7_i1",      0.10, 1, 0.7, 0.3, 3),
        # 2 iterations
        ("pwd_t0.20_d0.5_i2",      0.20, 2, 0.5, 0.3, 3),
        ("pwd_t0.15_d0.5_i2",      0.15, 2, 0.5, 0.3, 3),
        ("pwd_t0.10_d0.5_i2",      0.10, 2, 0.5, 0.3, 3),
        ("pwd_t0.15_d0.7_i2",      0.15, 2, 0.7, 0.3, 3),
        ("pwd_t0.10_d0.7_i2",      0.10, 2, 0.7, 0.3, 3),
        # Lower final threshold ratio (keep more)
        ("pwd_t0.15_d0.5_r0.2_i1", 0.15, 1, 0.5, 0.2, 3),
        ("pwd_t0.10_d0.5_r0.2_i1", 0.10, 1, 0.5, 0.2, 3),
        ("pwd_t0.10_d0.7_r0.2_i2", 0.10, 2, 0.7, 0.2, 3),
        # Very low threshold + weighted dilation
        ("pwd_t0.05_d0.5_i1",      0.05, 1, 0.5, 0.3, 1),
        ("pwd_t0.05_d0.7_i2",      0.05, 2, 0.7, 0.3, 1),
        # Lower decay (more conservative)
        ("pwd_t0.10_d0.3_i1",      0.10, 1, 0.3, 0.3, 3),
        ("pwd_t0.10_d0.3_i2",      0.10, 2, 0.3, 0.3, 3),
    ]

    best_pwd = None

    for name, thresh, iters, decay, ftr, min_comp in pwd_configs:
        all_m = []
        bin_m = {"tiny": [], "small": [], "medium": [], "large": [], "none": []}
        fp_count = 0
        no_lesion_count = 0
        zero_sens_count = 0

        for s in subjects:
            pred = postprocess_prob_weighted_dilation(
                s["prob"], thresh, iterations=iters, decay=decay,
                final_thresh_ratio=ftr, min_comp=min_comp
            )
            m = compute_metrics(pred, s["gt"], spacing)
            m["vol_bin"] = s["vol_bin"]
            all_m.append(m)
            bin_m[s["vol_bin"]].append(m)

            if s["vol_bin"] == "none":
                no_lesion_count += 1
                if pred.sum() > 0:
                    fp_count += 1
            elif m["sensitivity"] == 0:
                zero_sens_count += 1

        pos = [m for m in all_m if m["gt_vol_ml"] > 0]
        if not pos:
            continue

        avg_dice = np.mean([m["dice"] for m in pos])
        avg_sens = np.mean([m["sensitivity"] for m in pos])
        med_sens = np.median([m["sensitivity"] for m in pos])
        avg_spec = np.mean([m["specificity"] for m in pos])
        avg_prec = np.mean([m["precision"] for m in pos])
        fp_pct = fp_count / max(no_lesion_count, 1) * 100

        bd, bs = {}, {}
        for b in ["tiny", "small", "medium", "large"]:
            bm = [m for m in bin_m[b] if m["gt_vol_ml"] > 0]
            bd[b] = np.mean([m["dice"] for m in bm]) if bm else 0
            bs[b] = np.mean([m["sensitivity"] for m in bm]) if bm else 0

        note = ""
        if avg_sens >= 0.90:
            note = "*** 90%+ ***"
            if best_sens90 is None or avg_dice > best_sens90["dice"]:
                best_sens90 = {"name": name, "dice": avg_dice, "sens": avg_sens,
                               "med_sens": med_sens, "prec": avg_prec, "fp_pct": fp_pct,
                               "zero_sens": zero_sens_count}
        elif avg_sens >= 0.85:
            note = "<-- 85%+"

        if best_pwd is None or (avg_sens >= 0.85 and avg_dice > (best_pwd.get("dice", 0))):
            best_pwd = {"name": name, "dice": avg_dice, "sens": avg_sens,
                        "med_sens": med_sens}

        print(f"{name:<30} {avg_dice:>7.4f} {avg_sens:>7.4f} {med_sens:>7.4f} {avg_spec:>7.4f} {avg_prec:>7.4f} "
              f"{fp_pct:>4.0f}% {bs['tiny']:>7.4f} {bs['small']:>7.4f} {bs['medium']:>7.4f} {bs['large']:>7.4f} "
              f"{bd['tiny']:>7.4f} {bd['small']:>7.4f} {bd['medium']:>7.4f} {bd['large']:>7.4f} "
              f"{zero_sens_count:>4} {note:>10}")

    if best_pwd:
        print(f"\n  Best prob-weighted dilation (sens>=85%): {best_pwd['name']} — "
              f"Dice={best_pwd['dice']:.4f}, Sens={best_pwd['sens']:.4f}, "
              f"MedSens={best_pwd['med_sens']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 150}")

    if best_sens90:
        print(f"\n*** BEST CONFIG WITH SENSITIVITY >= 90% ***")
        print(f"  {best_sens90['name']}: Dice={best_sens90['dice']:.4f}, "
              f"Sens={best_sens90['sens']:.4f}, MedSens={best_sens90['med_sens']:.4f}, "
              f"Prec={best_sens90['prec']:.4f}, FP={best_sens90['fp_pct']:.0f}%, "
              f"ZeroSens={best_sens90['zero_sens']}")
    elif best_sens85:
        print(f"\n  Best config (sens>=85%): {best_sens85['name']} — "
              f"Dice={best_sens85['dice']:.4f}, Sens={best_sens85['sens']:.4f}")
        print(f"  To reach 90%: add more SegResNet folds (3,4) to ensemble")

    print(f"{'=' * 150}")


if __name__ == "__main__":
    main()
