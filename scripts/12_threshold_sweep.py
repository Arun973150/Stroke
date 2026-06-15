"""
Threshold Sweep + TTA-style Analysis
======================================
Finds optimal threshold on validation probability maps.
Also checks per-subject failure cases.

Usage:
  python scripts/12_threshold_sweep.py --config configs/soop_config.yaml --split val
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


def compute_dice(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    intersection = (pred & gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


def compute_sensitivity(pred, gt):
    tp = (pred & gt).sum()
    fn = (gt & ~pred).sum()
    return tp / max(tp + fn, 1)


def postprocess(binary, min_size=10):
    labeled, n = ndimage.label(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            binary[labeled == i] = 0
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)), iterations=1)
    return binary.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    pred_dir = Path(config["paths"]["checkpoints"]).parent / "predictions" / args.split

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    # Load all probability maps and ground truths
    print(f"Loading {len(subject_ids)} subjects...")
    data_pairs = []
    skipped = 0
    for sid in tqdm(subject_ids, desc="Loading"):
        prob_path = pred_dir / f"{sid}_prob.npy"
        npz_path = preproc_dir / f"{sid}.npz"
        if not prob_path.exists():
            skipped += 1
            continue
        prob = np.load(prob_path)
        gt = np.load(npz_path)["mask"].astype(bool)
        data_pairs.append((sid, prob, gt))

    print(f"Loaded: {len(data_pairs)}, Skipped: {skipped}")

    # ── Threshold sweep ──
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    min_sizes = [0, 5, 10, 20, 50]

    print(f"\n{'='*80}")
    print("THRESHOLD SWEEP")
    print(f"{'='*80}")
    print(f"{'Thresh':>8} {'MinSize':>8} {'Dice_mean':>10} {'Dice_med':>10} {'Sens_mean':>10} {'Sens_med':>10}")
    print("-" * 60)

    best_dice = 0
    best_config = {}

    combos = [(t, m) for t in thresholds for m in min_sizes]
    for thresh, min_size in tqdm(combos, desc="Sweeping"):
        if True:
            dices = []
            senss = []
            for sid, prob, gt in data_pairs:
                binary = (prob > thresh).astype(np.uint8)
                if min_size > 0:
                    binary = postprocess(binary, min_size)
                pred = binary.astype(bool)
                dices.append(compute_dice(pred, gt))
                senss.append(compute_sensitivity(pred, gt))

            mean_dice = np.mean(dices)
            med_dice = np.median(dices)
            mean_sens = np.mean(senss)
            med_sens = np.median(senss)

            marker = ""
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_config = {"threshold": thresh, "min_size": min_size,
                               "dice_mean": round(mean_dice, 4),
                               "dice_median": round(med_dice, 4),
                               "sensitivity_mean": round(mean_sens, 4)}
                marker = " ← BEST"

            # Only print min_size=10 for all thresholds, and all min_sizes for best threshold range
            if min_size == 10 or thresh in [0.3, 0.35, 0.4]:
                print(f"{thresh:>8.2f} {min_size:>8d} {mean_dice:>10.4f} {med_dice:>10.4f} "
                      f"{mean_sens:>10.4f} {med_sens:>10.4f}{marker}")

    print(f"\n{'='*80}")
    print(f"BEST CONFIG: threshold={best_config['threshold']}, min_component_size={best_config['min_size']}")
    print(f"  Dice:        {best_config['dice_mean']:.4f} (mean), {best_config['dice_median']:.4f} (median)")
    print(f"  Sensitivity: {best_config['sensitivity_mean']:.4f}")
    print(f"{'='*80}")

    # ── Failure analysis ──
    print(f"\n{'='*80}")
    print("FAILURE ANALYSIS (worst 15 subjects at best threshold)")
    print(f"{'='*80}")

    thresh = best_config["threshold"]
    results = []
    for sid, prob, gt in data_pairs:
        binary = (prob > thresh).astype(np.uint8)
        binary = postprocess(binary, best_config["min_size"])
        pred = binary.astype(bool)
        d = compute_dice(pred, gt)
        s = compute_sensitivity(pred, gt)
        gt_vol = gt.sum() / 1000.0
        results.append({"sid": sid, "dice": d, "sens": s, "gt_vol_ml": gt_vol,
                        "pred_vol_ml": pred.sum() / 1000.0})

    results.sort(key=lambda x: x["dice"])
    print(f"{'Subject':>12} {'Dice':>8} {'Sens':>8} {'GT_ml':>8} {'Pred_ml':>8}")
    for r in results[:15]:
        print(f"{r['sid']:>12} {r['dice']:>8.4f} {r['sens']:>8.4f} "
              f"{r['gt_vol_ml']:>8.2f} {r['pred_vol_ml']:>8.2f}")

    # ── Save best config ──
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "best_threshold_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved: {reports_dir / 'best_threshold_config.json'}")


if __name__ == "__main__":
    main()
