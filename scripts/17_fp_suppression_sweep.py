"""
False Positive Suppression Sweep (CPU-only)
=============================================
Sweeps max_prob_threshold and min_volume_ml on saved probability maps
to find the optimal values that reduce false positives WITHOUT hurting
sensitivity on real lesions (especially tiny ones).

Reports results stratified by lesion size so you can see the impact on
tiny/small/medium/large lesions separately.

Usage:
  python scripts/17_fp_suppression_sweep.py --config configs/soop_config.yaml --split val
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
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return 2.0 * intersection / total


def compute_sensitivity(pred, gt):
    gt_sum = gt.sum()
    if gt_sum == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / gt_sum


def postprocess_with_fp_suppression(prob, threshold, min_component_size,
                                      max_prob_thresh, min_vol_ml,
                                      spacing=(1.0, 1.0, 1.0)):
    """Apply threshold + FP suppression."""
    # FP check 1: max probability too low
    if max_prob_thresh > 0 and prob.max() < max_prob_thresh:
        return np.zeros_like(prob, dtype=np.uint8)

    binary = (prob > threshold).astype(np.uint8)

    # FP check 2: total volume too small
    if min_vol_ml > 0:
        voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
        total_vol_ml = binary.sum() * voxel_vol_ml
        if total_vol_ml < min_vol_ml:
            return np.zeros_like(prob, dtype=np.uint8)

    # Remove small components
    labeled, n = ndimage.label(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_component_size:
            binary[labeled == i] = 0

    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)), iterations=1)
    return binary.astype(np.uint8)


def get_volume_bin(vol_ml, volume_bins):
    """Classify lesion volume into a bin."""
    if vol_ml == 0:
        return "none"
    for name, (lo, hi) in volume_bins.items():
        if lo <= vol_ml < hi:
            return name
    return "large"


def main():
    parser = argparse.ArgumentParser(description="FP suppression sweep")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    pred_dir = Path(config["paths"]["checkpoints"]).parent / "predictions" / args.split
    spacing = tuple(config["preprocessing"]["target_spacing"])
    volume_bins = config["preprocessing"]["volume_bins"]

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    # Load all probability maps and GT masks
    print(f"Loading {len(subject_ids)} probability maps...")
    subjects = []
    for sid in tqdm(subject_ids, desc="Loading"):
        prob_path = pred_dir / f"{sid}_prob.npy"
        gt_path = preproc_dir / f"{sid}.npz"

        if not prob_path.exists() or not gt_path.exists():
            continue

        prob = np.load(prob_path)
        gt_data = np.load(gt_path)
        gt_mask = gt_data["mask"].astype(np.uint8)

        gt_vol_ml = gt_mask.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
        size_bin = get_volume_bin(gt_vol_ml, volume_bins)

        subjects.append({
            "sid": sid,
            "prob": prob,
            "gt": gt_mask,
            "gt_vol_ml": gt_vol_ml,
            "size_bin": size_bin,
            "has_lesion": gt_mask.sum() > 0,
        })

    n_with_lesion = sum(1 for s in subjects if s["has_lesion"])
    n_no_lesion = sum(1 for s in subjects if not s["has_lesion"])
    print(f"Loaded: {len(subjects)} subjects ({n_with_lesion} with lesion, {n_no_lesion} no lesion)")

    # Sweep parameters
    max_prob_thresholds = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    min_vol_mls = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

    seg_threshold = 0.35
    min_component_size = 5

    combos = [(mp, mv) for mp in max_prob_thresholds for mv in min_vol_mls]

    print(f"\nSweeping {len(combos)} combinations...")
    print(f"  max_prob_threshold: {max_prob_thresholds}")
    print(f"  min_volume_ml: {min_vol_mls}")

    results = []

    for max_prob_t, min_vol in tqdm(combos, desc="Sweeping"):
        dices_all = []
        senss_all = []
        fp_count = 0
        n_no_lesion_total = 0

        # Stratified
        strat = {}
        for bin_name in list(volume_bins.keys()) + ["none"]:
            strat[bin_name] = {"dices": [], "senss": [], "missed": 0, "total": 0}

        for subj in subjects:
            pred = postprocess_with_fp_suppression(
                subj["prob"], seg_threshold, min_component_size,
                max_prob_t, min_vol, spacing
            )

            dice = compute_dice(pred, subj["gt"])
            sens = compute_sensitivity(pred, subj["gt"])

            dices_all.append(dice)
            senss_all.append(sens)

            # FP tracking
            if not subj["has_lesion"]:
                n_no_lesion_total += 1
                if pred.sum() > 0:
                    fp_count += 1

            # Stratified
            b = subj["size_bin"]
            strat[b]["dices"].append(dice)
            strat[b]["senss"].append(sens)
            strat[b]["total"] += 1
            if subj["has_lesion"] and pred.sum() == 0:
                strat[b]["missed"] += 1

        result = {
            "max_prob_threshold": max_prob_t,
            "min_volume_ml": min_vol,
            "dice_mean": round(np.mean(dices_all), 4),
            "dice_median": round(np.median(dices_all), 4),
            "sensitivity_mean": round(np.mean(senss_all), 4),
            "sensitivity_median": round(np.median(senss_all), 4),
            "fp_rate": round(fp_count / max(n_no_lesion_total, 1), 4),
            "fp_count": fp_count,
            "n_no_lesion": n_no_lesion_total,
        }

        # Add stratified tiny lesion stats
        if strat["tiny"]["dices"]:
            result["tiny_dice_mean"] = round(np.mean(strat["tiny"]["dices"]), 4)
            result["tiny_sens_mean"] = round(np.mean(strat["tiny"]["senss"]), 4)
            result["tiny_missed"] = strat["tiny"]["missed"]
            result["tiny_total"] = strat["tiny"]["total"]

        results.append(result)

    # Sort by dice_median descending
    results.sort(key=lambda r: r["dice_median"], reverse=True)

    # Print top results
    print(f"\n{'=' * 110}")
    print(f"TOP 15 CONFIGURATIONS (sorted by Dice median)")
    print(f"{'=' * 110}")
    print(f"{'MaxProb':>8} {'MinVol':>8} {'Dice Mean':>10} {'Dice Med':>10} {'Sens Mean':>10} "
          f"{'FP Rate':>8} {'FPs':>5} {'Tiny Dice':>10} {'Tiny Miss':>10}")
    print(f"{'-' * 110}")

    for r in results[:15]:
        tiny_dice = f"{r.get('tiny_dice_mean', 'N/A'):>10}" if 'tiny_dice_mean' in r else f"{'N/A':>10}"
        tiny_miss = f"{r.get('tiny_missed', 'N/A')}/{r.get('tiny_total', '?')}" if 'tiny_missed' in r else "N/A"
        print(f"{r['max_prob_threshold']:>8.2f} {r['min_volume_ml']:>8.2f} {r['dice_mean']:>10.4f} "
              f"{r['dice_median']:>10.4f} {r['sensitivity_mean']:>10.4f} "
              f"{r['fp_rate']:>8.2%} {r['fp_count']:>5} {tiny_dice} {tiny_miss:>10}")

    # Print best for each priority
    print(f"\n{'=' * 110}")
    print("RECOMMENDATIONS:")
    print(f"{'=' * 110}")

    # Best Dice median
    best_dice = results[0]
    print(f"  Best Dice median:  max_prob={best_dice['max_prob_threshold']}, "
          f"min_vol={best_dice['min_volume_ml']} → Dice {best_dice['dice_median']:.4f}, "
          f"Sens {best_dice['sensitivity_mean']:.4f}, FP {best_dice['fp_rate']:.0%}")

    # Best that doesn't miss any tiny lesions
    safe_results = [r for r in results if r.get("tiny_missed", 999) == 0]
    if safe_results:
        best_safe = safe_results[0]
        print(f"  Best SAFE (0 tiny missed): max_prob={best_safe['max_prob_threshold']}, "
              f"min_vol={best_safe['min_volume_ml']} → Dice {best_safe['dice_median']:.4f}, "
              f"Sens {best_safe['sensitivity_mean']:.4f}, FP {best_safe['fp_rate']:.0%}")

    # Best that cuts FP rate below 50%
    low_fp = [r for r in results if r["fp_rate"] <= 0.5]
    if low_fp:
        best_lowfp = low_fp[0]
        print(f"  Best low-FP (≤50%): max_prob={best_lowfp['max_prob_threshold']}, "
              f"min_vol={best_lowfp['min_volume_ml']} → Dice {best_lowfp['dice_median']:.4f}, "
              f"Sens {best_lowfp['sensitivity_mean']:.4f}, FP {best_lowfp['fp_rate']:.0%}")

    # Save results
    output_path = Path(config["paths"]["reports"]) if "reports" in config["paths"] else pred_dir
    results_path = output_path / "fp_suppression_sweep.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Full results: {results_path}")


if __name__ == "__main__":
    main()
