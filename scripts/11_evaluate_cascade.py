"""
Cascade Evaluation (Phase 6)
==============================
Evaluates cascade predictions against ground truth with stratified analysis.

Metrics: Dice, Sensitivity, Specificity, Hausdorff95, Surface Dice
Stratified by: lesion volume bin (tiny/small/medium/large)

Usage:
  python scripts/11_evaluate_cascade.py --config configs/soop_config.yaml --split test
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    intersection = (pred & gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


def compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum()
    fn = (gt & ~pred).sum()
    return tp / max(tp + fn, 1)


def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum()
    return tn / max(tn + fp, 1)


def compute_hausdorff95(pred: np.ndarray, gt: np.ndarray, spacing=(1, 1, 1)) -> float:
    """Compute 95th percentile Hausdorff distance."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf") if pred.sum() != gt.sum() else 0.0

    pred_surface = _get_surface_points(pred, spacing)
    gt_surface = _get_surface_points(gt, spacing)

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float("inf")

    # Forward and backward distances
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_surface)
    tree_pred = cKDTree(pred_surface)

    dist_pred_to_gt, _ = tree_gt.query(pred_surface)
    dist_gt_to_pred, _ = tree_pred.query(gt_surface)

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def _get_surface_points(mask: np.ndarray, spacing=(1, 1, 1)) -> np.ndarray:
    """Extract surface voxel coordinates."""
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    coords = np.array(np.where(surface)).T.astype(float)
    coords *= np.array(spacing)
    return coords


def compute_volume_ml(mask: np.ndarray, spacing=(1, 1, 1)) -> float:
    voxel_vol = float(np.prod(spacing))
    return mask.sum() * voxel_vol / 1000.0


def classify_volume_bin(volume_ml: float) -> str:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate cascade predictions")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--pred-dir", default=None, help="Override prediction directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        Path(config["paths"]["checkpoints"]).parent / "predictions" / args.split
    )
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    print(f"Evaluating {args.split} ({len(subject_ids)} subjects)")
    print(f"Predictions: {pred_dir}")

    spacing = tuple(config["preprocessing"]["target_spacing"])
    rows = []

    for sid in tqdm(subject_ids, desc="Evaluating"):
        # Load ground truth
        npz_path = preproc_dir / f"{sid}.npz"
        data = np.load(npz_path)
        gt = data["mask"].astype(bool)

        # Load prediction
        pred_path = pred_dir / f"{sid}_pred.npy"
        if not pred_path.exists():
            print(f"  WARNING: Missing prediction for {sid}")
            continue
        pred = np.load(pred_path).astype(bool)

        gt_volume = compute_volume_ml(gt, spacing)
        pred_volume = compute_volume_ml(pred, spacing)

        row = {
            "subject_id": sid,
            "gt_volume_ml": round(gt_volume, 4),
            "pred_volume_ml": round(pred_volume, 4),
            "volume_bin": classify_volume_bin(gt_volume),
            "dice": round(compute_dice(pred, gt), 4),
            "sensitivity": round(compute_sensitivity(pred, gt), 4),
            "specificity": round(compute_specificity(pred, gt), 4),
            "hausdorff95": round(compute_hausdorff95(pred, gt, spacing), 2),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save per-subject results
    results_csv = reports_dir / f"eval_{args.split}.csv"
    df.to_csv(results_csv, index=False)

    # Overall metrics
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS ({args.split})")
    print(f"{'=' * 70}")

    # Overall
    print(f"\n  OVERALL (n={len(df)}):")
    for metric in ["dice", "sensitivity", "specificity", "hausdorff95"]:
        vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    {metric:15s}: {vals.mean():.4f} +/- {vals.std():.4f} "
              f"(median: {vals.median():.4f})")

    # Stratified by volume bin
    print(f"\n  STRATIFIED BY LESION SIZE:")
    for vol_bin in ["tiny", "small", "medium", "large", "none"]:
        subset = df[df["volume_bin"] == vol_bin]
        if len(subset) == 0:
            continue
        dice_vals = subset["dice"]
        print(f"\n    {vol_bin.upper()} (n={len(subset)}):")
        print(f"      Dice:        {dice_vals.mean():.4f} +/- {dice_vals.std():.4f}")
        print(f"      Sensitivity: {subset['sensitivity'].mean():.4f}")
        hd = subset["hausdorff95"].replace([np.inf], np.nan).dropna()
        if len(hd) > 0:
            print(f"      HD95:        {hd.mean():.2f}")

    # False positive analysis (predictions on no-lesion cases)
    no_lesion = df[df["volume_bin"] == "none"]
    if len(no_lesion) > 0:
        fp_cases = no_lesion[no_lesion["pred_volume_ml"] > 0]
        print(f"\n  FALSE POSITIVE ANALYSIS:")
        print(f"    No-lesion cases: {len(no_lesion)}")
        print(f"    False positives: {len(fp_cases)} ({len(fp_cases)/len(no_lesion):.1%})")

    # Save summary
    summary = {
        "split": args.split,
        "n_subjects": len(df),
        "overall": {
            metric: {
                "mean": round(float(df[metric].replace([np.inf], np.nan).mean()), 4),
                "std": round(float(df[metric].replace([np.inf], np.nan).std()), 4),
                "median": round(float(df[metric].replace([np.inf], np.nan).median()), 4),
            }
            for metric in ["dice", "sensitivity", "specificity", "hausdorff95"]
        },
        "stratified": {},
    }
    for vol_bin in ["tiny", "small", "medium", "large"]:
        subset = df[df["volume_bin"] == vol_bin]
        if len(subset) > 0:
            summary["stratified"][vol_bin] = {
                "n": len(subset),
                "dice_mean": round(float(subset["dice"].mean()), 4),
                "dice_std": round(float(subset["dice"].std()), 4),
                "sensitivity_mean": round(float(subset["sensitivity"].mean()), 4),
            }

    summary_path = reports_dir / f"eval_{args.split}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results CSV: {results_csv}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
