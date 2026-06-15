"""
Compare Path A (Cascade) vs Path B (Standalone nnU-Net) and Pick Best
======================================================================
For each subject, compares predictions from:
  - Path A: cascade pipeline (Stage 1 → ROI → Stage 2 ensemble)
  - Path B: standalone full-brain nnU-Net

Selection strategies:
  1. "union"     — merge both predictions (catch more lesions, higher sensitivity)
  2. "confidence"— pick the prediction with higher mean probability in lesion region
  3. "larger"    — pick the prediction with more lesion voxels (favors sensitivity)
  4. "pathA_unless_empty" — use Path A, but fall back to Path B if Path A finds nothing

Also evaluates each path independently + the combined result.

Usage:
  python scripts/16_compare_pathA_pathB.py --config configs/soop_config.yaml --split val
  python scripts/16_compare_pathA_pathB.py --config configs/soop_config.yaml --split val --strategy union
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return 2.0 * intersection / total


def compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_sum = gt.sum()
    if gt_sum == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / gt_sum


def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    neg = (gt == 0).sum()
    if neg == 0:
        return 1.0
    return np.logical_and(pred == 0, gt == 0).sum() / neg


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")
    fwd = directed_hausdorff(pred_pts, gt_pts)[0]
    bwd = directed_hausdorff(gt_pts, pred_pts)[0]
    # Approximate HD95 using max of directed distances
    # (true HD95 would need all pairwise distances, but this is fast)
    return max(fwd, bwd)


# ── Combination Strategies ──────────────────────────────────────────────────

def combine_union(pred_a: np.ndarray, pred_b: np.ndarray,
                  prob_a: np.ndarray = None, prob_b: np.ndarray = None) -> np.ndarray:
    """Union of both predictions — maximizes sensitivity."""
    return np.logical_or(pred_a, pred_b).astype(np.uint8)


def combine_confidence(pred_a: np.ndarray, pred_b: np.ndarray,
                       prob_a: np.ndarray = None, prob_b: np.ndarray = None) -> np.ndarray:
    """Pick prediction with higher average probability in predicted region."""
    if prob_a is None or prob_b is None:
        # Fall back to larger
        return combine_larger(pred_a, pred_b)

    conf_a = prob_a[pred_a > 0].mean() if pred_a.sum() > 0 else 0.0
    conf_b = prob_b[pred_b > 0].mean() if pred_b.sum() > 0 else 0.0

    return pred_a if conf_a >= conf_b else pred_b


def combine_larger(pred_a: np.ndarray, pred_b: np.ndarray,
                   prob_a: np.ndarray = None, prob_b: np.ndarray = None) -> np.ndarray:
    """Pick prediction with more lesion voxels (favors sensitivity)."""
    return pred_a if pred_a.sum() >= pred_b.sum() else pred_b


def combine_pathA_unless_empty(pred_a: np.ndarray, pred_b: np.ndarray,
                                prob_a: np.ndarray = None, prob_b: np.ndarray = None) -> np.ndarray:
    """Use Path A (cascade) by default, fall back to Path B if Path A finds nothing."""
    if pred_a.sum() > 0:
        return pred_a
    return pred_b


def combine_prob_average(pred_a: np.ndarray, pred_b: np.ndarray,
                         prob_a: np.ndarray = None, prob_b: np.ndarray = None,
                         threshold: float = 0.35) -> np.ndarray:
    """Average probability maps from both paths, then threshold."""
    if prob_a is None or prob_b is None:
        return combine_union(pred_a, pred_b)

    # Ensure same shape
    if prob_a.shape != prob_b.shape:
        # Resize prob_b to match prob_a
        import torch
        import torch.nn.functional as F
        prob_b_t = torch.from_numpy(prob_b).float().unsqueeze(0).unsqueeze(0)
        prob_b_t = F.interpolate(prob_b_t, size=prob_a.shape, mode="trilinear", align_corners=False)
        prob_b = prob_b_t.squeeze().numpy()

    avg_prob = (prob_a + prob_b) / 2.0
    binary = (avg_prob > threshold).astype(np.uint8)

    # Remove tiny components
    labeled, n = ndimage.label(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() < 5:
            binary[labeled == i] = 0

    return binary


STRATEGIES = {
    "union": combine_union,
    "confidence": combine_confidence,
    "larger": combine_larger,
    "pathA_unless_empty": combine_pathA_unless_empty,
    "prob_average": combine_prob_average,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Path A (cascade) vs Path B (standalone)")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--strategy", default="all",
                        choices=["union", "confidence", "larger", "pathA_unless_empty",
                                 "prob_average", "all"])
    parser.add_argument("--pathA-dir", default=None, help="Override Path A predictions dir")
    parser.add_argument("--pathB-dir", default=None, help="Override Path B predictions dir")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    base_dir = Path(config["paths"]["checkpoints"]).parent

    pathA_dir = Path(args.pathA_dir) if args.pathA_dir else base_dir / "predictions" / args.split
    pathB_dir = Path(args.pathB_dir) if args.pathB_dir else base_dir / "predictions_pathB" / args.split

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    print(f"Comparing Path A (cascade) vs Path B (standalone nnU-Net)")
    print(f"  Split: {args.split} ({len(subject_ids)} subjects)")
    print(f"  Path A: {pathA_dir}")
    print(f"  Path B: {pathB_dir}")

    strategies_to_run = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    # Evaluate each path and strategy
    all_results = {"pathA": [], "pathB": []}
    for s in strategies_to_run:
        all_results[s] = []

    for sid in tqdm(subject_ids, desc="Comparing"):
        # Load GT
        gt_path = preproc_dir / f"{sid}.npz"
        if not gt_path.exists():
            continue
        gt_data = np.load(gt_path)
        gt_mask = gt_data["mask"].astype(np.uint8)

        # Load Path A prediction
        predA_path = pathA_dir / f"{sid}_pred.npy"
        probA_path = pathA_dir / f"{sid}_prob.npy"
        pred_a = np.load(predA_path) if predA_path.exists() else np.zeros_like(gt_mask)
        prob_a = np.load(probA_path) if probA_path.exists() else None

        # Load Path B prediction
        predB_path = pathB_dir / f"{sid}_pred.npy"
        probB_path = pathB_dir / f"{sid}_prob.npy"
        pred_b = np.load(predB_path) if predB_path.exists() else np.zeros_like(gt_mask)
        prob_b = np.load(probB_path) if probB_path.exists() else None

        # Handle shape mismatch (Path B may be different size)
        if pred_a.shape != pred_b.shape and pred_b.sum() > 0:
            import torch
            import torch.nn.functional as F
            target_shape = pred_a.shape
            pred_b_t = torch.from_numpy(pred_b.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            pred_b_t = F.interpolate(pred_b_t, size=target_shape, mode="nearest")
            pred_b = pred_b_t.squeeze().numpy().astype(np.uint8)

            if prob_b is not None:
                prob_b_t = torch.from_numpy(prob_b).float().unsqueeze(0).unsqueeze(0)
                prob_b_t = F.interpolate(prob_b_t, size=target_shape, mode="trilinear", align_corners=False)
                prob_b = prob_b_t.squeeze().numpy()

        # Evaluate Path A
        all_results["pathA"].append({
            "subject_id": sid,
            "dice": compute_dice(pred_a, gt_mask),
            "sensitivity": compute_sensitivity(pred_a, gt_mask),
            "specificity": compute_specificity(pred_a, gt_mask),
        })

        # Evaluate Path B
        all_results["pathB"].append({
            "subject_id": sid,
            "dice": compute_dice(pred_b, gt_mask),
            "sensitivity": compute_sensitivity(pred_b, gt_mask),
            "specificity": compute_specificity(pred_b, gt_mask),
        })

        # Evaluate each combination strategy
        for strategy_name in strategies_to_run:
            combine_fn = STRATEGIES[strategy_name]
            combined = combine_fn(pred_a, pred_b, prob_a, prob_b)

            all_results[strategy_name].append({
                "subject_id": sid,
                "dice": compute_dice(combined, gt_mask),
                "sensitivity": compute_sensitivity(combined, gt_mask),
                "specificity": compute_specificity(combined, gt_mask),
            })

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"COMPARISON RESULTS ({args.split}, {len(subject_ids)} subjects)")
    print(f"{'=' * 70}")
    print(f"{'Method':<25} {'Dice Mean':>10} {'Dice Med':>10} {'Sens':>10} {'Spec':>10}")
    print(f"{'-' * 70}")

    best_dice = 0
    best_method = ""

    for method_name in ["pathA", "pathB"] + strategies_to_run:
        results = all_results[method_name]
        if not results:
            continue

        dices = [r["dice"] for r in results]
        senss = [r["sensitivity"] for r in results]
        specs = [r["specificity"] for r in results]

        dice_mean = np.mean(dices)
        dice_med = np.median(dices)
        sens_mean = np.mean(senss)
        spec_mean = np.mean(specs)

        marker = ""
        if dice_med > best_dice:
            best_dice = dice_med
            best_method = method_name

        print(f"{method_name:<25} {dice_mean:>10.4f} {dice_med:>10.4f} {sens_mean:>10.4f} {spec_mean:>10.4f}")

    print(f"{'-' * 70}")
    print(f"  Best by Dice median: {best_method} ({best_dice:.4f})")
    print(f"{'=' * 70}")

    # Save detailed results
    output_dir = base_dir / "comparison_results" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, results in all_results.items():
        if results:
            out_path = output_dir / f"{method_name}_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save summary
    summary = {}
    for method_name, results in all_results.items():
        if not results:
            continue
        dices = [r["dice"] for r in results]
        senss = [r["sensitivity"] for r in results]
        specs = [r["specificity"] for r in results]
        summary[method_name] = {
            "dice_mean": round(float(np.mean(dices)), 4),
            "dice_median": round(float(np.median(dices)), 4),
            "sensitivity_mean": round(float(np.mean(senss)), 4),
            "specificity_mean": round(float(np.mean(specs)), 4),
        }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detailed results: {output_dir}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
