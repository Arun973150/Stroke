"""
Ensemble Inference — 6-Model Path B (Full-Brain)
=================================================
Runs all 6 models (3 nnU-Net + 3 SegResNet) on validation/test set,
averages logits (pre-sigmoid), and applies postprocessing.

Models:
  - nnU-Net fold 0, 1 (default trainer)
  - nnU-Net fold 2 (ISLES TopK trainer)
  - SegResNet fold 0, 1, 2 (Tversky + Deep Supervision)

Usage:
  python scripts/21_ensemble_inference.py --config configs/soop_config.yaml --split val
  python scripts/21_ensemble_inference.py --config configs/soop_config.yaml --split val --threshold 0.2
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── SegResNet Model (same as training script) ──────────────────────────────

class DeepSupSegResNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_filters=32, dropout_prob=0.2):
        super().__init__()
        self.backbone = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        self.ds_head_4x = torch.nn.Conv3d(init_filters * 4, out_channels, kernel_size=1)
        self.ds_head_2x = torch.nn.Conv3d(init_filters * 2, out_channels, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]
        out = self.backbone.convInit(x)
        encoder_outputs = []
        for down_layer in self.backbone.down_layers:
            out = down_layer(out)
            encoder_outputs.append(out)

        for i in range(len(self.backbone.up_layers)):
            out = self.backbone.up_samples[i](out)
            skip = encoder_outputs[-(i + 2)]
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode="trilinear", align_corners=False)
            out = out + skip
            out = self.backbone.up_layers[i](out)

        final = self.backbone.conv_final(out)
        return final


# ── Load Models ─────────────────────────────────────────────────────────────

def _load_one_segresnet(ckpt_path: Path, name: str, device="cuda"):
    """Load a single SegResNet checkpoint."""
    if not ckpt_path.exists():
        print(f"  WARNING: {name} not found at {ckpt_path}")
        return None

    model = DeepSupSegResNet(in_channels=3, out_channels=1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "full_model_state_dict" in ckpt:
        model.load_state_dict(ckpt["full_model_state_dict"])
    elif "model_state_dict" in ckpt:
        model.backbone.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    dice = ckpt.get("val_dice", 0)
    print(f"  Loaded {name} (val Dice: {dice:.4f})")
    return {"model": model, "name": name, "dice": dice}


def load_segresnet_models(checkpoint_dir: Path, folds=(0, 1, 2), device="cuda"):
    """Load SegResNet checkpoints for given folds."""
    models = []
    for fold in folds:
        ckpt_path = checkpoint_dir / f"fold_{fold}" / "checkpoint_best.pth"
        m = _load_one_segresnet(ckpt_path, f"SegResNet_fold{fold}", device)
        if m is not None:
            models.append(m)
    return models


def load_extra_segresnet_models(ckpt_paths, device="cuda"):
    """Load arbitrary additional SegResNet checkpoints by explicit path."""
    models = []
    for ckpt_path in ckpt_paths:
        p = Path(ckpt_path)
        # Name derived from parent folders: e.g. pathB_segresnet_synth_a0.8/fold_0
        name = f"{p.parent.parent.name}_{p.parent.name}"
        m = _load_one_segresnet(p, name, device)
        if m is not None:
            models.append(m)
    return models


def load_nnunet_predictor(nnunet_results: Path, dataset_name: str,
                           folds=(0, 1, 2), device="cuda"):
    """Load nnU-Net models using nnU-Net's predictor API."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # Find all trainer directories
    dataset_dir = nnunet_results / dataset_name
    if not dataset_dir.exists():
        print(f"  WARNING: nnU-Net dataset dir not found: {dataset_dir}")
        return None

    # Find trainer dirs (default + ISLES)
    trainer_dirs = sorted(dataset_dir.iterdir())
    print(f"  Found nnU-Net trainer dirs: {[d.name for d in trainer_dirs if d.is_dir()]}")

    # We'll use nnU-Net's predictor which handles multi-fold ensembling
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=True,  # TTA
        device=torch.device(device),
        verbose=False,
    )

    return predictor, dataset_dir


# ── TTA for SegResNet ───────────────────────────────────────────────────────

def tta_predict_segresnet(model, image_tensor, patch_size=(128, 128, 128)):
    """Run sliding window inference with 8-flip TTA, return averaged logits."""
    flip_axes = [
        [],        # no flip
        [2],       # flip D
        [3],       # flip H
        [4],       # flip W
        [2, 3],    # flip D+H
        [2, 4],    # flip D+W
        [3, 4],    # flip H+W
        [2, 3, 4], # flip all
    ]

    logit_sum = None
    for axes in flip_axes:
        x = torch.flip(image_tensor, axes) if axes else image_tensor
        with torch.no_grad():
            logits = sliding_window_inference(
                x, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=0.5, mode="gaussian",
            )
        # Flip back
        if axes:
            logits = torch.flip(logits, axes)
        if logit_sum is None:
            logit_sum = logits
        else:
            logit_sum = logit_sum + logits

    return logit_sum / len(flip_axes)


# ── Postprocessing ──────────────────────────────────────────────────────────

def postprocess(prob: np.ndarray, threshold: float = 0.20,
                min_component_size: int = 10,
                component_mean_threshold: float = 0.0,
                spacing: tuple = (1, 1, 1)) -> np.ndarray:
    """
    Sensitivity-focused postprocessing.

    Args:
        prob: probability map (0-1)
        threshold: binarization threshold (low = more sensitive)
        min_component_size: remove components smaller than this (voxels)
        component_mean_threshold: remove components with mean prob below this
        spacing: voxel spacing for volume calculation
    """
    binary = (prob > threshold).astype(np.uint8)

    if min_component_size > 0 or component_mean_threshold > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            comp_mask = labeled == i
            comp_size = comp_mask.sum()
            # Remove tiny components
            if comp_size < min_component_size:
                binary[comp_mask] = 0
                continue
            # Remove low-confidence components
            if component_mean_threshold > 0:
                comp_mean_prob = prob[comp_mask].mean()
                if comp_mean_prob < component_mean_threshold:
                    binary[comp_mask] = 0

    # Morphological closing
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)),
                                     iterations=1).astype(np.uint8)
    return binary


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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="6-model ensemble inference (Path B)")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--component-mean-threshold", type=float, default=0.0)
    parser.add_argument("--segresnet-folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--extra-segresnet-ckpts", nargs="+", default=[],
                        help="Paths to additional SegResNet checkpoint.pth files "
                             "(e.g. checkpoints/pathB_segresnet_synth_a0.8/fold_0/checkpoint_best.pth)")
    parser.add_argument("--nnunet-folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--skip-nnunet", action="store_true", help="Skip nnU-Net, only use SegResNet")
    parser.add_argument("--skip-segresnet", action="store_true", help="Skip SegResNet, only use nnU-Net")
    parser.add_argument("--nnunet-pred-dirs", nargs="+", default=None,
                        help="Pre-computed nnU-Net probability map directories")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preprocessed-dir", default=None,
                        help="Override preprocessed data dir (e.g. isles_preprocessed/ for external validation)")
    parser.add_argument("--splits-dir", default=None,
                        help="Override splits dir (defaults to config value)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(args.preprocessed_dir) if args.preprocessed_dir else Path(config["paths"]["preprocessed"])
    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"])
    spacing = tuple(config["preprocessing"]["target_spacing"])
    patch_size = (args.patch_size,) * 3

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else (
        ckpt_dir.parent / "predictions_ensemble" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'=' * 70}")
    print(f"ENSEMBLE INFERENCE — Path B (Full-Brain)")
    print(f"{'=' * 70}")
    print(f"  Split: {args.split} ({len(subject_ids)} subjects)")
    print(f"  Threshold: {args.threshold}")
    print(f"  Min component size: {args.min_component_size}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")

    # ── Load SegResNet models ──
    segresnet_models = []
    if not args.skip_segresnet:
        print(f"\nLoading SegResNet models...")
        segresnet_models = load_segresnet_models(
            ckpt_dir / "pathB_segresnet", folds=args.segresnet_folds, device=device
        )
        if args.extra_segresnet_ckpts:
            print(f"  Loading {len(args.extra_segresnet_ckpts)} extra SegResNet checkpoints...")
            segresnet_models.extend(
                load_extra_segresnet_models(args.extra_segresnet_ckpts, device=device)
            )
        print(f"  Loaded {len(segresnet_models)} SegResNet models total")

    # ── Load nnU-Net predictions ──
    # nnU-Net probability maps from nnUNetv2_predict --save_probabilities
    nnunet_prob_dirs = []
    if not args.skip_nnunet:
        print(f"\nLooking for nnU-Net predictions...")
        if args.nnunet_pred_dirs:
            for d in args.nnunet_pred_dirs:
                nnunet_prob_dirs.append(Path(d))
                print(f"  Using provided: {d}")
        else:
            # Auto-detect from sentinel_stroke directory
            for dirname in ["predictions_nnunet_default", "predictions_nnunet_isles"]:
                pred_dir = ckpt_dir.parent / dirname
                if pred_dir.exists():
                    nnunet_prob_dirs.append(pred_dir)
                    print(f"  Found: {dirname}")

        if not nnunet_prob_dirs:
            print("  WARNING: No nnU-Net predictions found!")
            print("  Run nnU-Net inference first or use --nnunet-pred-dirs")
            if not segresnet_models:
                print("  ERROR: No models loaded. Exiting.")
                return

    print(f"\n  Total SegResNet models: {len(segresnet_models)}")
    print(f"  nnU-Net prediction dirs: {len(nnunet_prob_dirs)}")

    # ── Run inference ──
    print(f"\n{'=' * 70}")
    print("Running ensemble inference...")
    print(f"{'=' * 70}\n")

    all_metrics = []
    bin_metrics = {"tiny": [], "small": [], "medium": [], "large": [], "none": []}

    for sid in tqdm(subject_ids, desc="Subjects"):
        # Load preprocessed data
        npz_path = preproc_dir / f"{sid}.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        image = data["image"].astype(np.float32)  # (3, D, H, W)
        gt = data["mask"].astype(bool)

        gt_vol_ml = gt.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0
        vol_bin = classify_volume_bin(gt_vol_ml)

        # ── SegResNet ensemble (logit averaging with TTA) ──
        logit_sum = None
        n_contrib = 0

        if segresnet_models:
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 3, D, H, W)

            for m_info in segresnet_models:
                model = m_info["model"]
                logits = tta_predict_segresnet(model, image_tensor, patch_size)
                logits_np = logits.cpu().numpy()[0, 0]  # (D, H, W)

                if logit_sum is None:
                    logit_sum = logits_np.astype(np.float64)
                else:
                    logit_sum += logits_np.astype(np.float64)
                n_contrib += 1

        # ── nnU-Net predictions (load pre-computed from nnUNetv2_predict) ──
        for pred_dir in nnunet_prob_dirs:
            # nnUNetv2_predict saves {sid}.npz directly in output dir
            prob_path = pred_dir / f"{sid}.npz"
            if not prob_path.exists():
                continue

            prob_npz = np.load(prob_path)
            key = prob_npz.files[0]
            raw_prob = prob_npz[key]

            # nnU-Net saves softmax: (n_classes, D, H, W) — take lesion class
            if raw_prob.ndim == 4 and raw_prob.shape[0] >= 2:
                prob = raw_prob[1].astype(np.float64)
            elif raw_prob.ndim == 3:
                prob = raw_prob.astype(np.float64)
            else:
                continue

            # Convert probability to logit for averaging
            prob_clipped = np.clip(prob, 1e-7, 1 - 1e-7)
            logit = np.log(prob_clipped / (1 - prob_clipped))

            if logit_sum is None:
                logit_sum = logit
            else:
                # Resize if shapes differ
                if logit.shape != logit_sum.shape:
                    logit_t = torch.from_numpy(logit).unsqueeze(0).unsqueeze(0).float()
                    logit_t = F.interpolate(logit_t, size=logit_sum.shape,
                                            mode="trilinear", align_corners=False)
                    logit = logit_t.numpy()[0, 0]
                logit_sum += logit
            n_contrib += 1

        if logit_sum is None or n_contrib == 0:
            # No predictions available for this subject
            all_metrics.append({
                "subject_id": sid, "dice": 0, "sensitivity": 0,
                "specificity": 1, "precision": 0,
                "gt_vol_ml": gt_vol_ml, "vol_bin": vol_bin,
            })
            bin_metrics[vol_bin].append(all_metrics[-1])
            continue

        # Average logits → probability
        avg_logit = logit_sum / n_contrib
        ensemble_prob = 1.0 / (1.0 + np.exp(-avg_logit))

        # Save probability map
        np.save(output_dir / f"{sid}_prob.npy", ensemble_prob.astype(np.float32))

        # Postprocess
        pred = postprocess(
            ensemble_prob, threshold=args.threshold,
            min_component_size=args.min_component_size,
            component_mean_threshold=args.component_mean_threshold,
            spacing=spacing,
        )
        np.save(output_dir / f"{sid}_pred.npy", pred)

        # Compute metrics
        m = compute_metrics(pred, gt, spacing)
        m["subject_id"] = sid
        m["vol_bin"] = vol_bin
        m["n_models"] = n_contrib
        all_metrics.append(m)
        bin_metrics[vol_bin].append(m)

    # ── Results ──
    print(f"\n{'=' * 90}")
    print(f"ENSEMBLE RESULTS — {n_contrib} models, threshold={args.threshold}")
    print(f"{'=' * 90}")

    pos_metrics = [m for m in all_metrics if m.get("gt_vol_ml", 0) > 0]
    if pos_metrics:
        avg_dice = np.mean([m["dice"] for m in pos_metrics])
        avg_sens = np.mean([m["sensitivity"] for m in pos_metrics])
        avg_spec = np.mean([m["specificity"] for m in pos_metrics])
        avg_prec = np.mean([m["precision"] for m in pos_metrics])
        med_sens = np.median([m["sensitivity"] for m in pos_metrics])

        print(f"\nOverall (lesion-positive, n={len(pos_metrics)}):")
        print(f"  Mean Dice:        {avg_dice:.4f}")
        print(f"  Mean Sensitivity: {avg_sens:.4f}")
        print(f"  Med  Sensitivity: {med_sens:.4f}")
        print(f"  Mean Specificity: {avg_spec:.4f}")
        print(f"  Mean Precision:   {avg_prec:.4f}")

        # Per-bin
        print(f"\nPer volume bin:")
        print(f"  {'Bin':<10} {'N':>5} {'Dice':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8}")
        print(f"  {'-' * 50}")
        for b in ["tiny", "small", "medium", "large"]:
            bm = [m for m in bin_metrics[b] if m.get("gt_vol_ml", 0) > 0]
            if bm:
                print(f"  {b:<10} {len(bm):>5} {np.mean([m['dice'] for m in bm]):>8.4f} "
                      f"{np.mean([m['sensitivity'] for m in bm]):>8.4f} "
                      f"{np.mean([m['specificity'] for m in bm]):>8.4f} "
                      f"{np.mean([m['precision'] for m in bm]):>8.4f}")

        # FP rate on negative cases
        neg = bin_metrics.get("none", [])
        if neg:
            fp_count = sum(1 for m in neg if m.get("pred_vol_ml", 0) > 0)
            print(f"\n  Negative cases: {len(neg)}, False positives: {fp_count} ({100*fp_count/len(neg):.1f}%)")

        # Sensitivity distribution
        sens_vals = [m["sensitivity"] for m in pos_metrics]
        print(f"\nSensitivity distribution:")
        print(f"  >=0.90: {sum(1 for s in sens_vals if s >= 0.90)}/{len(sens_vals)} "
              f"({100*sum(1 for s in sens_vals if s >= 0.90)/len(sens_vals):.1f}%)")
        print(f"  >=0.80: {sum(1 for s in sens_vals if s >= 0.80)}/{len(sens_vals)} "
              f"({100*sum(1 for s in sens_vals if s >= 0.80)/len(sens_vals):.1f}%)")
        print(f"  >=0.50: {sum(1 for s in sens_vals if s >= 0.50)}/{len(sens_vals)} "
              f"({100*sum(1 for s in sens_vals if s >= 0.50)/len(sens_vals):.1f}%)")
        print(f"  ==0.00: {sum(1 for s in sens_vals if s == 0)}/{len(sens_vals)} "
              f"({100*sum(1 for s in sens_vals if s == 0)/len(sens_vals):.1f}%)")

    # Save summary
    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "min_component_size": args.min_component_size,
        "n_subjects": len(all_metrics),
        "n_models": n_contrib if 'n_contrib' in dir() else 0,
        "mean_dice": float(avg_dice) if pos_metrics else 0,
        "mean_sensitivity": float(avg_sens) if pos_metrics else 0,
        "median_sensitivity": float(med_sens) if pos_metrics else 0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProbability maps saved to: {output_dir}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
