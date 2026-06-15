"""
Stage 1 Detection Threshold Sweep
===================================
Re-runs Stage 1 at different thresholds, generates new ROIs,
runs Stage 2 ensemble on each, and evaluates.

This finds the optimal Stage 1 detection threshold.

Usage:
  python scripts/13_stage1_threshold_sweep.py --config configs/soop_config.yaml --split val
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.networks.nets import SegResNet
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


def load_stage1_model(ckpt_path, device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=16, in_channels=3, out_channels=1, dropout_prob=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_segresnet_fold(ckpt_path, device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=32, in_channels=3, out_channels=1, dropout_prob=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def run_stage1(model, image, input_shape, full_shape, threshold, margin_expand, device):
    image_t = torch.from_numpy(image).float().unsqueeze(0)
    image_low = F.interpolate(image_t, size=input_shape, mode="trilinear",
                              align_corners=False).to(device)
    with torch.no_grad():
        output = model(image_low)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return [], prob

    labeled, n = ndimage.label(binary)
    scale = [f / l for f, l in zip(full_shape, input_shape)]
    rois = []

    for i in range(1, n + 1):
        coords = np.where(labeled == i)
        bbox_min = [max(0, int(c.min() * s - (c.max() - c.min()) * s * margin_expand))
                    for c, s in zip(coords, scale)]
        bbox_max = [min(fs, int((c.max() + 1) * s + (c.max() - c.min()) * s * margin_expand))
                    for c, s, fs in zip(coords, scale, full_shape)]
        rois.append({"bbox_min": bbox_min, "bbox_max": bbox_max})

    return rois, prob


def run_stage2_ensemble(models, image, roi, crop_size, device):
    bbox_min = roi["bbox_min"]
    bbox_max = roi["bbox_max"]
    crop = image[:, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
    crop_t = torch.from_numpy(crop).float().unsqueeze(0)
    crop_resized = F.interpolate(crop_t, size=crop_size, mode="trilinear",
                                 align_corners=False).to(device)

    all_preds = []
    for model in models:
        with torch.no_grad():
            out = torch.sigmoid(model(crop_resized))
        all_preds.append(out.cpu())

    combined = torch.stack(all_preds, dim=0).mean(dim=0)
    pred_crop = combined.squeeze().numpy()

    roi_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    pred_t = torch.from_numpy(pred_crop).float().unsqueeze(0).unsqueeze(0)
    pred_full = F.interpolate(pred_t, size=roi_size, mode="trilinear",
                              align_corners=False).squeeze().numpy()
    return pred_full


def gaussian_weight(shape, sigma_fraction=0.3):
    grids = [np.linspace(-1, 1, s) for s in shape]
    z, y, x = np.meshgrid(*grids, indexing="ij")
    w = np.exp(-(z**2 + y**2 + x**2) / (2 * sigma_fraction**2))
    return w.astype(np.float32)


def stitch_predictions(full_shape, rois, roi_predictions):
    output = np.zeros(full_shape, dtype=np.float32)
    weight = np.zeros(full_shape, dtype=np.float32)
    for roi, pred in zip(rois, roi_predictions):
        bmin, bmax = roi["bbox_min"], roi["bbox_max"]
        roi_size = [bmax[i] - bmin[i] for i in range(3)]
        gauss = gaussian_weight(roi_size)
        output[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] += pred * gauss
        weight[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] += gauss
    mask = weight > 0
    output[mask] /= weight[mask]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, Total: {gpu_mem:.1f}GB, Free: {gpu_free:.1f}GB")

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"])

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    # Load Stage 1 model
    print("Loading Stage 1 model...")
    stage1_model = load_stage1_model(
        ckpt_dir / "stage1_detection" / "checkpoint_best.pth", device)

    # Load Stage 2 SegResNet models
    print("Loading Stage 2 SegResNet models...")
    stage2_models = []
    for fold_dir in sorted(ckpt_dir.glob("stage2_segresnet/fold_*")):
        ckpt_path = fold_dir / "checkpoint_best.pth"
        if ckpt_path.exists():
            stage2_models.append(load_segresnet_fold(ckpt_path, device))
            print(f"  Loaded {fold_dir.name}")
    print(f"Total Stage 2 models: {len(stage2_models)}")

    stage1_cfg = config["training"]["stage1"]
    stage2_cfg = config["training"]["stage2"]
    input_shape = tuple(stage1_cfg["input_shape"])
    crop_size = (128, 128, 128)
    margin_expand = stage2_cfg["roi_margin_expand"]

    # Stage 1 thresholds to sweep
    s1_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    # Stage 2 threshold (use default 0.5, or best from sweep if available)
    s2_threshold = 0.5
    best_config_path = Path(config["paths"]["reports"]) / "best_threshold_config.json"
    if best_config_path.exists():
        with open(best_config_path) as f:
            best = json.load(f)
        s2_threshold = best.get("threshold", 0.5)
        print(f"Using Stage 2 threshold from sweep: {s2_threshold}")

    print(f"\nSweeping Stage 1 thresholds: {s1_thresholds}")
    print(f"Stage 2 threshold: {s2_threshold}")
    print(f"Subjects: {len(subject_ids)}")

    # Pre-compute Stage 1 probabilities (run once, threshold multiple times)
    print("\nRunning Stage 1 inference on all subjects...")
    stage1_probs = {}
    gts = {}
    for sid in tqdm(subject_ids, desc="Stage 1"):
        npz_path = preproc_dir / f"{sid}.npz"
        data = np.load(npz_path)
        image = data["image"].astype(np.float32)
        gt = data["mask"].astype(bool)
        full_shape = image.shape[1:]

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        image_low = F.interpolate(image_t, size=input_shape, mode="trilinear",
                                  align_corners=False).to(device)
        with torch.no_grad():
            output = stage1_model(image_low)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        stage1_probs[sid] = (prob, image, full_shape)
        gts[sid] = gt

    # Sweep Stage 1 thresholds
    print(f"\n{'='*80}")
    print("STAGE 1 THRESHOLD SWEEP")
    print(f"{'='*80}")
    print(f"{'S1_Thresh':>10} {'N_ROIs':>8} {'Detected':>10} {'Missed':>8} "
          f"{'Dice_mean':>10} {'Dice_med':>10} {'Sens_mean':>10}")
    print("-" * 80)

    best_dice = 0
    best_s1_thresh = 0.2

    for s1_thresh in s1_thresholds:
        dices = []
        senss = []
        total_rois = 0
        detected = 0
        missed = 0

        for sid in tqdm(subject_ids, desc=f"S1={s1_thresh:.2f}", leave=False):
            prob, image, full_shape = stage1_probs[sid]
            gt = gts[sid]
            scale = [f / l for f, l in zip(full_shape, input_shape)]

            # Threshold Stage 1
            binary = (prob > s1_thresh).astype(np.uint8)

            if binary.sum() == 0:
                pred_final = np.zeros(full_shape, dtype=np.uint8)
                if gt.sum() > 0:
                    missed += 1
            else:
                labeled, n = ndimage.label(binary)
                rois = []
                for i in range(1, n + 1):
                    coords = np.where(labeled == i)
                    bbox_min = [max(0, int(c.min() * s - (c.max() - c.min()) * s * margin_expand))
                                for c, s in zip(coords, scale)]
                    bbox_max = [min(fs, int((c.max() + 1) * s + (c.max() - c.min()) * s * margin_expand))
                                for c, s, fs in zip(coords, scale, full_shape)]
                    rois.append({"bbox_min": bbox_min, "bbox_max": bbox_max})

                total_rois += len(rois)

                # Run Stage 2
                roi_preds = []
                for roi in rois:
                    pred = run_stage2_ensemble(stage2_models, image, roi, crop_size, device)
                    roi_preds.append(pred)

                stitched = stitch_predictions(full_shape, rois, roi_preds)
                pred_final = postprocess((stitched > s2_threshold).astype(np.uint8), min_size=10)

                if gt.sum() > 0 and pred_final.sum() > 0:
                    detected += 1
                elif gt.sum() > 0:
                    missed += 1

            pred_bool = pred_final.astype(bool)
            dices.append(compute_dice(pred_bool, gt))
            senss.append(compute_sensitivity(pred_bool, gt))

        mean_dice = np.mean(dices)
        med_dice = np.median(dices)
        mean_sens = np.mean(senss)
        avg_rois = total_rois / len(subject_ids)

        marker = ""
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_s1_thresh = s1_thresh
            marker = " ← BEST"

        print(f"{s1_thresh:>10.2f} {avg_rois:>8.1f} {detected:>10d} {missed:>8d} "
              f"{mean_dice:>10.4f} {med_dice:>10.4f} {mean_sens:>10.4f}{marker}")

    print(f"\n{'='*80}")
    print(f"BEST Stage 1 threshold: {best_s1_thresh}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"{'='*80}")

    # Save
    reports_dir = Path(config["paths"]["reports"])
    result = {
        "best_s1_threshold": best_s1_thresh,
        "best_dice": round(best_dice, 4),
        "s2_threshold": s2_threshold,
    }
    with open(reports_dir / "best_stage1_threshold.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {reports_dir / 'best_stage1_threshold.json'}")


if __name__ == "__main__":
    main()
