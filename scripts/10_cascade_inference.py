"""
Full Cascade Inference Pipeline (Phase 4)
==========================================
End-to-end inference: Stage 1 detection → ROI crop → Stage 2 ensemble → stitch.

Runs all trained models and produces final segmentation predictions.
Supports multiple ensemble strategies: average, majority vote, learned fusion.

Usage:
  python scripts/10_cascade_inference.py --config configs/soop_config.yaml --split test
  python scripts/10_cascade_inference.py --config configs/soop_config.yaml --split val --ensemble average
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.networks.nets import SegResNet, SwinUNETR
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Model Loading ────────────────────────────────────────────────────────────

def load_stage1_model(ckpt_path: Path, device: torch.device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=16, in_channels=3, out_channels=1, dropout_prob=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_segresnet_fold(ckpt_path: Path, device: torch.device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=32, in_channels=3, out_channels=1, dropout_prob=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_swin_unetr_fold(ckpt_path: Path, device: torch.device):
    model = SwinUNETR(
        img_size=(128, 128, 128), in_channels=3, out_channels=1,
        feature_size=48, use_v2=True,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_all_models(config: dict, device: torch.device) -> dict:
    """Load Stage 1 + all Stage 2 fold models (SegResNet, Swin-UNETR, nnU-Net)."""
    ckpt_dir = Path(config["paths"]["checkpoints"])
    folds = config["training"]["stage2"]["folds"]

    models = {
        "stage1": load_stage1_model(ckpt_dir / "stage1_detection" / "checkpoint_best.pth", device),
        "segresnet": [],
        "swin_unetr": [],
        "nnunet_pred_dir": None,  # nnU-Net uses pre-exported predictions, not live inference
    }

    for fold in folds:
        sr_path = ckpt_dir / "stage2_segresnet" / f"fold_{fold}" / "checkpoint_best.pth"
        if sr_path.exists():
            models["segresnet"].append(load_segresnet_fold(sr_path, device))

        sw_path = ckpt_dir / "stage2_swin_unetr" / f"fold_{fold}" / "checkpoint_best.pth"
        if sw_path.exists():
            models["swin_unetr"].append(load_swin_unetr_fold(sw_path, device))

    # nnU-Net: check for exported predictions directory
    nnunet_pred_dir = ckpt_dir / "stage2_nnunet" / "predictions"
    if nnunet_pred_dir.exists() and any(nnunet_pred_dir.glob("*.npz")):
        models["nnunet_pred_dir"] = nnunet_pred_dir

    n_nn = len(list(nnunet_pred_dir.glob("*.npz"))) if models["nnunet_pred_dir"] else 0
    print(f"Loaded: Stage1 + {len(models['segresnet'])} SegResNet + "
          f"{len(models['swin_unetr'])} Swin-UNETR + nnU-Net ({n_nn} predictions)")
    return models


# ── Stage 1: Detection ──────────────────────────────────────────────────────

def run_stage1(model, image: np.ndarray, input_shape: tuple,
               full_shape: tuple, threshold: float,
               margin_expand: float, device: torch.device) -> list:
    """Run Stage 1 detection and return ROI bounding boxes."""
    image_t = torch.from_numpy(image).float().unsqueeze(0)
    image_low = F.interpolate(image_t, size=input_shape, mode="trilinear",
                              align_corners=False).to(device)

    with torch.no_grad():
        output = model(image_low)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

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

    return rois


# ── Stage 2: Segmentation ───────────────────────────────────────────────────

def load_nnunet_prediction(nnunet_pred_dir: Path, subject_id: str,
                           roi_idx: int, crop_size: tuple) -> torch.Tensor:
    """Load pre-exported nnU-Net prediction for a specific ROI crop."""
    # Try matching by subject_id and roi_idx
    pattern = f"{subject_id}_roi{roi_idx:02d}_nnunet.npz"
    pred_path = nnunet_pred_dir / pattern

    if not pred_path.exists():
        return None

    data = np.load(pred_path)
    if "probability" in data:
        prob = data["probability"]  # (D, H, W)
    elif "prediction" in data:
        prob = data["prediction"].astype(np.float32)
    else:
        return None

    # Resize to crop_size if needed
    if prob.shape != tuple(crop_size):
        prob_t = torch.from_numpy(prob).float().unsqueeze(0).unsqueeze(0)
        prob_t = F.interpolate(prob_t, size=crop_size, mode="trilinear", align_corners=False)
        return prob_t.squeeze(0)  # (1, D, H, W)

    return torch.from_numpy(prob).float().unsqueeze(0)  # (1, D, H, W)


def _sliding_window_positions(volume_shape, window_size, overlap=0.5):
    """Generate sliding window start positions for a 3D volume."""
    positions = []
    step = [int(w * (1 - overlap)) for w in window_size]

    for d in range(0, max(volume_shape[0] - window_size[0] + 1, 1), step[0]):
        for h in range(0, max(volume_shape[1] - window_size[1] + 1, 1), step[1]):
            for w in range(0, max(volume_shape[2] - window_size[2] + 1, 1), step[2]):
                positions.append((d, h, w))

    # Ensure last position covers the end
    for dim in range(3):
        if volume_shape[dim] > window_size[dim]:
            last_pos = volume_shape[dim] - window_size[dim]
            # Check if we already have a position near the end
            covered = any(p[dim] >= last_pos - step[dim] // 2 for p in positions)
            if not covered:
                # Add end-aligned positions
                for p in list(positions):
                    new_p = list(p)
                    new_p[dim] = last_pos
                    new_p = tuple(new_p)
                    if new_p not in positions:
                        positions.append(new_p)

    return positions


def _run_model_on_window(model, window_input, device, tta_flips):
    """Run a single model with TTA on a window."""
    tta_preds = []
    for flip_dims in tta_flips:
        augmented = window_input
        for d in flip_dims:
            augmented = torch.flip(augmented, [d])
        with torch.no_grad():
            out = torch.sigmoid(model(augmented))
        for d in reversed(flip_dims):
            out = torch.flip(out, [d])
        tta_preds.append(out.cpu())
    return torch.stack(tta_preds).mean(dim=0)


def _gaussian_window_weight(shape, sigma_fraction=0.3):
    """Create a 3D Gaussian weight for a sliding window."""
    grids = [np.linspace(-1, 1, s) for s in shape]
    z, y, x = np.meshgrid(*grids, indexing="ij")
    w = np.exp(-(z**2 + y**2 + x**2) / (2 * sigma_fraction**2))
    return w.astype(np.float32)


def run_stage2_on_roi(models: dict, image: np.ndarray, roi: dict,
                      crop_size: tuple, device: torch.device,
                      ensemble: str = "average",
                      subject_id: str = "", roi_idx: int = 0) -> np.ndarray:
    """Run all Stage 2 models on a single ROI crop (SegResNet + Swin-UNETR + nnU-Net)."""
    bbox_min = roi["bbox_min"]
    bbox_max = roi["bbox_max"]

    # Crop
    crop = image[:, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
    roi_shape = crop.shape[1:]

    # Resize to model input
    crop_t = torch.from_numpy(crop).float().unsqueeze(0)
    crop_resized = F.interpolate(crop_t, size=crop_size, mode="trilinear",
                                 align_corners=False).to(device)

    # TTA flip axes
    tta_flips = [
        [],          # original
        [2],         # flip D
        [3],         # flip H
        [4],         # flip W
        [2, 3],      # flip D+H
        [2, 4],      # flip D+W
        [3, 4],      # flip H+W
        [2, 3, 4],   # flip D+H+W
    ]

    all_preds = []

    # SegResNet predictions with TTA
    for model in models["segresnet"]:
        all_preds.append(_run_model_on_window(model, crop_resized, device, tta_flips))

    # Swin-UNETR predictions with TTA
    for model in models["swin_unetr"]:
        all_preds.append(_run_model_on_window(model, crop_resized, device, tta_flips))

    # nnU-Net predictions (pre-exported)
    if models.get("nnunet_pred_dir"):
        nn_pred = load_nnunet_prediction(
            models["nnunet_pred_dir"], subject_id, roi_idx, crop_size
        )
        if nn_pred is not None:
            all_preds.append(nn_pred)

    if not all_preds:
        return np.zeros(roi_shape, dtype=np.float32)

    # Normalize all predictions to same shape (1, 1, D, H, W)
    normalized = []
    for p in all_preds:
        if p.dim() == 3:
            p = p.unsqueeze(0).unsqueeze(0)
        elif p.dim() == 4:
            p = p.unsqueeze(0)
        normalized.append(p)
    all_preds = normalized

    # Ensemble
    stacked = torch.stack(all_preds, dim=0)
    if ensemble == "average":
        combined = stacked.mean(dim=0)
    elif ensemble == "majority_vote":
        votes = (stacked > 0.5).float()
        combined = (votes.mean(dim=0) >= 0.5).float()
    else:
        combined = stacked.mean(dim=0)

    pred_crop = combined.squeeze().numpy()

    # Resize prediction back to original ROI size
    pred_t = torch.from_numpy(pred_crop).float().unsqueeze(0).unsqueeze(0)
    pred_full = F.interpolate(pred_t, size=list(roi_shape), mode="trilinear",
                              align_corners=False).squeeze().numpy()
    return pred_full


# ── Stitching ────────────────────────────────────────────────────────────────

def stitch_predictions(full_shape: tuple, rois: list,
                       roi_predictions: list) -> np.ndarray:
    """Stitch ROI predictions back into full-brain volume with Gaussian weighting."""
    output = np.zeros(full_shape, dtype=np.float32)
    weight = np.zeros(full_shape, dtype=np.float32)

    for roi, pred in zip(rois, roi_predictions):
        bmin = roi["bbox_min"]
        bmax = roi["bbox_max"]
        roi_size = [bmax[i] - bmin[i] for i in range(3)]

        # Gaussian weight (center has higher weight)
        gauss = _gaussian_weight(roi_size)

        output[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] += pred * gauss
        weight[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] += gauss

    # Normalize
    mask = weight > 0
    output[mask] /= weight[mask]
    return output


def _gaussian_weight(shape: list, sigma_fraction: float = 0.3) -> np.ndarray:
    """Create a 3D Gaussian weight volume."""
    grids = []
    for s in shape:
        g = np.linspace(-1, 1, s)
        grids.append(g)
    z, y, x = np.meshgrid(*grids, indexing="ij")
    sigma = sigma_fraction
    w = np.exp(-(z**2 + y**2 + x**2) / (2 * sigma**2))
    return w.astype(np.float32)


# ── Post-processing ─────────────────────────────────────────────────────────

def postprocess(pred: np.ndarray, threshold: float = 0.5,
                min_component_size: int = 10,
                max_prob_threshold: float = 0.45,
                min_volume_ml: float = 0.1,
                spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Threshold, remove small components, suppress false positives, morphological closing.

    False positive suppression:
      - If max probability in entire volume < max_prob_threshold → suppress all (low confidence)
      - If total predicted volume < min_volume_ml → suppress all (too small to be real)
    """
    # FP suppression: low confidence → no prediction
    if pred.max() < max_prob_threshold:
        return np.zeros_like(pred, dtype=np.uint8)

    binary = (pred > threshold).astype(np.uint8)

    # FP suppression: tiny volume → no prediction
    voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    total_vol_ml = binary.sum() * voxel_vol_ml
    if total_vol_ml < min_volume_ml:
        return np.zeros_like(pred, dtype=np.uint8)

    # Remove small connected components
    labeled, n = ndimage.label(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_component_size:
            binary[labeled == i] = 0

    # Morphological closing
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)), iterations=1)
    return binary.astype(np.uint8)


# ── Per-subject inference ────────────────────────────────────────────────────

def predict_subject(sid: str, models: dict, preproc_dir: Path,
                    config: dict, device: torch.device,
                    ensemble: str = "average") -> dict:
    """Full cascade inference on one subject."""
    npz_path = preproc_dir / f"{sid}.npz"
    data = np.load(npz_path)
    image = data["image"].astype(np.float32)
    mask_gt = data["mask"]
    full_shape = image.shape[1:]  # (D, H, W)

    stage1_cfg = config["training"]["stage1"]
    stage2_cfg = config["training"]["stage2"]

    # Stage 1: detect ROIs
    rois = run_stage1(
        models["stage1"], image,
        tuple(stage1_cfg["input_shape"]),
        full_shape,
        stage1_cfg["detection_threshold"],
        stage2_cfg["roi_margin_expand"],
        device
    )

    if not rois:
        return {
            "subject_id": sid,
            "prediction": np.zeros(full_shape, dtype=np.uint8),
            "n_rois": 0,
            "has_gt_lesion": bool(mask_gt.sum() > 0),
        }

    # Stage 2: segment each ROI (SegResNet + Swin-UNETR + nnU-Net ensemble)
    crop_size = (128, 128, 128)
    roi_preds = []
    for roi_idx, roi in enumerate(rois):
        pred = run_stage2_on_roi(models, image, roi, crop_size, device, ensemble,
                                 subject_id=sid, roi_idx=roi_idx)
        roi_preds.append(pred)

    # Stitch
    stitched = stitch_predictions(full_shape, rois, roi_preds)

    # Post-process with FP suppression (tuned via sweep script 17)
    spacing = tuple(config["preprocessing"]["target_spacing"])
    final = postprocess(stitched, threshold=0.35, min_component_size=5,
                        max_prob_threshold=0.40, min_volume_ml=0.20,
                        spacing=spacing)

    return {
        "subject_id": sid,
        "prediction": final,
        "probability": stitched,
        "n_rois": len(rois),
        "has_gt_lesion": bool(mask_gt.sum() > 0),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cascade inference pipeline")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ensemble", default="average", choices=["average", "majority_vote"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(config["paths"]["checkpoints"]).parent / "predictions" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    print(f"Split: {args.split} ({len(subject_ids)} subjects)")
    print(f"Ensemble: {args.ensemble}")
    print(f"Output: {output_dir}")

    # Load models
    models = load_all_models(config, device)

    # Run inference
    results = []
    for sid in tqdm(subject_ids, desc="Inference"):
        result = predict_subject(sid, models, preproc_dir, config, device, args.ensemble)

        # Save prediction
        pred_path = output_dir / f"{sid}_pred.npy"
        np.save(pred_path, result["prediction"])

        prob_path = output_dir / f"{sid}_prob.npy"
        if "probability" in result:
            np.save(prob_path, result["probability"])

        results.append({
            "subject_id": sid,
            "n_rois": result["n_rois"],
            "has_gt_lesion": result["has_gt_lesion"],
            "pred_voxels": int(result["prediction"].sum()),
        })

    # Save results log
    log_path = output_dir / "inference_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    detected = sum(1 for r in results if r["pred_voxels"] > 0 and r["has_gt_lesion"])
    total_pos = sum(1 for r in results if r["has_gt_lesion"])

    print(f"\n{'=' * 50}")
    print(f"INFERENCE COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Subjects: {len(results)}")
    print(f"  Detection rate: {detected}/{total_pos} ({detected/max(total_pos,1):.1%})")
    print(f"  Predictions: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
