"""
Generate ROI Bounding Boxes from Stage 1 (Phase 2)
====================================================
Runs the trained detection model on all subjects and extracts bounding boxes
around detected lesion regions for Stage 2 cropping.

For training data: uses both detected ROIs AND ground-truth ROIs (safety net).
For val/test data: uses detected ROIs only.

Outputs:
  - rois/<subject_id>.json  (list of ROI bounding boxes per subject)
  - rois/roi_summary.json   (detection stats)

Usage:
  python scripts/05_generate_stage1_rois.py --config configs/soop_config.yaml
"""

import argparse
import json
import sys
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


def load_detection_model(ckpt_path: Path, device: torch.device) -> SegResNet:
    """Load trained Stage 1 detection model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=3,
        out_channels=1,
        dropout_prob=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_rois(prob_map: np.ndarray, threshold: float = 0.2,
                 margin_expand: float = 0.5,
                 full_res_shape: tuple = (192, 192, 192),
                 low_res_shape: tuple = (96, 96, 96)) -> list:
    """
    Extract ROI bounding boxes from a low-res probability map.
    Scales boxes back to full resolution coordinates.
    """
    binary = (prob_map > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # Connected components
    labeled, n_components = ndimage.label(binary)

    scale_factors = [f / l for f, l in zip(full_res_shape, low_res_shape)]
    rois = []

    for comp_id in range(1, n_components + 1):
        component = (labeled == comp_id)
        coords = np.where(component)

        # Bounding box in low-res space
        bbox_min = [int(c.min()) for c in coords]
        bbox_max = [int(c.max()) + 1 for c in coords]

        # Scale to full resolution
        bbox_min_full = [int(b * s) for b, s in zip(bbox_min, scale_factors)]
        bbox_max_full = [int(b * s) for b, s in zip(bbox_max, scale_factors)]

        # Expand by margin
        sizes = [mx - mn for mn, mx in zip(bbox_min_full, bbox_max_full)]
        for i in range(3):
            expand = int(sizes[i] * margin_expand)
            bbox_min_full[i] = max(0, bbox_min_full[i] - expand)
            bbox_max_full[i] = min(full_res_shape[i], bbox_max_full[i] + expand)

        # Component stats
        prob_in_comp = prob_map[component]

        rois.append({
            "bbox_min": bbox_min_full,
            "bbox_max": bbox_max_full,
            "size": [mx - mn for mn, mx in zip(bbox_min_full, bbox_max_full)],
            "voxel_count_lowres": int(component.sum()),
            "mean_prob": round(float(prob_in_comp.mean()), 4),
            "max_prob": round(float(prob_in_comp.max()), 4),
        })

    # Merge overlapping ROIs
    rois = merge_overlapping_rois(rois, full_res_shape)
    return rois


def merge_overlapping_rois(rois: list, full_shape: tuple) -> list:
    """Merge ROIs whose bounding boxes overlap."""
    if len(rois) <= 1:
        return rois

    def boxes_overlap(a, b):
        for i in range(3):
            if a["bbox_max"][i] <= b["bbox_min"][i] or b["bbox_max"][i] <= a["bbox_min"][i]:
                return False
        return True

    merged = True
    while merged:
        merged = False
        new_rois = []
        used = set()
        for i in range(len(rois)):
            if i in used:
                continue
            current = rois[i].copy()
            for j in range(i + 1, len(rois)):
                if j in used:
                    continue
                if boxes_overlap(current, rois[j]):
                    # Merge
                    current["bbox_min"] = [
                        min(current["bbox_min"][k], rois[j]["bbox_min"][k]) for k in range(3)
                    ]
                    current["bbox_max"] = [
                        max(current["bbox_max"][k], rois[j]["bbox_max"][k]) for k in range(3)
                    ]
                    current["size"] = [
                        current["bbox_max"][k] - current["bbox_min"][k] for k in range(3)
                    ]
                    current["mean_prob"] = max(current["mean_prob"], rois[j]["mean_prob"])
                    current["max_prob"] = max(current["max_prob"], rois[j]["max_prob"])
                    used.add(j)
                    merged = True
            new_rois.append(current)
            used.add(i)
        rois = new_rois

    return rois


def gt_rois_from_mask(mask: np.ndarray, margin_expand: float = 0.5) -> list:
    """Extract ROI bounding boxes from ground truth mask."""
    if mask.sum() == 0:
        return []

    labeled, n_components = ndimage.label(mask > 0)
    rois = []

    for comp_id in range(1, n_components + 1):
        coords = np.where(labeled == comp_id)
        bbox_min = [int(c.min()) for c in coords]
        bbox_max = [int(c.max()) + 1 for c in coords]

        sizes = [mx - mn for mn, mx in zip(bbox_min, bbox_max)]
        for i in range(3):
            expand = int(sizes[i] * margin_expand)
            bbox_min[i] = max(0, bbox_min[i] - expand)
            bbox_max[i] = min(mask.shape[i], bbox_max[i] + expand)

        rois.append({
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "size": [mx - mn for mn, mx in zip(bbox_min, bbox_max)],
            "source": "ground_truth",
        })

    return rois


def process_subject(sid: str, model, preproc_dir: Path, device: torch.device,
                    input_shape: tuple, full_shape: tuple,
                    threshold: float, margin_expand: float,
                    include_gt: bool = False) -> dict:
    """Run detection on one subject and extract ROIs."""
    npz_path = preproc_dir / f"{sid}.npz"
    data = np.load(npz_path)
    image = data["image"].astype(np.float32)  # (3, D, H, W)
    mask = data["mask"]

    # Downsample
    image_t = torch.from_numpy(image).unsqueeze(0)
    image_low = F.interpolate(image_t, size=input_shape, mode="trilinear",
                              align_corners=False).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_low)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

    # Extract detected ROIs
    det_rois = extract_rois(prob_map, threshold, margin_expand,
                            full_shape, input_shape)

    # For training: add ground truth ROIs for any missed lesions
    gt_rois = []
    if include_gt and mask.sum() > 0:
        gt_rois = gt_rois_from_mask(mask, margin_expand)

    # Merge detected + GT (deduplicate)
    all_rois = det_rois.copy()
    if include_gt:
        for gt_roi in gt_rois:
            # Check if already covered by a detected ROI
            covered = False
            gt_center = [(gt_roi["bbox_min"][i] + gt_roi["bbox_max"][i]) // 2 for i in range(3)]
            for det_roi in det_rois:
                inside = all(det_roi["bbox_min"][i] <= gt_center[i] <= det_roi["bbox_max"][i] for i in range(3))
                if inside:
                    covered = True
                    break
            if not covered:
                gt_roi["source"] = "ground_truth_missed"
                all_rois.append(gt_roi)

    has_lesion = float(mask.sum() > 0)
    detected = len(det_rois) > 0

    return {
        "subject_id": sid,
        "has_lesion": has_lesion,
        "detected": detected,
        "n_detected_rois": len(det_rois),
        "n_gt_added": len(all_rois) - len(det_rois),
        "rois": all_rois,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 ROIs")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_cfg = config["training"]["stage1"]
    input_shape = tuple(stage1_cfg["input_shape"])
    target_shape = tuple(config["preprocessing"]["target_shape"])
    threshold = stage1_cfg["detection_threshold"]
    margin_expand = config["training"]["stage2"]["roi_margin_expand"]

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        Path(config["paths"]["checkpoints"]) / "stage1_detection" / "checkpoint_best.pth"
    )
    roi_dir = Path(config["paths"]["preprocessed"]).parent / "rois"
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {ckpt_path}")
    model = load_detection_model(ckpt_path, device)

    # Load all splits
    all_stats = {"train": {}, "val": {}, "test": {}}

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.json"
        with open(split_path) as f:
            subject_ids = json.load(f)

        include_gt = (split_name == "train")  # only add GT safety net for training

        print(f"\nProcessing {split_name} ({len(subject_ids)} subjects, include_gt={include_gt})")

        total_positive = 0
        total_detected = 0
        all_subject_rois = {}

        for sid in tqdm(subject_ids, desc=split_name):
            result = process_subject(
                sid, model, preproc_dir, device,
                input_shape, target_shape,
                threshold, margin_expand, include_gt
            )

            # Save per-subject ROI file
            roi_path = roi_dir / f"{sid}.json"
            with open(roi_path, "w") as f:
                json.dump(result, f, indent=2)

            if result["has_lesion"]:
                total_positive += 1
                if result["detected"]:
                    total_detected += 1

            all_subject_rois[sid] = {
                "n_rois": len(result["rois"]),
                "detected": result["detected"],
                "has_lesion": result["has_lesion"],
            }

        recall = total_detected / max(total_positive, 1)
        all_stats[split_name] = {
            "total": len(subject_ids),
            "positive_cases": total_positive,
            "detected_cases": total_detected,
            "recall": round(recall, 4),
        }
        print(f"  Detection recall: {recall:.3f} ({total_detected}/{total_positive})")

    # Save summary
    summary_path = roi_dir / "roi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'=' * 50}")
    print("ROI GENERATION COMPLETE")
    print(f"{'=' * 50}")
    for split_name, stats in all_stats.items():
        print(f"  {split_name:6s}: recall={stats['recall']:.3f} "
              f"({stats['detected_cases']}/{stats['positive_cases']} cases)")
    print(f"  ROIs saved to: {roi_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
