"""
Prepare Stage 2 Cropped Training Data (Phase 3)
=================================================
Uses ROI bounding boxes from Stage 1 to extract cropped patches from
full-resolution preprocessed data. Implements adaptive multi-scale patching
based on lesion volume.

For each ROI:
  - Crop the full-res image and mask around the ROI
  - Resize to a common training size (128^3)
  - Include negative patches (no lesion) at configured ratio

Outputs:
  - stage2_crops/<subject_id>_roi<N>.npz  (cropped patches)
  - stage2_crops/manifest.json            (index of all crops)

Usage:
  python scripts/06_prepare_stage2_crops.py --config configs/soop_config.yaml
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
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_adaptive_crop_size(lesion_volume_voxels: int, volume_bins: dict,
                           patch_sizes: dict, spacing: list) -> tuple:
    """Determine crop size based on lesion volume."""
    voxel_vol = float(np.prod(spacing))
    volume_ml = lesion_volume_voxels * voxel_vol / 1000.0

    if volume_ml < volume_bins["tiny"][1]:
        return tuple(patch_sizes["tiny"])
    elif volume_ml < volume_bins["small"][1]:
        return tuple(patch_sizes["small"])
    elif volume_ml < volume_bins["medium"][1]:
        return tuple(patch_sizes["medium"])
    else:
        return tuple(patch_sizes["large"])


def crop_volume(data: np.ndarray, bbox_min: list, bbox_max: list) -> np.ndarray:
    """Crop a 3D or 4D array using bounding box coordinates."""
    if data.ndim == 3:
        return data[bbox_min[0]:bbox_max[0],
                    bbox_min[1]:bbox_max[1],
                    bbox_min[2]:bbox_max[2]]
    elif data.ndim == 4:
        return data[:, bbox_min[0]:bbox_max[0],
                       bbox_min[1]:bbox_max[1],
                       bbox_min[2]:bbox_max[2]]
    raise ValueError(f"Unexpected ndim: {data.ndim}")


def resize_volume(data: np.ndarray, target_size: tuple,
                  mode: str = "trilinear") -> np.ndarray:
    """Resize 3D or 4D volume to target size."""
    if data.ndim == 3:
        t = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        interp_mode = "nearest" if mode == "nearest" else "trilinear"
        resized = F.interpolate(t, size=target_size, mode=interp_mode)
        return resized.squeeze().numpy()
    elif data.ndim == 4:
        t = torch.from_numpy(data).float().unsqueeze(0)
        resized = F.interpolate(t, size=target_size, mode="trilinear",
                                align_corners=False)
        return resized.squeeze(0).numpy()
    raise ValueError(f"Unexpected ndim: {data.ndim}")


def sample_negative_crop(image: np.ndarray, mask: np.ndarray,
                         crop_size: tuple, max_attempts: int = 50) -> dict:
    """Sample a random crop that contains no lesion."""
    shape = image.shape[1:]  # (D, H, W)

    for _ in range(max_attempts):
        center = [np.random.randint(crop_size[i] // 2, shape[i] - crop_size[i] // 2)
                  for i in range(3)]
        bbox_min = [center[i] - crop_size[i] // 2 for i in range(3)]
        bbox_max = [bbox_min[i] + crop_size[i] for i in range(3)]

        # Ensure within bounds
        bbox_min = [max(0, b) for b in bbox_min]
        bbox_max = [min(s, b) for s, b in zip(shape, bbox_max)]

        mask_crop = crop_volume(mask, bbox_min, bbox_max)
        if mask_crop.sum() == 0:
            image_crop = crop_volume(image, bbox_min, bbox_max)
            return {
                "image": image_crop,
                "mask": mask_crop,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "has_lesion": False,
            }

    return None


def process_subject(sid: str, preproc_dir: Path, roi_dir: Path,
                    output_dir: Path, config: dict,
                    common_size: tuple = (128, 128, 128)) -> list:
    """Generate all crops for one subject."""
    npz_path = preproc_dir / f"{sid}.npz"
    roi_path = roi_dir / f"{sid}.json"

    if not npz_path.exists() or not roi_path.exists():
        return []

    data = np.load(npz_path)
    image = data["image"].astype(np.float32)  # (3, D, H, W)
    mask = data["mask"].astype(np.uint8)       # (D, H, W)
    target_spacing = data.get("target_spacing", np.array([1.0, 1.0, 1.0]))

    with open(roi_path) as f:
        roi_data = json.load(f)

    volume_bins = config["preprocessing"]["volume_bins"]
    patch_sizes = config["training"]["stage2"]["patch_sizes"]
    neg_ratio = config["training"]["stage2"]["negative_patch_ratio"]

    crops_info = []
    crop_idx = 0

    # Process each ROI
    for roi in roi_data.get("rois", []):
        bbox_min = roi["bbox_min"]
        bbox_max = roi["bbox_max"]

        image_crop = crop_volume(image, bbox_min, bbox_max)
        mask_crop = crop_volume(mask, bbox_min, bbox_max)

        # Resize to common size
        image_resized = resize_volume(image_crop, common_size, mode="trilinear")
        mask_resized = resize_volume(mask_crop, common_size, mode="nearest")
        mask_resized = (mask_resized > 0.5).astype(np.uint8)

        # Chronic mask if available
        save_dict = {
            "image": image_resized.astype(np.float32),
            "mask": mask_resized,
            "original_bbox_min": np.array(bbox_min),
            "original_bbox_max": np.array(bbox_max),
            "has_lesion": mask_crop.sum() > 0,
        }

        if "mask_chronic" in data:
            chronic_crop = crop_volume(data["mask_chronic"], bbox_min, bbox_max)
            chronic_resized = resize_volume(chronic_crop, common_size, mode="nearest")
            save_dict["mask_chronic"] = (chronic_resized > 0.5).astype(np.uint8)

        crop_name = f"{sid}_roi{crop_idx:02d}"
        np.savez_compressed(output_dir / f"{crop_name}.npz", **save_dict)

        crops_info.append({
            "crop_name": crop_name,
            "subject_id": sid,
            "roi_idx": crop_idx,
            "has_lesion": bool(mask_crop.sum() > 0),
            "lesion_voxels_in_crop": int(mask_crop.sum()),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        })
        crop_idx += 1

    # Add negative crops
    n_positive = len([c for c in crops_info if c["has_lesion"]])
    n_negative_needed = max(1, int(n_positive * neg_ratio / (1 - neg_ratio)))

    for neg_i in range(n_negative_needed):
        neg_crop = sample_negative_crop(image, mask, common_size)
        if neg_crop is not None:
            crop_name = f"{sid}_neg{neg_i:02d}"
            np.savez_compressed(output_dir / f"{crop_name}.npz", **{
                "image": resize_volume(neg_crop["image"], common_size, "trilinear").astype(np.float32),
                "mask": np.zeros(common_size, dtype=np.uint8),
                "has_lesion": False,
            })
            crops_info.append({
                "crop_name": crop_name,
                "subject_id": sid,
                "roi_idx": -1,
                "has_lesion": False,
                "lesion_voxels_in_crop": 0,
            })

    return crops_info


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 2 cropped data")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    roi_dir = Path(config["paths"]["preprocessed"]).parent / "rois"
    splits_dir = Path(config["paths"]["splits"])
    output_dir = Path(config["paths"]["preprocessed"]).parent / "stage2_crops" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    print(f"Preparing Stage 2 crops for {args.split} ({len(subject_ids)} subjects)")
    print(f"Output: {output_dir}")

    all_crops = []
    for sid in tqdm(subject_ids, desc="Cropping"):
        crops = process_subject(sid, preproc_dir, roi_dir, output_dir, config)
        all_crops.extend(crops)

    # Save manifest
    manifest = {
        "split": args.split,
        "total_crops": len(all_crops),
        "positive_crops": sum(1 for c in all_crops if c["has_lesion"]),
        "negative_crops": sum(1 for c in all_crops if not c["has_lesion"]),
        "crops": all_crops,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"STAGE 2 CROPS ({args.split})")
    print(f"{'=' * 50}")
    print(f"  Total crops:    {manifest['total_crops']}")
    print(f"  Positive crops: {manifest['positive_crops']}")
    print(f"  Negative crops: {manifest['negative_crops']}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
