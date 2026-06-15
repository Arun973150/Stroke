"""
Convert Full-Brain Preprocessed Volumes to nnU-Net Dataset003 Format
=====================================================================
Unlike Dataset002 (cascade-cropped ROIs), this creates a FULL-BRAIN dataset
for standalone nnU-Net training (Path B safety net).

Each subject's preprocessed .npz (3-channel: TRACE, ADC, FLAIR + mask)
is converted to nnU-Net's NIfTI folder structure at native 1mm isotropic.

  nnUNet_raw/Dataset003_SOOPFullBrain/
  ├── dataset.json
  ├── imagesTr/
  │   ├── sub-0001_0000.nii.gz  (TRACE)
  │   ├── sub-0001_0001.nii.gz  (ADC)
  │   ├── sub-0001_0002.nii.gz  (FLAIR)
  │   └── ...
  ├── labelsTr/
  │   ├── sub-0001.nii.gz
  │   └── ...
  └── imagesTs/  (val/test subjects)

Usage:
  python scripts/14_convert_fullbrain_to_nnunet.py --config configs/soop_config.yaml
  python scripts/14_convert_fullbrain_to_nnunet.py --config configs/soop_config.yaml --include-val-as-test
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


def npz_to_nifti(data: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> nib.Nifti1Image:
    """Convert a 3D numpy array to a NIfTI image with given spacing."""
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return nib.Nifti1Image(data, affine)


def convert_subjects(subject_ids: list, preproc_dir: Path, images_dir: Path,
                     labels_dir: Path, spacing: tuple, desc: str = "Converting") -> list:
    """Convert a list of subjects from .npz to nnU-Net NIfTI format."""
    case_list = []
    skipped = 0

    for sid in tqdm(subject_ids, desc=desc):
        npz_path = preproc_dir / f"{sid}.npz"
        if not npz_path.exists():
            print(f"  WARNING: Missing {npz_path}")
            skipped += 1
            continue

        data = np.load(npz_path)
        image = data["image"]  # (3, D, H, W)
        mask = data["mask"]    # (D, H, W)

        # Use subject ID as case ID (e.g., sub-0001)
        case_id = sid

        # Save each channel as separate NIfTI
        for ch_idx in range(image.shape[0]):
            ch_data = image[ch_idx].astype(np.float32)
            nii = npz_to_nifti(ch_data, spacing)
            nib.save(nii, str(images_dir / f"{case_id}_{ch_idx:04d}.nii.gz"))

        # Save label — binarize (any non-zero → 1)
        mask_data = (mask > 0).astype(np.uint8)
        nii_mask = npz_to_nifti(mask_data, spacing)
        nib.save(nii_mask, str(labels_dir / f"{case_id}.nii.gz"))

        has_lesion = bool(mask.sum() > 0)
        lesion_vol_ml = float(mask.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0)

        case_list.append({
            "case_id": case_id,
            "has_lesion": has_lesion,
            "lesion_volume_ml": round(lesion_vol_ml, 3),
        })

    if skipped:
        print(f"  Skipped {skipped} missing subjects")

    return case_list


def create_dataset_json(output_dir: Path, num_training: int):
    """Create nnU-Net dataset.json for full-brain dataset."""
    dataset = {
        "channel_names": {
            "0": "TRACE",
            "1": "ADC",
            "2": "FLAIR",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "SOOPFullBrain",
        "description": "SOOP full-brain stroke lesion segmentation (standalone Path B)",
        "reference": "Sentinel Stroke v2",
        "licence": "see SOOP dataset license",
        "release": "1.0",
    }

    dataset_json_path = output_dir / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    return dataset_json_path


def main():
    parser = argparse.ArgumentParser(description="Convert full-brain volumes to nnU-Net Dataset003")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--dataset-id", default="003", help="nnU-Net dataset ID")
    parser.add_argument("--dataset-name", default="SOOPFullBrain", help="nnU-Net dataset name")
    parser.add_argument("--include-val-as-test", action="store_true",
                        help="Also convert val split to imagesTs for prediction")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    nnunet_raw = Path(config["paths"]["nnunet_raw"])
    spacing = tuple(config["preprocessing"]["target_spacing"])

    dataset_dir_name = f"Dataset{args.dataset_id}_{args.dataset_name}"
    output_dir = nnunet_raw / dataset_dir_name

    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"

    for d in [images_tr, labels_tr]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Converting full-brain volumes to nnU-Net format")
    print(f"  Preprocessed: {preproc_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Spacing:      {spacing}")

    # Load train split
    with open(splits_dir / "train.json") as f:
        train_ids = json.load(f)

    print(f"\n  Train subjects: {len(train_ids)}")
    train_cases = convert_subjects(train_ids, preproc_dir, images_tr, labels_tr, spacing, "Train")

    # Optionally convert val as imagesTs
    val_cases = []
    if args.include_val_as_test:
        images_ts = output_dir / "imagesTs"
        images_ts.mkdir(parents=True, exist_ok=True)

        # Also save val labels for evaluation
        labels_ts = output_dir / "labelsTs"
        labels_ts.mkdir(parents=True, exist_ok=True)

        with open(splits_dir / "val.json") as f:
            val_ids = json.load(f)

        print(f"\n  Val subjects (→ imagesTs): {len(val_ids)}")
        val_cases = convert_subjects(val_ids, preproc_dir, images_ts, labels_ts, spacing, "Val")

    # Create dataset.json
    ds_json = create_dataset_json(output_dir, len(train_cases))

    # Save case mapping
    mapping = {
        "dataset_dir": dataset_dir_name,
        "dataset_id": args.dataset_id,
        "spacing": list(spacing),
        "train_cases": train_cases,
        "val_cases": val_cases,
    }
    mapping_path = output_dir / "case_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n{'=' * 60}")
    print("FULL-BRAIN NNUNET DATASET CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Dataset:      {dataset_dir_name}")
    print(f"  Train cases:  {len(train_cases)}")
    print(f"  Val cases:    {len(val_cases)}")
    print(f"  dataset.json: {ds_json}")
    print(f"  Mapping:      {mapping_path}")
    print(f"\n  Next steps:")
    print(f"    1. nnUNetv2_plan_and_preprocess -d {args.dataset_id} -pl nnUNetPlannerResEncM --verify_dataset_integrity")
    print(f"    2. nnUNetv2_train {args.dataset_id} 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
