"""
Convert Stage 2 Crops to nnU-Net Dataset Format (Phase 3)
==========================================================
Takes the cropped .npz patches from Stage 1 ROIs and converts them into
nnU-Net's required folder structure:

  nnUNet_raw/Dataset002_SOOPCascade/
  ├── dataset.json
  ├── imagesTr/
  │   ├── case_0001_0000.nii.gz  (TRACE)
  │   ├── case_0001_0001.nii.gz  (ADC)
  │   ├── case_0001_0002.nii.gz  (FLAIR)
  │   └── ...
  ├── labelsTr/
  │   ├── case_0001.nii.gz
  │   └── ...
  └── imagesTs/  (validation crops)

Usage:
  python scripts/06b_convert_crops_to_nnunet.py --config configs/soop_config.yaml
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


def convert_split(crop_dir: Path, manifest: dict, images_dir: Path,
                  labels_dir: Path, spacing: tuple,
                  start_idx: int = 0) -> tuple:
    """
    Convert crops from one split into nnU-Net NIfTI format.
    Returns (case_list, next_idx).
    """
    case_list = []

    for i, crop_info in enumerate(tqdm(manifest["crops"], desc=f"Converting {crop_dir.name}")):
        crop_name = crop_info["crop_name"]
        npz_path = crop_dir / f"{crop_name}.npz"

        if not npz_path.exists():
            print(f"  WARNING: Missing {npz_path}")
            continue

        data = np.load(npz_path)
        image = data["image"]  # (3, D, H, W)
        mask = data["mask"]    # (D, H, W)

        case_id = f"case_{start_idx + i:05d}"

        # Save each channel as separate NIfTI (nnU-Net convention)
        channel_names = ["TRACE", "ADC", "FLAIR"]
        for ch_idx in range(3):
            ch_data = image[ch_idx].astype(np.float32)
            nii = npz_to_nifti(ch_data, spacing)
            nib.save(nii, str(images_dir / f"{case_id}_{ch_idx:04d}.nii.gz"))

        # Save label
        mask_data = mask.astype(np.uint8)
        nii_mask = npz_to_nifti(mask_data, spacing)
        nib.save(nii_mask, str(labels_dir / f"{case_id}.nii.gz"))

        case_list.append({
            "case_id": case_id,
            "original_crop": crop_name,
            "subject_id": crop_info["subject_id"],
            "has_lesion": crop_info["has_lesion"],
        })

    return case_list, start_idx + len(manifest["crops"])


def create_dataset_json(output_dir: Path, train_cases: list, val_cases: list,
                        dataset_name: str, dataset_id: str):
    """Create nnU-Net dataset.json."""
    # nnU-Net v2 dataset.json format
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
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "SOOP cascade Stage 2 crops for nnU-Net training",
        "reference": "Sentinel Stroke v2",
        "licence": "see SOOP dataset license",
        "release": "1.0",
    }

    dataset_json_path = output_dir / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset_json_path


def main():
    parser = argparse.ArgumentParser(description="Convert Stage 2 crops to nnU-Net format")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--dataset-id", default="002", help="nnU-Net dataset ID")
    parser.add_argument("--dataset-name", default="SOOPCascade", help="nnU-Net dataset name")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    crop_base = Path(config["paths"]["preprocessed"]).parent / "stage2_crops"
    nnunet_raw = Path(config["paths"]["nnunet_raw"])
    spacing = tuple(config["preprocessing"]["target_spacing"])

    dataset_dir_name = f"Dataset{args.dataset_id}_{args.dataset_name}"
    output_dir = nnunet_raw / dataset_dir_name

    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_ts = output_dir / "imagesTs"

    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Converting crops to nnU-Net format")
    print(f"  Source:  {crop_base}")
    print(f"  Output:  {output_dir}")
    print(f"  Spacing: {spacing}")

    # Convert training crops
    train_manifest_path = crop_base / "train" / "manifest.json"
    if not train_manifest_path.exists():
        print(f"ERROR: Train manifest not found: {train_manifest_path}")
        print("  Run 06_prepare_stage2_crops.py --split train first.")
        sys.exit(1)

    with open(train_manifest_path) as f:
        train_manifest = json.load(f)

    train_cases, next_idx = convert_split(
        crop_base / "train", train_manifest,
        images_tr, labels_tr, spacing, start_idx=0
    )

    # Convert validation crops as test set for nnU-Net
    # (nnU-Net uses its own internal splits for CV, but we provide val as imagesTs
    #  so we can run nnUNetv2_predict on them)
    val_manifest_path = crop_base / "val" / "manifest.json"
    val_cases = []
    if val_manifest_path.exists():
        with open(val_manifest_path) as f:
            val_manifest = json.load(f)

        # Val goes to imagesTs (no labels needed for inference, but we save them
        # separately for evaluation)
        val_labels_dir = output_dir / "labelsTs"
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        val_cases, _ = convert_split(
            crop_base / "val", val_manifest,
            images_ts, val_labels_dir, spacing, start_idx=next_idx
        )

    # Create dataset.json
    ds_json = create_dataset_json(
        output_dir, train_cases, val_cases,
        args.dataset_name, args.dataset_id
    )

    # Save case mapping (for later: map nnU-Net case IDs back to subject/ROI)
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
    print("NNUNET DATASET CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Dataset:      {dataset_dir_name}")
    print(f"  Train cases:  {len(train_cases)}")
    print(f"  Val cases:    {len(val_cases)}")
    print(f"  dataset.json: {ds_json}")
    print(f"  Mapping:      {mapping_path}")
    print(f"\n  Next steps:")
    print(f"    1. export nnUNet_raw={nnunet_raw}")
    print(f"    2. export nnUNet_preprocessed={config['paths']['nnunet_preprocessed']}")
    print(f"    3. export nnUNet_results={config['paths']['nnunet_results']}")
    print(f"    4. nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"    5. python scripts/07b_train_stage2_nnunet.py --config configs/soop_config.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
