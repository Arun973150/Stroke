"""
Export nnU-Net Predictions Back to Crop Format (Phase 3)
=========================================================
Converts nnU-Net's NIfTI predictions back to numpy arrays that the cascade
inference pipeline can consume alongside SegResNet and Swin-UNETR predictions.

Maps nnU-Net case IDs back to original subject/ROI using case_mapping.json.

Usage:
  python scripts/07c_export_nnunet_predictions.py --config configs/soop_config.yaml
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


def main():
    parser = argparse.ArgumentParser(description="Export nnU-Net predictions")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--dataset-id", default="002")
    parser.add_argument("--configuration", default="3d_fullres")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nnunet_raw = Path(config["paths"]["nnunet_raw"])
    nnunet_results = Path(config["paths"]["nnunet_results"])

    # Find dataset directory
    dataset_dir = None
    for d in nnunet_raw.iterdir():
        if d.is_dir() and d.name.startswith(f"Dataset{args.dataset_id}"):
            dataset_dir = d
            break

    if not dataset_dir:
        print(f"ERROR: Dataset{args.dataset_id} not found in {nnunet_raw}")
        sys.exit(1)

    # Load case mapping
    mapping_path = dataset_dir / "case_mapping.json"
    if not mapping_path.exists():
        print(f"ERROR: case_mapping.json not found: {mapping_path}")
        sys.exit(1)

    with open(mapping_path) as f:
        mapping = json.load(f)

    # Find nnU-Net predictions
    dataset_name = dataset_dir.name
    pred_dir = nnunet_results / dataset_name / "predictions_cascade"

    if not pred_dir.exists():
        # Try trainer-specific path
        trainer_dirs = list((nnunet_results / dataset_name).glob(f"nnUNetTrainer__{args.configuration}*"))
        if trainer_dirs:
            # Look for crossval predictions
            for td in trainer_dirs:
                cv_pred = td / "crossval_results_folds_0_1_2_3_4"
                if cv_pred.exists():
                    pred_dir = cv_pred
                    break
                # Or fold-specific validation
                for fold_dir in td.glob("fold_*"):
                    val_dir = fold_dir / "validation"
                    if val_dir.exists():
                        pred_dir = val_dir
                        break

    if not pred_dir.exists():
        print(f"ERROR: No predictions found. Expected: {pred_dir}")
        print("  Run 07b_train_stage2_nnunet.py first.")
        sys.exit(1)

    print(f"Dataset: {dataset_name}")
    print(f"Predictions: {pred_dir}")

    # Output directory for exported predictions
    export_dir = Path(config["paths"]["checkpoints"]) / "stage2_nnunet" / "predictions"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Build case_id → crop_name mapping
    all_cases = mapping.get("train_cases", []) + mapping.get("val_cases", [])
    case_to_crop = {c["case_id"]: c for c in all_cases}

    # Export predictions
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    exported = 0
    not_found = 0

    for pred_file in tqdm(pred_files, desc="Exporting"):
        case_id = pred_file.name.replace(".nii.gz", "")

        if case_id not in case_to_crop:
            not_found += 1
            continue

        crop_info = case_to_crop[case_id]

        # Load prediction
        pred_nii = nib.load(str(pred_file))
        pred_data = pred_nii.get_fdata()

        # Also check for probability file (.npz from nnU-Net)
        prob_file = pred_dir / f"{case_id}.npz"
        prob_data = None
        if prob_file.exists():
            prob_npz = np.load(prob_file)
            # nnU-Net saves softmax probabilities as (n_classes, D, H, W)
            if "probabilities" in prob_npz:
                prob_data = prob_npz["probabilities"]
            elif len(prob_npz.files) > 0:
                # Usually the first array is probabilities
                prob_data = prob_npz[prob_npz.files[0]]

        # Save in our format
        save_dict = {
            "prediction": pred_data.astype(np.uint8),
            "case_id": case_id,
            "crop_name": crop_info["original_crop"],
            "subject_id": crop_info["subject_id"],
        }
        if prob_data is not None:
            # Take the lesion class probability (index 1 for binary)
            if prob_data.ndim == 4 and prob_data.shape[0] >= 2:
                save_dict["probability"] = prob_data[1].astype(np.float32)
            elif prob_data.ndim == 3:
                save_dict["probability"] = prob_data.astype(np.float32)

        output_path = export_dir / f"{crop_info['original_crop']}_nnunet.npz"
        np.savez_compressed(output_path, **save_dict)
        exported += 1

    # Save export manifest
    manifest = {
        "source": str(pred_dir),
        "exported": exported,
        "not_found": not_found,
        "export_dir": str(export_dir),
    }
    with open(export_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 50}")
    print("NNUNET EXPORT COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Exported:  {exported}")
    print(f"  Not found: {not_found}")
    print(f"  Output:    {export_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
