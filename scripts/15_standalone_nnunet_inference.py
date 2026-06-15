"""
Standalone nnU-Net Full-Brain Inference (Path B Safety Net)
============================================================
Runs nnU-Net trained on full-brain volumes (Dataset003) directly on
preprocessed subjects — NO cascade, NO Stage 1, NO ROI cropping.

This is the "safety net" path: if the cascade (Path A) misses a lesion
because Stage 1 failed to detect it, Path B can still catch it.

Outputs per-subject predictions + probability maps in the same format
as the cascade pipeline for direct comparison.

Usage:
  python scripts/15_standalone_nnunet_inference.py --config configs/soop_config.yaml --split val
  python scripts/15_standalone_nnunet_inference.py --config configs/soop_config.yaml --split test --folds 0 1 2 3 4
"""

import argparse
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")


def find_nnunet_checkpoints(nnunet_results: Path, dataset_id: str,
                             plans: str, folds: list) -> list:
    """Find trained nnU-Net checkpoint folders for given folds."""
    # Look for Dataset003_SOOPFullBrain directory
    dataset_dirs = [d for d in nnunet_results.iterdir()
                    if d.is_dir() and d.name.startswith(f"Dataset{dataset_id}")]

    if not dataset_dirs:
        raise FileNotFoundError(f"No Dataset{dataset_id} found in {nnunet_results}")

    dataset_dir = dataset_dirs[0]

    # Find trainer directory matching plans
    trainer_dirs = list(dataset_dir.glob(f"nnUNetTrainer__nnUNet*Plans*"))
    if not trainer_dirs:
        # Try with specific plans name
        trainer_dirs = list(dataset_dir.glob(f"nnUNetTrainer__{plans}__*"))

    if not trainer_dirs:
        raise FileNotFoundError(f"No trainer dirs found in {dataset_dir}")

    trainer_dir = trainer_dirs[0]

    fold_dirs = []
    for fold in folds:
        fold_dir = trainer_dir / f"fold_{fold}"
        ckpt = fold_dir / "checkpoint_final.pth"
        if not ckpt.exists():
            ckpt = fold_dir / "checkpoint_best.pth"
        if ckpt.exists():
            fold_dirs.append({"fold": fold, "checkpoint": ckpt, "fold_dir": fold_dir})
        else:
            print(f"  WARNING: No checkpoint for fold {fold} at {fold_dir}")

    return fold_dirs, trainer_dir


def run_nnunet_inference_on_subject(subject_id: str, preproc_dir: Path,
                                     nnunet_results: Path, dataset_id: str,
                                     output_dir: Path, plans: str = "nnUNetResEncUNetMPlans",
                                     folds: list = None):
    """
    Use nnU-Net's built-in predictor for a single subject.
    This is a simplified wrapper — for production, use nnUNetv2_predict CLI.
    """
    # This function is a placeholder — actual inference uses nnUNetv2_predict CLI
    pass


def postprocess(pred: np.ndarray, threshold: float = 0.35,
                min_component_size: int = 5) -> np.ndarray:
    """Threshold, remove small components, morphological closing."""
    binary = (pred > threshold).astype(np.uint8)

    labeled, n = ndimage.label(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_component_size:
            binary[labeled == i] = 0

    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3, 3)), iterations=1)
    return binary.astype(np.uint8)


def collect_nnunet_predictions(pred_dir: Path, subject_ids: list,
                                preproc_dir: Path, output_dir: Path,
                                threshold: float = 0.35,
                                min_component_size: int = 5):
    """
    Collect nnU-Net predictions (from nnUNetv2_predict output) and convert
    to our standard format (numpy arrays with postprocessing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for sid in tqdm(subject_ids, desc="Processing predictions"):
        # nnU-Net prediction NIfTI
        pred_nii_path = pred_dir / f"{sid}.nii.gz"

        if not pred_nii_path.exists():
            print(f"  WARNING: No prediction for {sid}")
            results.append({
                "subject_id": sid,
                "status": "missing",
                "pred_voxels": 0,
            })
            continue

        pred_nii = nib.load(str(pred_nii_path))
        pred_data = pred_nii.get_fdata()

        # Load probability map if available (.npz from --npz flag)
        prob_path = pred_dir / f"{sid}.npz"
        prob_data = None
        if prob_path.exists():
            prob_npz = np.load(prob_path)
            key = prob_npz.files[0] if prob_npz.files else None
            if key is not None:
                raw_prob = prob_npz[key]
                # nnU-Net saves softmax: (n_classes, D, H, W)
                if raw_prob.ndim == 4 and raw_prob.shape[0] >= 2:
                    prob_data = raw_prob[1].astype(np.float32)  # lesion class
                elif raw_prob.ndim == 3:
                    prob_data = raw_prob.astype(np.float32)

        # If we have probabilities, postprocess from them (better than binary)
        if prob_data is not None:
            final = postprocess(prob_data, threshold=threshold,
                               min_component_size=min_component_size)
            np.save(output_dir / f"{sid}_prob.npy", prob_data)
        else:
            # Use binary prediction directly
            final = pred_data.astype(np.uint8)

        np.save(output_dir / f"{sid}_pred.npy", final)

        # Load GT for stats
        gt_path = preproc_dir / f"{sid}.npz"
        has_gt_lesion = False
        if gt_path.exists():
            gt_data = np.load(gt_path)
            has_gt_lesion = bool(gt_data["mask"].sum() > 0)

        results.append({
            "subject_id": sid,
            "status": "ok",
            "pred_voxels": int(final.sum()),
            "has_gt_lesion": has_gt_lesion,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone nnU-Net inference (Path B)")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--dataset-id", default="003")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-component-size", type=int, default=5)
    parser.add_argument("--nnunet-pred-dir", default=None,
                        help="Directory with nnUNetv2_predict output (skip inference, just postprocess)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    nnunet_results = Path(config["paths"]["nnunet_results"])

    with open(splits_dir / f"{args.split}.json") as f:
        subject_ids = json.load(f)

    print(f"Standalone nnU-Net Inference (Path B)")
    print(f"  Split: {args.split} ({len(subject_ids)} subjects)")
    print(f"  Dataset: {args.dataset_id}")
    print(f"  Plans: {args.plans}")
    print(f"  Folds: {args.folds}")
    print(f"  Threshold: {args.threshold}, Min size: {args.min_component_size}")

    # Output directory
    output_dir = Path(config["paths"]["checkpoints"]).parent / "predictions_pathB" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.nnunet_pred_dir:
        # User already ran nnUNetv2_predict — just postprocess
        pred_dir = Path(args.nnunet_pred_dir)
        print(f"\n  Using existing predictions from: {pred_dir}")
    else:
        # Run nnUNetv2_predict via CLI
        folds_str = " ".join(str(f) for f in args.folds)
        pred_dir = output_dir / "raw_nnunet"
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Find the dataset directory name
        dataset_dirs = [d.name for d in nnunet_results.iterdir()
                        if d.is_dir() and d.name.startswith(f"Dataset{args.dataset_id}")]
        if not dataset_dirs:
            print(f"ERROR: No Dataset{args.dataset_id} in {nnunet_results}")
            print("  Run training first.")
            sys.exit(1)

        dataset_name = dataset_dirs[0]

        # Build input directory (imagesTs or create temp symlinks)
        nnunet_raw = Path(config["paths"]["nnunet_raw"])
        input_dir = nnunet_raw / dataset_name / "imagesTs"

        if not input_dir.exists() or not any(input_dir.glob("*.nii.gz")):
            print(f"\n  imagesTs not found. Creating from preprocessed {args.split} data...")
            input_dir.mkdir(parents=True, exist_ok=True)
            spacing = tuple(config["preprocessing"]["target_spacing"])

            for sid in tqdm(subject_ids, desc="Preparing input"):
                npz_path = preproc_dir / f"{sid}.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                image = data["image"]  # (3, D, H, W)
                for ch_idx in range(image.shape[0]):
                    ch_data = image[ch_idx].astype(np.float32)
                    affine = np.eye(4)
                    affine[0, 0] = spacing[0]
                    affine[1, 1] = spacing[1]
                    affine[2, 2] = spacing[2]
                    nii = nib.Nifti1Image(ch_data, affine)
                    nib.save(nii, str(input_dir / f"{sid}_{ch_idx:04d}.nii.gz"))

        # Print the command for the user to run
        print(f"\n{'=' * 60}")
        print("RUN THIS COMMAND FOR INFERENCE:")
        print(f"{'=' * 60}")
        print(f"nnUNetv2_predict \\")
        print(f"  -i {input_dir} \\")
        print(f"  -o {pred_dir} \\")
        print(f"  -d {args.dataset_id} \\")
        print(f"  -p {args.plans} \\")
        print(f"  -c 3d_fullres \\")
        print(f"  -f {folds_str} \\")
        print(f"  --save_probabilities")
        print(f"\nThen re-run this script with:")
        print(f"  python scripts/15_standalone_nnunet_inference.py --config configs/soop_config.yaml \\")
        print(f"    --split {args.split} --nnunet-pred-dir {pred_dir}")
        print(f"{'=' * 60}")
        return

    # Postprocess predictions
    print(f"\nPostprocessing predictions...")
    results = collect_nnunet_predictions(
        pred_dir, subject_ids, preproc_dir, output_dir,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
    )

    # Save log
    log_path = output_dir / "inference_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    detected = sum(1 for r in ok if r["pred_voxels"] > 0 and r.get("has_gt_lesion", False))
    total_pos = sum(1 for r in ok if r.get("has_gt_lesion", False))

    print(f"\n{'=' * 60}")
    print("STANDALONE NNUNET INFERENCE COMPLETE (PATH B)")
    print(f"{'=' * 60}")
    print(f"  Subjects processed: {len(ok)}")
    print(f"  Missing predictions: {len(results) - len(ok)}")
    print(f"  Detection rate: {detected}/{total_pos} ({detected/max(total_pos,1):.1%})")
    print(f"  Predictions: {output_dir}")
    print(f"\n  Next: run scripts/16_compare_pathA_pathB.py to pick best per subject")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
