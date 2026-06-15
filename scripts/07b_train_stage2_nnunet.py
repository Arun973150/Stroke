"""
Stage 2: nnU-Net Training on Cropped ROIs (Phase 3)
=====================================================
Wraps nnU-Net v2's native pipeline: plan_and_preprocess → train → predict.
Runs all 5 folds sequentially (or a specific fold).

Prerequisites:
  - Run 06b_convert_crops_to_nnunet.py first
  - Set nnU-Net environment variables (or this script sets them from config)

Usage:
  python scripts/07b_train_stage2_nnunet.py --config configs/soop_config.yaml
  python scripts/07b_train_stage2_nnunet.py --config configs/soop_config.yaml --fold 0
  python scripts/07b_train_stage2_nnunet.py --config configs/soop_config.yaml --skip-planning
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def set_nnunet_env(config: dict):
    """Set nnU-Net environment variables from config."""
    os.environ["nnUNet_raw"] = config["paths"]["nnunet_raw"]
    os.environ["nnUNet_preprocessed"] = config["paths"]["nnunet_preprocessed"]
    os.environ["nnUNet_results"] = config["paths"]["nnunet_results"]

    # Create dirs
    for key in ["nnunet_raw", "nnunet_preprocessed", "nnunet_results"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)

    print("nnU-Net environment:")
    print(f"  nnUNet_raw:          {os.environ['nnUNet_raw']}")
    print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"  nnUNet_results:      {os.environ['nnUNet_results']}")


def run_command(cmd: list, description: str) -> int:
    """Run a shell command with live output."""
    print(f"\n{'─' * 60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'─' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    status = "SUCCESS" if result.returncode == 0 else "FAILED"

    print(f"\n  [{status}] {description} ({hours}h {mins}m)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net on Stage 2 crops")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--dataset-id", default="002")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train specific fold (default: all)")
    parser.add_argument("--skip-planning", action="store_true",
                        help="Skip plan_and_preprocess (already done)")
    parser.add_argument("--trainer", default="nnUNetTrainer",
                        help="nnU-Net trainer class")
    parser.add_argument("--configuration", default="3d_fullres",
                        help="nnU-Net configuration")
    parser.add_argument("--continue-training", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_nnunet_env(config)

    dataset_id = args.dataset_id
    folds = [args.fold] if args.fold is not None else config["training"]["stage2"]["folds"]

    print(f"\nDataset ID:    {dataset_id}")
    print(f"Configuration: {args.configuration}")
    print(f"Trainer:       {args.trainer}")
    print(f"Folds:         {folds}")

    # ── Step 1: Plan and Preprocess ──────────────────────────────────────────
    if not args.skip_planning:
        rc = run_command(
            ["nnUNetv2_plan_and_preprocess",
             "-d", dataset_id,
             "--verify_dataset_integrity",
             "-c", args.configuration],
            f"Planning and preprocessing Dataset{dataset_id}"
        )
        if rc != 0:
            print("ERROR: Planning failed. Check dataset.json and file structure.")
            sys.exit(1)

    # ── Step 2: Train each fold ──────────────────────────────────────────────
    results = []
    for fold in folds:
        cmd = [
            "nnUNetv2_train",
            dataset_id,
            args.configuration,
            str(fold),
            "-tr", args.trainer,
        ]

        if args.continue_training:
            cmd.append("--c")

        rc = run_command(cmd, f"Training fold {fold}")
        results.append((fold, rc))

        if rc != 0:
            print(f"WARNING: Fold {fold} training failed!")

    # ── Step 3: Find best configuration ──────────────────────────────────────
    # Only if all folds completed
    all_success = all(rc == 0 for _, rc in results)

    if all_success and len(folds) == 5:
        run_command(
            ["nnUNetv2_find_best_configuration",
             dataset_id,
             "-c", args.configuration,
             "-tr", args.trainer],
            "Finding best configuration"
        )

    # ── Step 4: Run prediction on validation set ─────────────────────────────
    if all_success:
        nnunet_raw = Path(config["paths"]["nnunet_raw"])
        nnunet_results = Path(config["paths"]["nnunet_results"])
        dataset_name = None

        # Find the dataset directory name
        for d in nnunet_raw.iterdir():
            if d.is_dir() and d.name.startswith(f"Dataset{dataset_id}"):
                dataset_name = d.name
                break

        if dataset_name:
            images_ts = nnunet_raw / dataset_name / "imagesTs"
            pred_output = nnunet_results / dataset_name / "predictions_cascade"
            pred_output.mkdir(parents=True, exist_ok=True)

            if images_ts.exists() and any(images_ts.iterdir()):
                # Use all folds for ensemble prediction
                fold_str = " ".join(str(f) for f in folds)

                run_command(
                    ["nnUNetv2_predict",
                     "-i", str(images_ts),
                     "-o", str(pred_output),
                     "-d", dataset_id,
                     "-c", args.configuration,
                     "-tr", args.trainer,
                     "-f"] + [str(f) for f in folds] + [
                     "--save_probabilities"],
                    "Predicting on validation set"
                )
            else:
                print("  No imagesTs found, skipping prediction.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("NNUNET STAGE 2 TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for fold, rc in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"  [{status}] Fold {fold}")

    print(f"\n  Results: {config['paths']['nnunet_results']}")
    print(f"  To export predictions:")
    print(f"    python scripts/07c_export_nnunet_predictions.py --config configs/soop_config.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
