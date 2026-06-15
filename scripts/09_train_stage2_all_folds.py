"""
Train All Stage 2 Folds (Phase 3 convenience wrapper)
======================================================
Trains both SegResNet and Swin-UNETR across all 5 folds sequentially.

Usage:
  python scripts/09_train_stage2_all_folds.py --config configs/soop_config.yaml
  python scripts/09_train_stage2_all_folds.py --config configs/soop_config.yaml --model segresnet
  python scripts/09_train_stage2_all_folds.py --config configs/soop_config.yaml --model swin_unetr --folds 0 1 2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run_training(script: str, config: str, fold: int, extra_args: list = None):
    """Run a training script for one fold."""
    cmd = [sys.executable, script, "--config", config, "--fold", str(fold)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STARTING: {Path(script).name} fold={fold}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n{status}: {Path(script).name} fold={fold} ({hours}h {mins}m)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train all Stage 2 folds")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--model", default="all", choices=["segresnet", "swin_unetr", "all"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    folds = args.folds or config["training"]["stage2"]["folds"]
    scripts_dir = Path(__file__).parent

    models = []
    if args.model in ("segresnet", "all"):
        models.append(("07_train_stage2_segresnet.py", "SegResNet"))
    if args.model in ("swin_unetr", "all"):
        models.append(("08_train_stage2_swin_unetr.py", "Swin-UNETR"))

    extra = []
    if args.epochs:
        extra.extend(["--epochs", str(args.epochs)])

    total_start = time.time()
    results = []

    for script_name, model_name in models:
        for fold in folds:
            rc = run_training(
                str(scripts_dir / script_name),
                args.config, fold, extra
            )
            results.append((model_name, fold, rc))

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    mins = int((total_elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETE ({hours}h {mins}m)")
    print(f"{'='*60}")
    for model_name, fold, rc in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"  [{status}] {model_name} fold {fold}")
    print(f"{'='*60}")

    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f"\n{len(failed)} training(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
