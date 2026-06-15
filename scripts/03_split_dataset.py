"""
Stratified Dataset Split (Phase 1)
====================================
Splits preprocessed SOOP subjects into train/val/test with stratification by:
  - Lesion volume bin (tiny/small/medium/large/none)
  - Presence of chronic lesion
  - Negative controls distributed evenly

Uses the audit report for stratification metadata.

Outputs:
  - splits/train.json, val.json, test.json  (lists of subject IDs)
  - splits/split_summary.json               (distribution stats)

Usage:
  python scripts/03_split_dataset.py --config configs/soop_config.yaml
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit


def load_audit(reports_dir: Path) -> pd.DataFrame:
    """Load audit report CSV."""
    audit_csv = reports_dir / "audit_report.csv"
    if not audit_csv.exists():
        print(f"ERROR: Audit report not found: {audit_csv}")
        print("  Run 00_audit_soop.py first.")
        sys.exit(1)
    return pd.read_csv(audit_csv)


def build_stratification_key(row: pd.Series) -> str:
    """Create a composite stratification key from subject metadata."""
    parts = []

    # Volume bin
    vol_bin = row.get("volume_bin", "none")
    if pd.isna(vol_bin):
        vol_bin = "none"
    parts.append(f"vol_{vol_bin}")

    # Chronic lesion presence
    has_chronic = row.get("chronic_empty")
    if pd.isna(has_chronic):
        parts.append("chr_unk")
    elif has_chronic:
        parts.append("chr_no")
    else:
        parts.append("chr_yes")

    return "__".join(parts)


def split_dataset(config: dict):
    """Perform stratified train/val/test split."""
    reports_dir = Path(config["paths"]["reports"])
    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    preproc_cfg = config["preprocessing"]
    train_ratio = preproc_cfg["train_ratio"]
    val_ratio = preproc_cfg["val_ratio"]
    test_ratio = preproc_cfg["test_ratio"]

    # Load audit data
    df = load_audit(reports_dir)

    # Filter to only subjects that have been preprocessed
    preprocessed_ids = set(
        p.stem for p in preproc_dir.glob("sub-*.npz")
    )
    df = df[df["subject_id"].isin(preprocessed_ids)].copy()
    print(f"Subjects with preprocessed data: {len(df)}")

    if len(df) == 0:
        print("ERROR: No preprocessed subjects found. Run 01_preprocess_soop.py first.")
        sys.exit(1)

    # Build stratification keys
    df["strat_key"] = df.apply(build_stratification_key, axis=1)

    # Collapse rare strata (< 5 subjects) into an "other" bin
    strat_counts = df["strat_key"].value_counts()
    rare_keys = strat_counts[strat_counts < 5].index.tolist()
    df.loc[df["strat_key"].isin(rare_keys), "strat_key"] = "other"

    print(f"\nStratification groups:")
    for key, count in df["strat_key"].value_counts().items():
        print(f"  {key:30s}: {count}")

    # First split: train+val vs test
    test_fraction = test_ratio
    val_fraction_of_trainval = val_ratio / (train_ratio + val_ratio)

    subject_ids = df["subject_id"].values
    strat_keys = df["strat_key"].values

    # Split 1: (train+val) vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=42)
    trainval_idx, test_idx = next(sss1.split(subject_ids, strat_keys))

    trainval_ids = subject_ids[trainval_idx]
    trainval_strat = strat_keys[trainval_idx]
    test_ids = subject_ids[test_idx]

    # Split 2: train vs val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_trainval, random_state=42)
    train_idx, val_idx = next(sss2.split(trainval_ids, trainval_strat))

    train_ids = trainval_ids[train_idx].tolist()
    val_ids = trainval_ids[val_idx].tolist()
    test_ids = test_ids.tolist()

    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/val overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/test overlap!"

    # Save splits
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_path = splits_dir / f"{name}.json"
        with open(split_path, "w") as f:
            json.dump(sorted(ids), f, indent=2)

    # Summary stats
    def split_stats(ids, df):
        subset = df[df["subject_id"].isin(ids)]
        return {
            "count": len(ids),
            "volume_bins": subset["volume_bin"].value_counts().to_dict(),
            "has_acute_mask": int(subset["has_mask_acute"].sum()) if "has_mask_acute" in subset else 0,
            "strat_distribution": subset["strat_key"].value_counts().to_dict(),
        }

    summary = {
        "total_subjects": len(df),
        "train": split_stats(train_ids, df),
        "val": split_stats(val_ids, df),
        "test": split_stats(test_ids, df),
        "ratios": {
            "train": round(len(train_ids) / len(df), 3),
            "val": round(len(val_ids) / len(df), 3),
            "test": round(len(test_ids) / len(df), 3),
        },
    }

    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'=' * 50}")
    print("DATASET SPLIT")
    print(f"{'=' * 50}")
    print(f"  Train: {len(train_ids):5d} ({summary['ratios']['train']:.1%})")
    print(f"  Val:   {len(val_ids):5d} ({summary['ratios']['val']:.1%})")
    print(f"  Test:  {len(test_ids):5d} ({summary['ratios']['test']:.1%})")

    print(f"\n  Volume bin distribution:")
    for split_name in ["train", "val", "test"]:
        bins = summary[split_name]["volume_bins"]
        print(f"    {split_name:6s}: {bins}")

    print(f"\n  Saved to: {splits_dir}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Split SOOP dataset")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    split_dataset(config)


if __name__ == "__main__":
    main()
