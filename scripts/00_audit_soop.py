"""
SOOP Dataset Audit (Phase 0)
=============================
Scans every subject in the SOOP BIDS directory and produces a comprehensive
report: which modalities exist, voxel spacings, image dims, lesion volumes,
ADC value ranges, and flags problems.

Outputs:
  - audit_report.csv   (one row per subject)
  - audit_summary.json (aggregate statistics)
  - flagged_subjects.csv (subjects with issues)

Usage:
  python scripts/00_audit_soop.py --config configs/soop_config.yaml
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── BIDS helpers ──────────────────────────────────────────────────────────────

def find_file(subject_dir: Path, pattern: str):
    """Return first match or None."""
    matches = list(subject_dir.rglob(pattern))
    return matches[0] if matches else None


def discover_subject_files(subject_dir: Path) -> dict:
    """Find TRACE, ADC, FLAIR and mask files for one subject."""
    sid = subject_dir.name
    return {
        "subject_id": sid,
        "trace_nii": find_file(subject_dir, f"{sid}_rec-TRACE_dwi.nii.gz"),
        "trace_json": find_file(subject_dir, f"{sid}_rec-TRACE_dwi.json"),
        "adc_nii": find_file(subject_dir, f"{sid}_rec-ADC_dwi.nii.gz"),
        "adc_json": find_file(subject_dir, f"{sid}_rec-ADC_dwi.json"),
        "flair_nii": find_file(subject_dir, f"{sid}_FLAIR.nii.gz"),
        "flair_json": find_file(subject_dir, f"{sid}_FLAIR.json"),
    }


def discover_masks(derivatives_dir: Path, sid: str) -> dict:
    """Find acute, chronic, and combined masks in derivatives."""
    mask_dir = derivatives_dir / "lesion_masks" / sid / "dwi"
    if not mask_dir.exists():
        # Try alternative layouts
        mask_dir = derivatives_dir / "lesion_masks" / sid
        if not mask_dir.exists():
            mask_dir = derivatives_dir / sid / "dwi"
    return {
        "mask_acute": find_file(mask_dir, "*desc-lesionAcute_mask.nii.gz") if mask_dir.exists() else None,
        "mask_chronic": find_file(mask_dir, "*desc-lesionChronic_mask.nii.gz") if mask_dir.exists() else None,
        "mask_combined": find_file(mask_dir, "*desc-lesion_mask.nii.gz") if mask_dir.exists() else None,
    }


# ── Volume / spacing helpers ─────────────────────────────────────────────────

def get_nii_info(nii_path: Path) -> dict:
    """Load header only to get spacing, shape, dtype."""
    try:
        img = nib.load(str(nii_path))
        hdr = img.header
        return {
            "shape": list(img.shape[:3]),
            "spacing": [round(float(s), 4) for s in hdr.get_zooms()[:3]],
            "dtype": str(hdr.get_data_dtype()),
        }
    except Exception as e:
        return {"shape": None, "spacing": None, "dtype": None, "error": str(e)}


def compute_lesion_stats(mask_path: Path, spacing: list) -> dict:
    """Compute volume (ml) and voxel count from a binary mask."""
    try:
        data = nib.load(str(mask_path)).get_fdata()
        binary = (data > 0.5).astype(np.uint8)
        voxel_count = int(binary.sum())
        voxel_vol_mm3 = float(np.prod(spacing))
        volume_ml = voxel_count * voxel_vol_mm3 / 1000.0
        return {
            "voxel_count": voxel_count,
            "volume_ml": round(volume_ml, 4),
            "is_empty": voxel_count == 0,
        }
    except Exception as e:
        return {"voxel_count": None, "volume_ml": None, "is_empty": None, "error": str(e)}


def read_json_sidecar(json_path: Path) -> dict:
    """Read acquisition params from a BIDS JSON sidecar."""
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_adc_range(nii_path: Path, brain_thresh: float = 0.0) -> dict:
    """Get min/max/mean ADC within a rough brain mask."""
    try:
        data = nib.load(str(nii_path)).get_fdata()
        brain = data[data > brain_thresh]
        if len(brain) == 0:
            return {"adc_min": None, "adc_max": None, "adc_mean": None}
        return {
            "adc_min": round(float(brain.min()), 2),
            "adc_max": round(float(brain.max()), 2),
            "adc_mean": round(float(brain.mean()), 2),
        }
    except Exception as e:
        return {"adc_min": None, "adc_max": None, "adc_mean": None, "error": str(e)}


# ── Per-subject audit ─────────────────────────────────────────────────────────

def audit_subject(subject_dir: Path, derivatives_dir: Path) -> dict:
    """Run full audit on a single subject."""
    files = discover_subject_files(subject_dir)
    masks = discover_masks(derivatives_dir, files["subject_id"])

    row = {"subject_id": files["subject_id"]}

    # Modality presence
    row["has_trace"] = files["trace_nii"] is not None
    row["has_adc"] = files["adc_nii"] is not None
    row["has_flair"] = files["flair_nii"] is not None
    row["has_mask_acute"] = masks["mask_acute"] is not None
    row["has_mask_chronic"] = masks["mask_chronic"] is not None

    # TRACE info
    if files["trace_nii"]:
        info = get_nii_info(files["trace_nii"])
        row["trace_shape"] = str(info["shape"])
        row["trace_spacing"] = str(info["spacing"])
        row["trace_spacing_x"] = info["spacing"][0] if info["spacing"] else None
        row["trace_spacing_y"] = info["spacing"][1] if info["spacing"] else None
        row["trace_spacing_z"] = info["spacing"][2] if info["spacing"] else None

    # ADC info
    if files["adc_nii"]:
        info = get_nii_info(files["adc_nii"])
        row["adc_shape"] = str(info["shape"])
        row["adc_spacing"] = str(info["spacing"])
        adc_range = get_adc_range(files["adc_nii"])
        row.update(adc_range)

    # ADC JSON sidecar
    if files["adc_json"]:
        sidecar = read_json_sidecar(files["adc_json"])
        row["adc_bvalue"] = sidecar.get("DiffusionBValue")

    # FLAIR info
    if files["flair_nii"]:
        info = get_nii_info(files["flair_nii"])
        row["flair_shape"] = str(info["shape"])
        row["flair_spacing"] = str(info["spacing"])

    # Lesion mask stats (use TRACE spacing for volume calc)
    trace_spacing = [row.get("trace_spacing_x", 1), row.get("trace_spacing_y", 1), row.get("trace_spacing_z", 1)]
    if masks["mask_acute"]:
        stats = compute_lesion_stats(masks["mask_acute"], trace_spacing)
        row["acute_volume_ml"] = stats.get("volume_ml")
        row["acute_voxels"] = stats.get("voxel_count")
        row["acute_empty"] = stats.get("is_empty")

    if masks["mask_chronic"]:
        stats = compute_lesion_stats(masks["mask_chronic"], trace_spacing)
        row["chronic_volume_ml"] = stats.get("volume_ml")
        row["chronic_empty"] = stats.get("is_empty")

    # Volume bin
    vol = row.get("acute_volume_ml")
    if vol is not None and vol > 0:
        if vol < 1:
            row["volume_bin"] = "tiny"
        elif vol < 5:
            row["volume_bin"] = "small"
        elif vol < 50:
            row["volume_bin"] = "medium"
        else:
            row["volume_bin"] = "large"
    else:
        row["volume_bin"] = "none"

    # Flags
    flags = []
    if not row["has_trace"]:
        flags.append("MISSING_TRACE")
    if not row["has_adc"]:
        flags.append("MISSING_ADC")
    if not row["has_flair"]:
        flags.append("MISSING_FLAIR")
    if not row.get("has_mask_acute"):
        flags.append("NO_ACUTE_MASK")
    if row.get("acute_empty"):
        flags.append("EMPTY_ACUTE_MASK")
    z_spacing = row.get("trace_spacing_z")
    if z_spacing and z_spacing > 6.0:
        flags.append("THICK_SLICES")
    adc_max = row.get("adc_max")
    if adc_max and adc_max > 10000:
        flags.append("ADC_UNUSUAL_RANGE")
    row["flags"] = "|".join(flags) if flags else ""

    return row


# ── Summary stats ────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics from audit dataframe."""
    summary = {
        "total_subjects": len(df),
        "has_all_modalities": int((df["has_trace"] & df["has_adc"] & df["has_flair"]).sum()),
        "missing_trace": int((~df["has_trace"]).sum()),
        "missing_adc": int((~df["has_adc"]).sum()),
        "missing_flair": int((~df["has_flair"]).sum()),
        "has_acute_mask": int(df["has_mask_acute"].sum()),
        "has_chronic_mask": int(df["has_mask_chronic"].sum()),
        "empty_acute_masks": int(df["acute_empty"].sum()) if "acute_empty" in df else 0,
    }

    # Volume stats
    vols = df["acute_volume_ml"].dropna()
    vols_nonzero = vols[vols > 0]
    if len(vols_nonzero) > 0:
        summary["lesion_volume_stats"] = {
            "count": int(len(vols_nonzero)),
            "mean_ml": round(float(vols_nonzero.mean()), 3),
            "median_ml": round(float(vols_nonzero.median()), 3),
            "min_ml": round(float(vols_nonzero.min()), 4),
            "max_ml": round(float(vols_nonzero.max()), 2),
            "std_ml": round(float(vols_nonzero.std()), 3),
        }

    # Volume bin distribution
    summary["volume_bin_distribution"] = df["volume_bin"].value_counts().to_dict()

    # Spacing stats (from TRACE)
    for axis, col in [("x", "trace_spacing_x"), ("y", "trace_spacing_y"), ("z", "trace_spacing_z")]:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[f"spacing_{axis}"] = {
                "median": round(float(vals.median()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "unique_count": int(vals.nunique()),
            }

    # ADC range stats
    adc_means = df["adc_mean"].dropna()
    if len(adc_means) > 0:
        summary["adc_mean_stats"] = {
            "min": round(float(adc_means.min()), 2),
            "max": round(float(adc_means.max()), 2),
            "median": round(float(adc_means.median()), 2),
        }

    # Flagged subjects count
    flagged = df[df["flags"] != ""]
    summary["flagged_subjects"] = len(flagged)
    summary["flag_counts"] = defaultdict(int)
    for flags_str in flagged["flags"]:
        for f in flags_str.split("|"):
            summary["flag_counts"][f] += 1
    summary["flag_counts"] = dict(summary["flag_counts"])

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit SOOP dataset")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (for quick test)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw_data"])
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        sys.exit(1)

    # Find derivatives dir
    derivatives_dir = Path(config["paths"].get("derivatives", str(raw_dir / "derivatives")))
    if not derivatives_dir.exists():
        print(f"WARNING: Derivatives directory not found: {derivatives_dir}")
        print("  Mask audit will be skipped.")

    # Discover all subjects
    subject_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    if args.max_subjects:
        subject_dirs = subject_dirs[:args.max_subjects]

    print(f"Found {len(subject_dirs)} subjects in {raw_dir}")
    print(f"Reports will be saved to {reports_dir}\n")

    # Audit each subject
    rows = []
    for sd in tqdm(subject_dirs, desc="Auditing subjects"):
        row = audit_subject(sd, derivatives_dir)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save full audit
    audit_csv = reports_dir / "audit_report.csv"
    df.to_csv(audit_csv, index=False)
    print(f"\nFull audit saved: {audit_csv}")

    # Save flagged subjects
    flagged = df[df["flags"] != ""]
    flagged_csv = reports_dir / "flagged_subjects.csv"
    flagged.to_csv(flagged_csv, index=False)
    print(f"Flagged subjects ({len(flagged)}): {flagged_csv}")

    # Compute and save summary
    summary = compute_summary(df)
    summary_json = reports_dir / "audit_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Total subjects:        {summary['total_subjects']}")
    print(f"  All 3 modalities:      {summary['has_all_modalities']}")
    print(f"  Missing TRACE:         {summary['missing_trace']}")
    print(f"  Missing ADC:           {summary['missing_adc']}")
    print(f"  Missing FLAIR:         {summary['missing_flair']}")
    print(f"  Has acute mask:        {summary['has_acute_mask']}")
    print(f"  Has chronic mask:      {summary['has_chronic_mask']}")
    print(f"  Empty acute masks:     {summary['empty_acute_masks']}")
    print(f"  Flagged subjects:      {summary['flagged_subjects']}")

    if "lesion_volume_stats" in summary:
        vs = summary["lesion_volume_stats"]
        print(f"\n  Lesion volumes (ml):")
        print(f"    Count:   {vs['count']}")
        print(f"    Mean:    {vs['mean_ml']}")
        print(f"    Median:  {vs['median_ml']}")
        print(f"    Min:     {vs['min_ml']}")
        print(f"    Max:     {vs['max_ml']}")

    print(f"\n  Volume bins:")
    for bin_name, count in sorted(summary["volume_bin_distribution"].items()):
        print(f"    {bin_name:8s}: {count}")

    if summary["flag_counts"]:
        print(f"\n  Flags:")
        for flag, count in sorted(summary["flag_counts"].items()):
            print(f"    {flag:25s}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    main()
