"""
SOOP Quality Control (Phase 1)
===============================
Runs automated QC checks on preprocessed data and generates visual montages.

Checks per subject:
  1. TRACE-ADC consistency: lesion region should be TRACE-hyperintense, ADC-hypointense
  2. Brain mask coverage (no clipping)
  3. Lesion mask within brain
  4. Empty mask detection
  5. Intensity range sanity

Outputs:
  - qc_report.csv        (pass/fail per subject per check)
  - qc_summary.json      (aggregate)
  - montages/             (PNG montage per subject)
  - qc_failures.csv      (subjects that failed any check)

Usage:
  python scripts/02_quality_control.py --config configs/soop_config.yaml
  python scripts/02_quality_control.py --config configs/soop_config.yaml --no-montage
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── QC Checks ────────────────────────────────────────────────────────────────

def check_trace_adc_consistency(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    In acute lesion regions, TRACE should be hyperintense (above brain mean)
    and ADC should be hypointense (below brain mean).
    """
    trace = image[0]  # channel 0
    adc = image[1]    # channel 1

    brain = (np.abs(trace) > 0.01) | (np.abs(adc) > 0.01)
    lesion = mask > 0

    if lesion.sum() == 0:
        return {"check": "trace_adc_consistency", "status": "skip", "reason": "no_lesion"}

    brain_trace_mean = trace[brain].mean() if brain.sum() > 0 else 0
    brain_adc_mean = adc[brain].mean() if brain.sum() > 0 else 0

    lesion_trace_mean = trace[lesion].mean()
    lesion_adc_mean = adc[lesion].mean()

    trace_ok = lesion_trace_mean > brain_trace_mean
    adc_ok = lesion_adc_mean < brain_adc_mean

    status = "pass" if (trace_ok and adc_ok) else "fail"
    return {
        "check": "trace_adc_consistency",
        "status": status,
        "brain_trace_mean": round(float(brain_trace_mean), 4),
        "lesion_trace_mean": round(float(lesion_trace_mean), 4),
        "trace_ok": trace_ok,
        "brain_adc_mean": round(float(brain_adc_mean), 4),
        "lesion_adc_mean": round(float(lesion_adc_mean), 4),
        "adc_ok": adc_ok,
    }


def check_brain_coverage(image: np.ndarray) -> dict:
    """Check that brain isn't clipped at volume edges."""
    trace = image[0]
    brain = np.abs(trace) > 0.01

    if brain.sum() == 0:
        return {"check": "brain_coverage", "status": "fail", "reason": "no_brain_signal"}

    # Check if brain touches edges (any face)
    touches = []
    if brain[0, :, :].sum() > 50:
        touches.append("z_min")
    if brain[-1, :, :].sum() > 50:
        touches.append("z_max")
    if brain[:, 0, :].sum() > 50:
        touches.append("y_min")
    if brain[:, -1, :].sum() > 50:
        touches.append("y_max")
    if brain[:, :, 0].sum() > 50:
        touches.append("x_min")
    if brain[:, :, -1].sum() > 50:
        touches.append("x_max")

    status = "warn" if len(touches) > 0 else "pass"
    return {
        "check": "brain_coverage",
        "status": status,
        "touches_edges": touches,
    }


def check_mask_within_brain(image: np.ndarray, mask: np.ndarray) -> dict:
    """Check that lesion mask is within the brain region."""
    if mask.sum() == 0:
        return {"check": "mask_in_brain", "status": "skip", "reason": "no_lesion"}

    brain = np.abs(image[0]) > 0.01
    lesion_outside = (mask > 0) & (~brain)
    outside_ratio = lesion_outside.sum() / mask.sum()

    status = "pass" if outside_ratio < 0.05 else "fail"
    return {
        "check": "mask_in_brain",
        "status": status,
        "outside_ratio": round(float(outside_ratio), 4),
    }


def check_intensity_range(image: np.ndarray) -> dict:
    """Sanity check that normalized intensities are reasonable."""
    issues = []
    channel_names = ["trace", "adc", "flair"]

    for i, name in enumerate(channel_names):
        ch = image[i]
        nonzero = ch[np.abs(ch) > 1e-6]
        if len(nonzero) == 0:
            issues.append(f"{name}_all_zero")
            continue
        if np.abs(nonzero).max() > 20:
            issues.append(f"{name}_extreme_values")
        if np.isnan(nonzero).any():
            issues.append(f"{name}_has_nan")
        if np.isinf(nonzero).any():
            issues.append(f"{name}_has_inf")

    status = "pass" if len(issues) == 0 else "fail"
    return {"check": "intensity_range", "status": status, "issues": issues}


def check_empty_mask(mask: np.ndarray, expected_has_lesion: bool = True) -> dict:
    """Flag if mask is empty when it shouldn't be."""
    is_empty = mask.sum() == 0
    if expected_has_lesion and is_empty:
        return {"check": "empty_mask", "status": "warn", "reason": "expected_lesion_but_empty"}
    return {"check": "empty_mask", "status": "pass", "voxel_count": int(mask.sum())}


# ── Montage Generation ───────────────────────────────────────────────────────

def generate_montage(image: np.ndarray, mask: np.ndarray, sid: str,
                     output_path: Path):
    """Generate a 3-row montage: TRACE, ADC, FLAIR with mask overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick slices: find the slice with most lesion, or center if no lesion
    if mask.sum() > 0:
        lesion_per_slice = mask.sum(axis=(1, 2))
        center_z = int(np.argmax(lesion_per_slice))
    else:
        center_z = image.shape[1] // 2

    # 5 evenly spaced slices around center
    d = image.shape[1]
    offsets = [-d // 6, -d // 12, 0, d // 12, d // 6]
    slices = [max(0, min(d - 1, center_z + o)) for o in offsets]

    channel_names = ["TRACE", "ADC", "FLAIR"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f"{sid}", fontsize=14, fontweight="bold")

    for row, (ch_idx, ch_name) in enumerate(zip(range(3), channel_names)):
        for col, sl in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(image[ch_idx, sl, :, :].T, cmap="gray", origin="lower")

            # Overlay mask
            if mask[sl, :, :].sum() > 0:
                mask_slice = mask[sl, :, :].T.astype(float)
                ax.imshow(mask_slice, cmap="Reds", alpha=0.35, origin="lower",
                          vmin=0, vmax=1)

            ax.set_title(f"{ch_name} z={sl}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Per-subject QC ───────────────────────────────────────────────────────────

def qc_subject(npz_path: Path, montage_dir: Path = None) -> dict:
    """Run all QC checks on one preprocessed subject."""
    sid = npz_path.stem
    data = np.load(npz_path)
    image = data["image"]  # (3, D, H, W)
    mask = data["mask"]    # (D, H, W)

    checks = [
        check_trace_adc_consistency(image, mask),
        check_brain_coverage(image),
        check_mask_within_brain(image, mask),
        check_intensity_range(image),
        check_empty_mask(mask),
    ]

    overall = "pass"
    for c in checks:
        if c["status"] == "fail":
            overall = "fail"
            break
        if c["status"] == "warn" and overall == "pass":
            overall = "warn"

    result = {
        "subject_id": sid,
        "overall": overall,
        "checks": checks,
    }

    # Generate montage
    if montage_dir is not None:
        try:
            montage_path = montage_dir / f"{sid}.png"
            generate_montage(image, mask, sid, montage_path)
            result["montage"] = str(montage_path)
        except Exception as e:
            result["montage_error"] = str(e)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quality control for preprocessed SOOP data")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--no-montage", action="store_true", help="Skip montage generation")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preproc_dir = Path(config["paths"]["preprocessed"])
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    montage_dir = None
    if not args.no_montage:
        montage_dir = reports_dir / "montages"
        montage_dir.mkdir(parents=True, exist_ok=True)

    # Find all preprocessed files
    npz_files = sorted(preproc_dir.glob("sub-*.npz"))
    if args.max_subjects:
        npz_files = npz_files[:args.max_subjects]

    if not npz_files:
        print(f"ERROR: No preprocessed files found in {preproc_dir}")
        sys.exit(1)

    print(f"Running QC on {len(npz_files)} subjects")
    if montage_dir:
        print(f"Montages will be saved to {montage_dir}")

    # Run QC
    all_results = []
    for npz_path in tqdm(npz_files, desc="QC"):
        r = qc_subject(npz_path, montage_dir)
        all_results.append(r)

    # Build summary CSV
    import pandas as pd
    rows = []
    for r in all_results:
        row = {"subject_id": r["subject_id"], "overall": r["overall"]}
        for c in r["checks"]:
            row[c["check"]] = c["status"]
        rows.append(row)

    df = pd.DataFrame(rows)
    qc_csv = reports_dir / "qc_report.csv"
    df.to_csv(qc_csv, index=False)

    # Failures
    failures = df[df["overall"].isin(["fail", "warn"])]
    failures_csv = reports_dir / "qc_failures.csv"
    failures.to_csv(failures_csv, index=False)

    # Summary
    summary = {
        "total": len(df),
        "pass": int((df["overall"] == "pass").sum()),
        "warn": int((df["overall"] == "warn").sum()),
        "fail": int((df["overall"] == "fail").sum()),
        "per_check": {},
    }
    for check_col in [c for c in df.columns if c not in ("subject_id", "overall")]:
        summary["per_check"][check_col] = df[check_col].value_counts().to_dict()

    summary_json = reports_dir / "qc_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'=' * 50}")
    print("QC SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Pass:  {summary['pass']}")
    print(f"  Warn:  {summary['warn']}")
    print(f"  Fail:  {summary['fail']}")
    print(f"\n  Reports: {reports_dir}")
    print(f"  QC CSV:  {qc_csv}")
    print(f"  Failures: {failures_csv} ({len(failures)} subjects)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
