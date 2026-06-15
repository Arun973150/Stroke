"""
SOOP Preprocessing Pipeline (Phase 1)
======================================
For each subject:
  1. BIDS-aware file discovery (TRACE, ADC, FLAIR, acute mask)
  2. Co-register FLAIR → TRACE space (rigid, mutual information)
  3. Skull strip using SynthStrip (on FLAIR, propagate mask)
  4. Intensity normalize per-modality
  5. Resample to target isotropic resolution
  6. Crop/pad to fixed dimensions
  7. Save as .npz (channels + mask)

Outputs per subject:
  preprocessed/<subject_id>.npz  with keys:
    - image: (3, D, H, W) float32  [TRACE, ADC, FLAIR]
    - mask:  (D, H, W) uint8       [acute lesion]
    - mask_chronic: (D, H, W) uint8 [chronic lesion, if exists]
    - spacing: original spacing
    - affine: original affine

Usage:
  python scripts/01_preprocess_soop.py --config configs/soop_config.yaml
  python scripts/01_preprocess_soop.py --config configs/soop_config.yaml --subjects sub-1 sub-2
  python scripts/01_preprocess_soop.py --config configs/soop_config.yaml --workers 8
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── BIDS File Discovery ─────────────────────────────────────────────────────

def find_file(base_dir: Path, pattern: str):
    matches = list(base_dir.rglob(pattern))
    return matches[0] if matches else None


def discover_files(raw_dir: Path, derivatives_dir: Path, sid: str) -> dict:
    """Discover all files for a subject using BIDS naming."""
    sub_dir = raw_dir / sid
    files = {
        "trace": find_file(sub_dir, f"{sid}_rec-TRACE_dwi.nii.gz"),
        "adc": find_file(sub_dir, f"{sid}_rec-ADC_dwi.nii.gz"),
        "flair": find_file(sub_dir, f"{sid}_FLAIR.nii.gz"),
        "adc_json": find_file(sub_dir, f"{sid}_rec-ADC_dwi.json"),
    }
    # Masks
    mask_dir = derivatives_dir / "lesion_masks" / sid / "dwi"
    if not mask_dir.exists():
        mask_dir = derivatives_dir / "lesion_masks" / sid
    if not mask_dir.exists():
        mask_dir = derivatives_dir / sid / "dwi"

    files["mask_acute"] = find_file(mask_dir, "*desc-lesionAcute_mask.nii.gz") if mask_dir.exists() else None
    files["mask_chronic"] = find_file(mask_dir, "*desc-lesionChronic_mask.nii.gz") if mask_dir.exists() else None
    return files


# ── Registration ─────────────────────────────────────────────────────────────

def rigid_register(fixed_path: Path, moving_path: Path) -> sitk.Image:
    """Register moving image to fixed image space using rigid + mutual info."""
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)

    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = reg.Execute(fixed, moving)
        resampled = sitk.Resample(
            moving, fixed, final_transform,
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        return resampled
    except Exception as e:
        print(f"    WARNING: Registration failed ({e}), using resampling only")
        resampled = sitk.Resample(
            moving, fixed, sitk.Transform(),
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        return resampled


def resample_mask_to_ref(mask_path: Path, ref_image: sitk.Image) -> sitk.Image:
    """Resample a mask to reference space using nearest-neighbor."""
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    resampled = sitk.Resample(
        mask, ref_image, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, mask.GetPixelID()
    )
    return resampled


# ── Skull Stripping ──────────────────────────────────────────────────────────

def skull_strip_synthstrip(image_sitk: sitk.Image) -> sitk.Image:
    """
    Skull strip using SynthStrip via command line.
    Falls back to simple threshold if SynthStrip is not installed.
    Returns binary brain mask.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.nii.gz"
        mask_path = Path(tmpdir) / "mask.nii.gz"

        sitk.WriteImage(image_sitk, str(in_path))

        try:
            subprocess.run(
                ["synthstrip", "-i", str(in_path), "-m", str(mask_path)],
                capture_output=True, timeout=300, check=True
            )
            brain_mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
            return brain_mask
        except (FileNotFoundError, subprocess.CalledProcessError):
            # SynthStrip not available — use mri_synthstrip (FreeSurfer)
            try:
                subprocess.run(
                    ["mri_synthstrip", "-i", str(in_path), "--mask", str(mask_path)],
                    capture_output=True, timeout=300, check=True
                )
                brain_mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
                return brain_mask
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

    # Final fallback: Otsu threshold
    print("    WARNING: SynthStrip not found, using Otsu threshold fallback")
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(image_sitk)
    # Morphological cleanup
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    mask = sitk.BinaryFillhole(mask)
    return sitk.Cast(mask, sitk.sitkUInt8)


# ── Intensity Normalization ──────────────────────────────────────────────────

def normalize_zscore(data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Z-score normalize using brain-only voxels, clip to [p1, p99]."""
    brain_voxels = data[brain_mask > 0]
    if len(brain_voxels) == 0:
        return data
    p1, p99 = np.percentile(brain_voxels, [1, 99])
    data = np.clip(data, p1, p99)
    brain_voxels = data[brain_mask > 0]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    if std < 1e-8:
        return data - mean
    return (data - mean) / std


def normalize_adc(data: np.ndarray, brain_mask: np.ndarray,
                  adc_json: dict = None, clip_range=(0, 3000)) -> np.ndarray:
    """
    Normalize ADC map.
    If values look like standard units (x10^-6 mm^2/s): clip and scale to [0, 1].
    Otherwise: z-score normalize.
    """
    brain_voxels = data[brain_mask > 0]
    if len(brain_voxels) == 0:
        return data

    brain_mean = brain_voxels.mean()

    # Heuristic: if mean > 100, likely in x10^-6 mm^2/s units
    if brain_mean > 100:
        data = np.clip(data, clip_range[0], clip_range[1])
        data = data / clip_range[1]  # scale to [0, 1]
    else:
        # Likely already scaled or different units — z-score
        data = normalize_zscore(data, brain_mask)

    return data


# ── Resampling ───────────────────────────────────────────────────────────────

def resample_to_spacing(image_sitk: sitk.Image, target_spacing: list,
                        interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample a SimpleITK image to target isotropic spacing."""
    orig_size = image_sitk.GetSize()
    orig_spacing = image_sitk.GetSpacing()

    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)
    ]

    resampled = sitk.Resample(
        image_sitk,
        new_size,
        sitk.Transform(),
        interpolator,
        image_sitk.GetOrigin(),
        target_spacing,
        image_sitk.GetDirection(),
        0.0,
        image_sitk.GetPixelID()
    )
    return resampled


# ── Crop / Pad ───────────────────────────────────────────────────────────────

def crop_pad_to_shape(data: np.ndarray, target_shape: tuple,
                      pad_value: float = 0.0) -> np.ndarray:
    """Center crop or pad a 3D array to target shape."""
    result = np.full(target_shape, pad_value, dtype=data.dtype)

    # Compute start/end for source and destination
    starts_src = []
    ends_src = []
    starts_dst = []
    ends_dst = []

    for i in range(3):
        if data.shape[i] >= target_shape[i]:
            # Crop
            offset = (data.shape[i] - target_shape[i]) // 2
            starts_src.append(offset)
            ends_src.append(offset + target_shape[i])
            starts_dst.append(0)
            ends_dst.append(target_shape[i])
        else:
            # Pad
            offset = (target_shape[i] - data.shape[i]) // 2
            starts_src.append(0)
            ends_src.append(data.shape[i])
            starts_dst.append(offset)
            ends_dst.append(offset + data.shape[i])

    result[
        starts_dst[0]:ends_dst[0],
        starts_dst[1]:ends_dst[1],
        starts_dst[2]:ends_dst[2]
    ] = data[
        starts_src[0]:ends_src[0],
        starts_src[1]:ends_src[1],
        starts_src[2]:ends_src[2]
    ]
    return result


# ── Main per-subject processing ──────────────────────────────────────────────

def process_subject(sid: str, raw_dir: Path, derivatives_dir: Path,
                    output_dir: Path, config: dict) -> dict:
    """Process a single subject through the full pipeline."""
    result = {"subject_id": sid, "status": "success", "errors": []}
    output_path = output_dir / f"{sid}.npz"

    if output_path.exists():
        result["status"] = "skipped"
        return result

    try:
        # 1) Discover files
        files = discover_files(raw_dir, derivatives_dir, sid)

        if files["trace"] is None:
            result["status"] = "failed"
            result["errors"].append("MISSING_TRACE")
            return result

        if files["adc"] is None:
            result["status"] = "failed"
            result["errors"].append("MISSING_ADC")
            return result

        # Load TRACE as the reference space
        trace_sitk = sitk.ReadImage(str(files["trace"]), sitk.sitkFloat32)
        adc_sitk = sitk.ReadImage(str(files["adc"]), sitk.sitkFloat32)

        # 2) Co-register FLAIR → TRACE space
        if files["flair"] is not None:
            flair_sitk = rigid_register(files["trace"], files["flair"])
        else:
            # No FLAIR — create zeros
            flair_sitk = sitk.Image(trace_sitk.GetSize(), sitk.sitkFloat32)
            flair_sitk.CopyInformation(trace_sitk)
            result["errors"].append("MISSING_FLAIR_ZEROED")

        # 3) Skull strip (on FLAIR if available, else TRACE)
        strip_input = flair_sitk if files["flair"] else trace_sitk
        brain_mask_sitk = skull_strip_synthstrip(strip_input)

        # Ensure brain mask is in TRACE space
        if brain_mask_sitk.GetSize() != trace_sitk.GetSize():
            brain_mask_sitk = sitk.Resample(
                brain_mask_sitk, trace_sitk, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
            )

        brain_mask_np = sitk.GetArrayFromImage(brain_mask_sitk).astype(np.uint8)

        # 4) Convert to numpy and normalize
        trace_np = sitk.GetArrayFromImage(trace_sitk).astype(np.float32)
        adc_np = sitk.GetArrayFromImage(adc_sitk).astype(np.float32)
        flair_np = sitk.GetArrayFromImage(flair_sitk).astype(np.float32)

        # Read ADC JSON for unit detection
        adc_json = {}
        if files["adc_json"]:
            with open(files["adc_json"]) as f:
                adc_json = json.load(f)

        trace_np = normalize_zscore(trace_np, brain_mask_np)
        adc_np = normalize_adc(adc_np, brain_mask_np, adc_json,
                               config["preprocessing"].get("adc_clip_range", [0, 3000]))
        flair_np = normalize_zscore(flair_np, brain_mask_np)

        # Apply brain mask
        trace_np *= brain_mask_np
        adc_np *= brain_mask_np
        flair_np *= brain_mask_np

        # 5) Resample to target spacing
        target_spacing = config["preprocessing"]["target_spacing"]
        orig_spacing = list(trace_sitk.GetSpacing())

        # Build SimpleITK images from normalized numpy
        def np_to_sitk(arr, ref):
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(ref)
            return img

        trace_sitk_n = resample_to_spacing(np_to_sitk(trace_np, trace_sitk), target_spacing)
        adc_sitk_n = resample_to_spacing(np_to_sitk(adc_np, trace_sitk), target_spacing)
        flair_sitk_n = resample_to_spacing(np_to_sitk(flair_np, trace_sitk), target_spacing)

        trace_np = sitk.GetArrayFromImage(trace_sitk_n).astype(np.float32)
        adc_np = sitk.GetArrayFromImage(adc_sitk_n).astype(np.float32)
        flair_np = sitk.GetArrayFromImage(flair_sitk_n).astype(np.float32)

        # 6) Crop/pad to target shape
        target_shape = tuple(config["preprocessing"]["target_shape"])
        trace_np = crop_pad_to_shape(trace_np, target_shape)
        adc_np = crop_pad_to_shape(adc_np, target_shape)
        flair_np = crop_pad_to_shape(flair_np, target_shape)

        # Stack channels: (3, D, H, W)
        image = np.stack([trace_np, adc_np, flair_np], axis=0)

        # 7) Process masks
        save_dict = {
            "image": image,
            "spacing": np.array(orig_spacing),
            "target_spacing": np.array(target_spacing),
        }

        if files["mask_acute"]:
            mask_sitk = sitk.ReadImage(str(files["mask_acute"]), sitk.sitkUInt8)
            mask_sitk = resample_to_spacing(mask_sitk, target_spacing,
                                            interpolator=sitk.sitkNearestNeighbor)
            mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)
            mask_np = crop_pad_to_shape(mask_np, target_shape)
            save_dict["mask"] = mask_np
        else:
            # No mask = negative control
            save_dict["mask"] = np.zeros(target_shape, dtype=np.uint8)
            result["errors"].append("NO_ACUTE_MASK_ZEROED")

        if files["mask_chronic"]:
            cmask_sitk = sitk.ReadImage(str(files["mask_chronic"]), sitk.sitkUInt8)
            cmask_sitk = resample_to_spacing(cmask_sitk, target_spacing,
                                             interpolator=sitk.sitkNearestNeighbor)
            cmask_np = sitk.GetArrayFromImage(cmask_sitk).astype(np.uint8)
            cmask_np = crop_pad_to_shape(cmask_np, target_shape)
            save_dict["mask_chronic"] = cmask_np

        np.savez_compressed(output_path, **save_dict)

    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess SOOP dataset")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Process specific subjects only")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default 1)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw_data"])
    derivatives_dir = Path(config["paths"].get("derivatives", str(raw_dir.parent / "derivatives")))
    output_dir = Path(config["paths"]["preprocessed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: Raw data not found: {raw_dir}")
        sys.exit(1)

    # Discover subjects
    if args.subjects:
        subject_ids = args.subjects
    else:
        subject_ids = sorted([
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])

    print(f"Subjects to process: {len(subject_ids)}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}\n")

    # Process subjects
    results = []

    if args.workers <= 1:
        for sid in tqdm(subject_ids, desc="Preprocessing"):
            r = process_subject(sid, raw_dir, derivatives_dir, output_dir, config)
            results.append(r)
            if r["status"] == "failed":
                print(f"  FAILED {sid}: {r['errors']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_subject, sid, raw_dir, derivatives_dir, output_dir, config): sid
                for sid in subject_ids
            }
            pbar = tqdm(total=len(futures), desc="Preprocessing")
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                pbar.update(1)
                if r["status"] == "failed":
                    print(f"  FAILED {r['subject_id']}: {r['errors']}")
            pbar.close()

    # Summary
    statuses = [r["status"] for r in results]
    print(f"\n{'=' * 50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Success:  {statuses.count('success')}")
    print(f"  Skipped:  {statuses.count('skipped')}")
    print(f"  Failed:   {statuses.count('failed')}")

    # Save processing log
    log_path = output_dir / "preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
