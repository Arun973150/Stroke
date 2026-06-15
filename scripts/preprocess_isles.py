"""
Preprocess ISLES-2022 to SOOP-compatible .npz format
=====================================================
Converts ISLES-2022 data into the same format as our SOOP preprocessed files,
so we can run our 5-model ensemble for external validation.

Mapping:
  ISLES channel      ->  SOOP slot
  dwi.nii.gz         ->  Channel 0 (TRACE slot)   [both are b=1000 DWI]
  adc.nii.gz         ->  Channel 1 (ADC)
  FLAIR.nii.gz       ->  Channel 2 (FLAIR)

Preprocessing:
  - Resample all to 1mm isotropic
  - FLAIR resampled to DWI space (nearest affine)
  - Crop/pad to 192^3
  - Z-score normalization per channel (brain-masked)
  - Save as .npz with "image" (3,D,H,W) and "mask" keys

Usage:
  python scripts/preprocess_isles.py --isles-dir v1/ISLES-2022/ISLES-2022 --out-dir isles_preprocessed
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def resample_to_spacing(data, orig_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """Resample 3D array to target isotropic spacing."""
    zoom_factors = [orig_spacing[i] / target_spacing[i] for i in range(3)]
    return zoom(data, zoom_factors, order=order, mode="constant", cval=0)


def resample_to_reference(moving_img, ref_img, order=1):
    """Resample `moving_img` (nibabel) to match `ref_img` grid.

    Uses affine-aware resampling — assumes DWI/ADC are already co-registered
    in scanner coords (typical for ISLES which comes from same session).
    For FLAIR, this produces best-effort resampling without rigid alignment;
    acceptable for external validation but not perfect.
    """
    from scipy.ndimage import affine_transform

    moving_data = moving_img.get_fdata()
    ref_shape = ref_img.shape

    # Transform: moving_coords = inv(A_moving) @ A_ref @ ref_coords
    ref_affine = ref_img.affine
    mov_affine = moving_img.affine

    transform = np.linalg.inv(mov_affine) @ ref_affine
    mat = transform[:3, :3]
    offset = transform[:3, 3]

    resampled = affine_transform(
        moving_data, mat, offset=offset, output_shape=ref_shape,
        order=order, mode="constant", cval=0,
    )
    return resampled


def crop_or_pad_to_shape(data, target_shape=(192, 192, 192), constant_values=0):
    """Center-crop or pad each axis to target shape."""
    result = data.copy()
    for axis in range(3):
        current = result.shape[axis]
        target = target_shape[axis]
        if current < target:
            # Pad symmetrically
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_config = [(0, 0)] * result.ndim
            pad_config[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_config, mode="constant",
                            constant_values=constant_values)
        elif current > target:
            # Center crop
            start = (current - target) // 2
            end = start + target
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(start, end)
            result = result[tuple(slicer)]
    return result


def zscore_normalize(image, brain_mask=None):
    """Z-score within brain mask (or non-zero voxels if no mask)."""
    if brain_mask is None:
        brain_mask = image > 0
    if brain_mask.sum() == 0:
        return image.astype(np.float32)

    mean = image[brain_mask].mean()
    std = image[brain_mask].std()
    if std < 1e-8:
        return image.astype(np.float32)

    normed = (image - mean) / std
    normed[~brain_mask] = 0
    return normed.astype(np.float32)


def preprocess_subject(subj_dir, deriv_dir, out_path, target_shape=(192, 192, 192),
                        target_spacing=(1.0, 1.0, 1.0)):
    """Preprocess one ISLES subject and save as .npz matching SOOP format."""
    sid = subj_dir.name
    ses_dir = subj_dir / "ses-0001"

    dwi_path = ses_dir / "dwi" / f"{sid}_ses-0001_dwi.nii.gz"
    adc_path = ses_dir / "dwi" / f"{sid}_ses-0001_adc.nii.gz"
    flair_path = ses_dir / "anat" / f"{sid}_ses-0001_FLAIR.nii.gz"
    mask_path = deriv_dir / sid / "ses-0001" / f"{sid}_ses-0001_msk.nii.gz"

    for p in [dwi_path, adc_path, flair_path, mask_path]:
        if not p.exists():
            return {"sid": sid, "status": "missing", "file": p.name}

    # Load — DWI as reference
    dwi_img = nib.load(dwi_path)
    adc_img = nib.load(adc_path)
    flair_img = nib.load(flair_path)
    mask_img = nib.load(mask_path)

    dwi = dwi_img.get_fdata().astype(np.float32)
    # If DWI is 4D (multi-volume), take b=1000 (usually last or highest)
    if dwi.ndim == 4:
        dwi = dwi[..., -1]

    # Resample ADC and FLAIR to DWI grid (affine-aware)
    adc = resample_to_reference(adc_img, dwi_img, order=1).astype(np.float32)
    flair = resample_to_reference(flair_img, dwi_img, order=1).astype(np.float32)
    mask = mask_img.get_fdata().astype(np.float32)
    if mask.shape != dwi.shape:
        mask = resample_to_reference(mask_img, dwi_img, order=0)
    mask = (mask > 0.5).astype(np.uint8)

    # Resample all to 1mm iso
    orig_spacing = dwi_img.header.get_zooms()[:3]
    dwi = resample_to_spacing(dwi, orig_spacing, target_spacing, order=1)
    adc = resample_to_spacing(adc, orig_spacing, target_spacing, order=1)
    flair = resample_to_spacing(flair, orig_spacing, target_spacing, order=1)
    mask = resample_to_spacing(mask.astype(np.float32), orig_spacing,
                                target_spacing, order=0)
    mask = (mask > 0.5).astype(np.uint8)

    # Center crop/pad to 192^3
    dwi = crop_or_pad_to_shape(dwi, target_shape)
    adc = crop_or_pad_to_shape(adc, target_shape)
    flair = crop_or_pad_to_shape(flair, target_shape)
    mask = crop_or_pad_to_shape(mask, target_shape)

    # Brain mask from DWI
    brain_mask = dwi > np.percentile(dwi[dwi > 0], 5) if (dwi > 0).any() else dwi > 0

    # Z-score normalize each channel
    dwi_n = zscore_normalize(dwi, brain_mask)
    adc_n = zscore_normalize(adc, brain_mask)
    flair_n = zscore_normalize(flair, brain_mask)

    image = np.stack([dwi_n, adc_n, flair_n], axis=0).astype(np.float32)

    np.savez_compressed(out_path, image=image, mask=mask)

    return {
        "sid": sid, "status": "ok",
        "mask_vol_vox": int(mask.sum()),
        "mask_vol_ml": float(mask.sum() * np.prod(target_spacing) / 1000.0),
        "shape": list(image.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isles-dir", required=True,
                        help="Path to ISLES-2022 root (contains sub-strokecase*)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N subjects (for testing)")
    args = parser.parse_args()

    isles_dir = Path(args.isles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deriv_dir = isles_dir / "derivatives"
    subj_dirs = sorted([d for d in isles_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sub-strokecase")])

    if args.limit:
        subj_dirs = subj_dirs[:args.limit]

    print(f"Processing {len(subj_dirs)} subjects from {isles_dir}")
    print(f"Output: {out_dir}")

    results = []
    errors = 0

    for subj_dir in tqdm(subj_dirs, desc="Preprocessing"):
        sid = subj_dir.name
        out_path = out_dir / f"{sid}.npz"

        if out_path.exists():
            # Skip already done
            continue

        try:
            result = preprocess_subject(subj_dir, deriv_dir, out_path)
            results.append(result)
            if result["status"] != "ok":
                errors += 1
                tqdm.write(f"  {sid}: {result['status']} ({result.get('file', '')})")
        except Exception as e:
            errors += 1
            tqdm.write(f"  {sid}: EXCEPTION {type(e).__name__}: {e}")
            results.append({"sid": sid, "status": "error", "error": str(e)})

    # Save manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save test split (all subjects)
    ok_sids = [r["sid"] for r in results if r.get("status") == "ok"]
    splits_dir = out_dir.parent / "splits"
    splits_dir.mkdir(exist_ok=True)
    with open(splits_dir / "isles_test.json", "w") as f:
        json.dump(ok_sids, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ISLES PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Success: {len(ok_sids)}")
    print(f"  Errors:  {errors}")
    print(f"  Output:  {out_dir}")
    print(f"  Split:   {splits_dir / 'isles_test.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
