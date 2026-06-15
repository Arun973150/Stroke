"""
Convert NIfTI files to DICOM series for radiologist review.

Output structure:
  debug_cases/dicom/sub-XX/
    DWI/      — TRACE images (bright = restricted diffusion)
    ADC/      — ADC maps
    MASK/     — Ground truth lesion mask (binary)

Radiologist opens in PACS/Horos/OsiriX and overlays MASK on DWI.

Usage:
  python scripts/nifti_to_dicom.py --raw-dir debug_cases/raw --out-dir debug_cases/dicom
"""

import argparse
import subprocess
from pathlib import Path


def convert_one(nii_path, out_dir, series_desc, series_num):
    """Use nii2dcm to convert one NIfTI to a DICOM series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nii2dcm",
        str(nii_path),
        str(out_dir),
        "-d", "MR",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"    OK: {nii_path.name} -> {out_dir} ({len(list(out_dir.glob('*.dcm')))} slices)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    FAIL: {nii_path.name}: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"    FAIL: nii2dcm not installed. Run: pip install nii2dcm")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="debug_cases/raw")
    parser.add_argument("--out-dir", default="debug_cases/dicom")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    subjects = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for sid_dir in subjects:
        sid = sid_dir.name
        print(f"\n{sid}:")

        # Find DWI, ADC, mask
        dwi = next(sid_dir.glob("*rec-TRACE_dwi.nii.gz"), None)
        adc = next(sid_dir.glob("*rec-ADC_dwi.nii.gz"), None)
        mask = next(sid_dir.glob("*lesionAcute_mask.nii.gz"), None)

        if dwi:
            convert_one(dwi, out_dir / sid / "DWI", "DWI_TRACE", 1)
        else:
            print("    missing DWI")

        if adc:
            convert_one(adc, out_dir / sid / "ADC", "ADC", 2)
        else:
            print("    missing ADC")

        if mask:
            convert_one(mask, out_dir / sid / "MASK", "LESION_MASK", 3)
        else:
            print("    missing mask")

    print(f"\nDone. DICOM files in: {out_dir}")
    print("\nTo view:")
    print("  1. Open Horos/OsiriX/RadiAnt/MicroDicom (free DICOM viewers)")
    print("  2. Import the sub-XX folder")
    print("  3. Open DWI series, overlay MASK series")


if __name__ == "__main__":
    main()
