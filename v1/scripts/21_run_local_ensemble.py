#!/usr/bin/env python3
"""
21. Run Local Ensemble
======================
The "One-Click" local inference tool for Sentinel Stroke.

Usage:
    # Basic usage (uses all 16 models + v6 fusion)
    python scripts/21_run_local_ensemble.py --dwi DWI.nii.gz --adc ADC.nii.gz --flair FLAIR.nii.gz --output result.nii.gz
    
    # Skip nnU-Net for faster inference (uses 10 models)
    python scripts/21_run_local_ensemble.py --dwi DWI.nii.gz --adc ADC.nii.gz --flair FLAIR.nii.gz --output result.nii.gz --no-nnunet
    
    # Use specific fusion checkpoint
    python scripts/21_run_local_ensemble.py --dwi DWI.nii.gz --adc ADC.nii.gz --flair FLAIR.nii.gz --fusion checkpoints/fusion_v6_best.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import run_local_case
from src.utils import print_header, print_success, print_info, print_warning

def main():
    parser = argparse.ArgumentParser(description="Sentinel Stroke Local Inference")
    parser.add_argument('--dwi', type=str, required=True, help='Path to DWI NIfTI')
    parser.add_argument('--adc', type=str, required=True, help='Path to ADC NIfTI')
    parser.add_argument('--flair', type=str, required=True, help='Path to FLAIR NIfTI')
    parser.add_argument('--output', type=str, default='stroke_segmentation.nii.gz', help='Output NIfTI path')
    parser.add_argument('--fusion', type=str, default=None, help='Path to fusion checkpoint (default: auto-detect)')
    parser.add_argument('--no-nnunet', action='store_true', help='Skip nnU-Net for faster inference')
    
    args = parser.parse_args()
    
    print_header("Sentinel Stroke: Local Inference Engine")
    print_info(f"Input DWI: {args.dwi}")
    print_info(f"Input ADC: {args.adc}")
    print_info(f"Input FLAIR: {args.flair}")
    
    if args.no_nnunet:
        print_warning("nnU-Net disabled - using 10 models instead of 15")
    
    if args.fusion:
        print_info(f"Fusion checkpoint: {args.fusion}")
    
    try:
        run_local_case(
            args.dwi, 
            args.adc, 
            args.flair, 
            args.output,
            fusion_checkpoint=args.fusion,
            use_nnunet=not args.no_nnunet,
        )
        print_success("\nInference Complete!")
        print_info(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"\n[Error] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
