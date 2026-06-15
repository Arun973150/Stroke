#!/usr/bin/env python3
"""
01. ISLES 2022 Preprocessing Script
====================================
Preprocesses raw ISLES 2022 data for nnU-Net and MONAI training.

This script:
1. Loads DWI, ADC, FLAIR, and ground truth masks
2. Resamples to isotropic spacing (1.95mm default)
3. Clips intensities to remove outliers
4. Crops to brain region
5. Saves in both nnU-Net and MONAI formats

Usage:
    python scripts/01_preprocess_isles.py --config configs/nnunet_config.yaml
    python scripts/01_preprocess_isles.py --raw-data /path/to/ISLES-2022 --output /path/to/output
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, print_header, print_step, print_success, 
    print_warning, print_error, print_info, get_progress_bar
)

warnings.filterwarnings('ignore')


class ISLESPreprocessor:
    """
    Preprocessor for ISLES 2022 stroke segmentation dataset.
    
    Handles:
    - Multi-modal input (DWI, ADC, FLAIR)
    - Resampling to target spacing
    - Intensity normalization
    - Output in both nnU-Net and MONAI formats
    """
    
    def __init__(
        self,
        raw_data_path: str,
        output_path: str,
        target_spacing: Tuple[float, float, float] = (1.95, 1.95, 1.95),
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ):
        """
        Initialize preprocessor.
        
        Args:
            raw_data_path: Path to raw ISLES 2022 data
            output_path: Path for preprocessed output
            target_spacing: Target voxel spacing in mm (x, y, z)
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.target_spacing = np.array(target_spacing)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Output directories
        self.nnunet_path = self.output_path / 'nnunet_format'
        self.monai_path = self.output_path / 'monai_format'
        self.metadata_path = self.output_path / 'metadata'
        
        # Tracking
        self.case_records: List[Dict] = []
        self.failed_cases: List[Dict] = []
        
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create output directory structure."""
        # nnU-Net format directories
        for split in ['Tr', 'Ts']:
            (self.nnunet_path / f'images{split}').mkdir(parents=True, exist_ok=True)
            (self.nnunet_path / f'labels{split}').mkdir(parents=True, exist_ok=True)
        
        # MONAI format directories
        for split in ['train', 'val', 'test']:
            (self.monai_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.monai_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Metadata directory
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def discover_cases(self) -> List[str]:
        """Find all subject directories in raw data."""
        cases = sorted([
            p.name for p in self.raw_data_path.iterdir()
            if p.is_dir() and p.name.startswith('sub-strokecase')
        ])
        return cases
    
    def get_file_paths(self, subject_id: str) -> Dict[str, Path]:
        """Get file paths for all modalities and mask."""
        session = 'ses-0001'
        base = self.raw_data_path / subject_id / session
        
        return {
            'dwi': base / 'dwi' / f'{subject_id}_{session}_dwi.nii.gz',
            'adc': base / 'dwi' / f'{subject_id}_{session}_adc.nii.gz',
            'flair': base / 'anat' / f'{subject_id}_{session}_FLAIR.nii.gz',
            'mask': self.raw_data_path / 'derivatives' / subject_id / session /
                    f'{subject_id}_{session}_msk.nii.gz'
        }
    
    def compute_target_shape(
        self, 
        shape: Tuple[int, ...], 
        src_spacing: np.ndarray
    ) -> Tuple[int, ...]:
        """Calculate target shape after resampling."""
        scale = src_spacing / self.target_spacing
        return tuple(np.round(np.array(shape) * scale).astype(int))
    
    def resample_volume(
        self,
        vol: np.ndarray,
        src_spacing: np.ndarray,
        tgt_shape: Tuple[int, ...],
        order: int
    ) -> np.ndarray:
        """
        Resample volume to target spacing and shape.
        
        Args:
            vol: Input volume
            src_spacing: Original voxel spacing
            tgt_shape: Target shape after resampling
            order: Interpolation order (3=cubic for images, 0=nearest for masks)
            
        Returns:
            Resampled volume
        """
        zoom_factors = src_spacing / self.target_spacing
        zoom_factors *= np.array(tgt_shape) / np.array(vol.shape)
        
        out = zoom(vol, zoom_factors, order=order, mode='constant', cval=0)
        
        # Ensure exact target shape
        if out.shape != tgt_shape:
            tmp = np.zeros(tgt_shape, dtype=out.dtype)
            slices = tuple(slice(0, min(out.shape[i], tgt_shape[i])) for i in range(3))
            tmp[slices] = out[slices]
            out = tmp
        
        return out
    
    def conservative_crop(
        self,
        volumes: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
        padding: int = 10
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], Dict]:
        """Crop volumes to region containing data with padding."""
        # Find union of all non-zero regions
        union = np.zeros_like(volumes[0], dtype=bool)
        for v in volumes:
            union |= (v > 0)
        
        coords = np.where(union)
        if len(coords[0]) == 0:
            return volumes, mask, {}
        
        # Calculate bounding box with padding
        z0 = int(max(coords[0].min() - padding, 0))
        z1 = int(min(coords[0].max() + padding + 1, volumes[0].shape[0]))
        y0 = int(max(coords[1].min() - padding, 0))
        y1 = int(min(coords[1].max() + padding + 1, volumes[0].shape[1]))
        x0 = int(max(coords[2].min() - padding, 0))
        x1 = int(min(coords[2].max() + padding + 1, volumes[0].shape[2]))
        
        # Crop all volumes
        cropped_vols = [v[z0:z1, y0:y1, x0:x1] for v in volumes]
        cropped_mask = mask[z0:z1, y0:y1, x0:x1] if mask is not None else None
        
        crop_bbox = {
            'z': [z0, z1],
            'y': [y0, y1],
            'x': [x0, x1]
        }
        
        return cropped_vols, cropped_mask, crop_bbox
    
    def compute_intensity_stats(
        self, 
        vol: np.ndarray, 
        mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute intensity statistics for normalization."""
        if mask is not None and mask.sum() > 0:
            vox = vol[mask]
        else:
            vox = vol[vol > 0]
        
        if len(vox) == 0:
            return {
                'p1': 0.0, 'p99': 0.0,
                'mean': 0.0, 'std': 1.0,
                'min': 0.0, 'max': 0.0
            }
        
        return {
            'p1': float(np.percentile(vox, 1)),
            'p99': float(np.percentile(vox, 99)),
            'mean': float(np.mean(vox)),
            'std': float(np.std(vox)),
            'min': float(np.min(vox)),
            'max': float(np.max(vox))
        }
    
    def clip_intensity(self, vol: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
        """Clip intensity values to percentile range."""
        return np.clip(vol, stats['p1'], stats['p99'])
    
    def analyze_lesion(self, mask: np.ndarray) -> Tuple[float, str, int]:
        """Analyze lesion size and categorize."""
        voxel_count = int(mask.sum())
        volume_ml = (voxel_count * np.prod(self.target_spacing)) / 1000.0
        
        if voxel_count == 0:
            category = 'none'
        elif volume_ml < 5:
            category = 'small'
        elif volume_ml < 50:
            category = 'medium'
        else:
            category = 'large'
        
        return volume_ml, category, voxel_count
    
    def process_case(self, subject_id: str, idx: int, split: str = 'train') -> Dict:
        """
        Process a single case through the full pipeline.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-strokecase0001')
            idx: Case index for output naming
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            Metadata dictionary for this case
        """
        paths = self.get_file_paths(subject_id)
        
        # Verify all files exist
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {key}: {path}")
        
        # Load all modalities
        dwi_nii = nib.load(paths['dwi'])
        adc_nii = nib.load(paths['adc'])
        flair_nii = nib.load(paths['flair'])
        mask_nii = nib.load(paths['mask'])
        
        dwi_data = dwi_nii.get_fdata().astype(np.float32)
        adc_data = adc_nii.get_fdata().astype(np.float32)
        flair_data = flair_nii.get_fdata().astype(np.float32)
        mask_data = (mask_nii.get_fdata() > 0).astype(np.uint8)
        
        # Get original spacing from affine
        original_spacing = np.abs(np.diag(dwi_nii.affine)[:3])
        
        # Compute target shape
        tgt_shape = self.compute_target_shape(dwi_data.shape, original_spacing)
        
        # Resample all volumes
        dwi_r = self.resample_volume(dwi_data, original_spacing, tgt_shape, order=3)
        adc_r = self.resample_volume(adc_data, original_spacing, tgt_shape, order=3)
        flair_r = self.resample_volume(flair_data, original_spacing, tgt_shape, order=3)
        mask_r = self.resample_volume(mask_data, original_spacing, tgt_shape, order=0)
        
        # Crop to region of interest
        [dwi_r, adc_r, flair_r], mask_r, crop_bbox = self.conservative_crop(
            [dwi_r, adc_r, flair_r], mask_r, padding=10
        )
        
        # Compute brain mask for intensity stats
        brain_mask = (dwi_r > 0) | (adc_r > 0) | (flair_r > 0)
        
        # Compute and apply intensity clipping
        stats_dwi = self.compute_intensity_stats(dwi_r, brain_mask)
        stats_adc = self.compute_intensity_stats(adc_r, brain_mask)
        stats_flair = self.compute_intensity_stats(flair_r, brain_mask)
        
        dwi_r = self.clip_intensity(dwi_r, stats_dwi)
        adc_r = self.clip_intensity(adc_r, stats_adc)
        flair_r = self.clip_intensity(flair_r, stats_flair)
        
        # Analyze lesion
        lesion_ml, category, voxel_count = self.analyze_lesion(mask_r)
        
        # Create affine with target spacing
        affine = np.eye(4)
        affine[:3, :3] = np.diag(self.target_spacing)
        
        # Generate case ID
        case_id = f'case_{idx:04d}'
        
        # Save in both formats
        self._save_nnunet_format(case_id, dwi_r, adc_r, flair_r, mask_r, affine, split)
        self._save_monai_format(case_id, dwi_r, adc_r, flair_r, mask_r, affine, split)
        
        # Create metadata
        metadata = {
            'case_id': case_id,
            'subject_id': subject_id,
            'split': split,
            'original_shape': [int(x) for x in dwi_data.shape],
            'resampled_shape': [int(x) for x in tgt_shape],
            'final_shape': [int(x) for x in dwi_r.shape],
            'original_spacing': [float(x) for x in original_spacing],
            'target_spacing': [float(x) for x in self.target_spacing],
            'crop_bbox': crop_bbox,
            'lesion_volume_ml': float(lesion_ml),
            'lesion_voxels': int(voxel_count),
            'lesion_category': category,
            'intensity_stats': {
                'dwi': stats_dwi,
                'adc': stats_adc,
                'flair': stats_flair
            }
        }
        
        # Save individual case metadata
        with open(self.metadata_path / f'{case_id}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _save_nnunet_format(
        self,
        case_id: str,
        dwi: np.ndarray,
        adc: np.ndarray,
        flair: np.ndarray,
        mask: np.ndarray,
        affine: np.ndarray,
        split: str
    ) -> None:
        """
        Save in nnU-Net format.
        
        nnU-Net expects:
        - imagesTr/caseid_0000.nii.gz (channel 0 = DWI)
        - imagesTr/caseid_0001.nii.gz (channel 1 = ADC)
        - imagesTr/caseid_0002.nii.gz (channel 2 = FLAIR)
        - labelsTr/caseid.nii.gz
        
        Note: Both train and val go to imagesTr/labelsTr (nnU-Net does its own CV)
        """
        # Train and Val both go to Tr folders (nnU-Net handles CV internally)
        suffix = 'Tr' if split in ['train', 'val'] else 'Ts'
        
        img_dir = self.nnunet_path / f'images{suffix}'
        lbl_dir = self.nnunet_path / f'labels{suffix}'
        
        nib.save(nib.Nifti1Image(dwi, affine), img_dir / f'{case_id}_0000.nii.gz')
        nib.save(nib.Nifti1Image(adc, affine), img_dir / f'{case_id}_0001.nii.gz')
        nib.save(nib.Nifti1Image(flair, affine), img_dir / f'{case_id}_0002.nii.gz')
        nib.save(nib.Nifti1Image(mask, affine), lbl_dir / f'{case_id}.nii.gz')
    
    def _save_monai_format(
        self,
        case_id: str,
        dwi: np.ndarray,
        adc: np.ndarray,
        flair: np.ndarray,
        mask: np.ndarray,
        affine: np.ndarray,
        split: str
    ) -> None:
        """Save in MONAI format (multi-channel stacked)."""
        img_dir = self.monai_path / split / 'images'
        lbl_dir = self.monai_path / split / 'labels'
        
        # Stack modalities as channels (C, H, W, D)
        stacked = np.stack([dwi, adc, flair], axis=0)
        
        nib.save(nib.Nifti1Image(stacked, affine), img_dir / f'{case_id}.nii.gz')
        nib.save(nib.Nifti1Image(mask, affine), lbl_dir / f'{case_id}.nii.gz')
    
    def create_dataset_json(self) -> None:
        """Create nnU-Net dataset.json file."""
        # Count train + val together for nnU-Net (it handles CV internally)
        num_training = len([r for r in self.case_records if r['split'] in ['train', 'val']])
        num_test = len([r for r in self.case_records if r['split'] == 'test'])
        
        dataset_info = {
            "channel_names": {
                "0": "DWI",
                "1": "ADC",
                "2": "FLAIR"
            },
            "labels": {
                "background": 0,
                "lesion": 1
            },
            "numTraining": num_training,
            "numTest": num_test,
            "file_ending": ".nii.gz",
            "dataset_name": "ISLES2022",
            "reference": "https://isles22.grand-challenge.org/",
            "description": "Ischemic stroke lesion segmentation from DWI, ADC, and FLAIR"
        }
        
        with open(self.nnunet_path / 'dataset.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def create_monai_datalist(self) -> None:
        """Create MONAI datalist.json file."""
        datalist = {
            'training': [],
            'validation': [],
            'testing': []
        }
        
        for record in self.case_records:
            case_id = record['case_id']
            split_map = {'train': 'training', 'val': 'validation', 'test': 'testing'}
            split_key = split_map[record['split']]
            
            entry = {
                'image': f"{record['split']}/images/{case_id}.nii.gz",
                'label': f"{record['split']}/labels/{case_id}.nii.gz",
                'case_id': case_id
            }
            datalist[split_key].append(entry)
        
        with open(self.monai_path / 'datalist.json', 'w') as f:
            json.dump(datalist, f, indent=2)
    
    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        print_header("ISLES 2022 - Global Preprocessing Pipeline")
        
        # Discover cases
        cases = self.discover_cases()
        if len(cases) == 0:
            print_error(f"No cases found in {self.raw_data_path}")
            print_info("Expected directories starting with 'sub-strokecase'")
            return
        
        print_info(f"Found {len(cases)} cases")
        
        # Calculate split sizes
        n_train = int(len(cases) * self.train_ratio)
        n_val = int(len(cases) * self.val_ratio)
        
        train_cases = cases[:n_train]
        val_cases = cases[n_train:n_train + n_val]
        test_cases = cases[n_train + n_val:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_cases)} cases ({self.train_ratio:.0%})")
        print(f"  Val:   {len(val_cases)} cases ({self.val_ratio:.0%})")
        print(f"  Test:  {len(test_cases)} cases ({1-self.train_ratio-self.val_ratio:.0%})")
        
        # Process all cases
        print_step(1, 4, "Processing cases...")
        
        idx = 1
        all_splits = [('train', train_cases), ('val', val_cases), ('test', test_cases)]
        
        for split, case_list in all_splits:
            print(f"\nProcessing {split} split ({len(case_list)} cases)...")
            
            for subject_id in tqdm(case_list, desc=f"  {split}"):
                try:
                    metadata = self.process_case(subject_id, idx, split)
                    self.case_records.append(metadata)
                    idx += 1
                except Exception as e:
                    print_warning(f"Failed: {subject_id} - {str(e)}")
                    self.failed_cases.append({
                        'subject_id': subject_id,
                        'split': split,
                        'error': str(e)
                    })
        
        # Create summary files
        print_step(2, 4, "Creating summary files...")
        
        df_records = pd.DataFrame(self.case_records)
        df_records.to_csv(self.metadata_path / 'case_records.csv', index=False)
        print_success(f"Saved case_records.csv ({len(df_records)} cases)")
        
        if self.failed_cases:
            df_failed = pd.DataFrame(self.failed_cases)
            df_failed.to_csv(self.metadata_path / 'failed_cases.csv', index=False)
            print_warning(f"Some cases failed - see failed_cases.csv")
        
        # Create dataset JSONs
        print_step(3, 4, "Creating dataset configuration files...")
        
        self.create_dataset_json()
        print_success("Created nnU-Net dataset.json")
        
        self.create_monai_datalist()
        print_success("Created MONAI datalist.json")
        
        # Print summary
        print_step(4, 4, "Preprocessing complete!")
        print_header("Summary")
        
        print(f"\nSuccessfully processed: {len(self.case_records)} cases")
        print(f"Failed cases: {len(self.failed_cases)}")
        
        if len(df_records) > 0:
            print("\nLesion size distribution:")
            for cat in ['small', 'medium', 'large', 'none']:
                count = len(df_records[df_records['lesion_category'] == cat])
                pct = count / len(df_records) * 100
                print(f"  {cat}: {count} cases ({pct:.1f}%)")
            print(f"\nAverage lesion volume: {df_records['lesion_volume_ml'].mean():.2f} mL")
        
        print(f"\nOutput saved to: {self.output_path}")
        print(f"  nnunet_format/  (for nnU-Net v2)")
        print(f"  monai_format/   (for Swin-UNETR & SegResNet)")
        print(f"  metadata/       (statistics & metadata)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess ISLES 2022 data for stroke segmentation"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        help='Override: Path to raw ISLES-2022 data'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Override: Path for preprocessed output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print_warning(f"Config file not found: {args.config}")
        print_info("Using command line arguments or defaults")
        config = {'paths': {}, 'preprocessing': {}}
    
    # Override with command line arguments
    raw_data = args.raw_data or config.get('paths', {}).get('raw_data', '/home/ubuntu/data/ISLES-2022')
    output = args.output or config.get('paths', {}).get('preprocessed', '/home/ubuntu/data/preprocessed')
    
    preproc_config = config.get('preprocessing', {})
    target_spacing = tuple(preproc_config.get('target_spacing', [1.95, 1.95, 1.95]))
    train_ratio = preproc_config.get('train_ratio', 0.70)
    val_ratio = preproc_config.get('val_ratio', 0.15)
    
    # Create preprocessor and run
    preprocessor = ISLESPreprocessor(
        raw_data_path=raw_data,
        output_path=output,
        target_spacing=target_spacing,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
