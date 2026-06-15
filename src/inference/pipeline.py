"""
Local Inference Pipeline
========================
Unified service for running the full 16-model Sentinel Stroke ensemble locally.

This service manages:
1. High-fidelity preprocessing of raw NIfTI files.
2. Parallel inference across 5 folds of nnU-Net, SegResNet and Swin-UNETR.
3. Post-processing and fusion using the v4/v6 Specialist Brain.
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# MONAI imports
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, NormalizeIntensity, 
    Orientation, Spacing, CenterSpatialCrop, Resize, ToTensor
)
from monai.inferers import sliding_window_inference

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segresnet import SegResNetWrapper
from src.models.swin_unetr import SwinUNETRWrapper
from src.models.fusion_network import AdaptiveFusionNetwork

# Try to import nnU-Net wrapper
try:
    from src.models.nnunet_wrapper import nnUNetWrapper, load_nnunet_ensemble
    HAS_NNUNET = True
except ImportError:
    HAS_NNUNET = False
    print("[WARNING] nnU-Net wrapper not available. Will use fallback.")

class LocalEnsemblePipeline:
    """
    Orchestrates 16-model ensemble inference on raw MRI scans.
    
    Models:
    - 5 nnU-Net folds
    - 5 SegResNet folds
    - 5 Swin-UNETR folds
    - 1 Fusion network (v4 or v6)
    """
    
    def __init__(
        self,
        base_models_dir: str = "trained_models",
        fusion_checkpoint: Optional[str] = None,
        device: str = "cuda",
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        use_nnunet: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.roi_size = roi_size
        self.models_dir = Path(base_models_dir)
        self.use_nnunet = use_nnunet and HAS_NNUNET
        
        print(f"[Pipeline] Initializing on device: {self.device}")
        
        # 1. Load nnU-Net Folds (0-4) - NEW!
        self.nnunets = []
        if self.use_nnunet:
            nn_dir = self.models_dir / "nnunet"
            for fold in range(5):
                ckpt = nn_dir / f"fold_{fold}" / "checkpoint_final.pth"
                if ckpt.exists():
                    try:
                        model, _ = nnUNetWrapper.load_checkpoint(ckpt, device=self.device)
                        model.eval()
                        self.nnunets.append(model)
                    except Exception as e:
                        print(f"[WARNING] Failed to load nnU-Net fold {fold}: {e}")
            print(f"[Pipeline] Loaded {len(self.nnunets)} nnU-Net folds")
        
        # 2. Load SegResNet Folds (0-4)
        self.segresnets = []
        seg_dir = self.models_dir / "segresnet"
        for fold in range(5):
            ckpt = seg_dir / f"fold_{fold}" / "checkpoint_best.pth"
            if ckpt.exists():
                model, _ = SegResNetWrapper.load_checkpoint(ckpt, device=self.device)
                model.eval()
                self.segresnets.append(model)
        print(f"[Pipeline] Loaded {len(self.segresnets)} SegResNet folds")
        
        # 3. Load Swin-UNETR Folds (0-4)
        self.swin_unetrs = []
        swin_dir = self.models_dir / "swin_unetr"
        for fold in range(5):
            ckpt = swin_dir / f"fold_{fold}" / "checkpoint_best.pth"
            if ckpt.exists():
                model, _ = SwinUNETRWrapper.load_checkpoint(ckpt, device=self.device)
                model.eval()
                self.swin_unetrs.append(model)
        print(f"[Pipeline] Loaded {len(self.swin_unetrs)} Swin-UNETR folds")
                
        # 4. Load Fusion Network (v4 or v6)
        if fusion_checkpoint:
            fusion_ckpt = Path(fusion_checkpoint)
        else:
            # Try v6 first, then v4
            fusion_ckpt = Path("checkpoints/fusion_v6_best.pth")
            if not fusion_ckpt.exists():
                fusion_ckpt = self.models_dir / "fusion_network_v4_specialist.pth"
        
        self.fusion_net = AdaptiveFusionNetwork(in_channels=8, use_residual=True).to(self.device)
        if fusion_ckpt.exists():
            ckpt_data = torch.load(fusion_ckpt, map_location=self.device, weights_only=False)
            self.fusion_net.load_state_dict(ckpt_data['model_state_dict'])
            print(f"[Pipeline] Loaded fusion network from: {fusion_ckpt}")
        else:
            print(f"[WARNING] Fusion checkpoint not found: {fusion_ckpt}")
        self.fusion_net.eval()
        
        total_models = len(self.nnunets) + len(self.segresnets) + len(self.swin_unetrs) + 1
        print(f"[Pipeline] Total models loaded: {total_models}")

    def preprocess_raw(self, dwi_path: str, adc_path: str, flair_path: str) -> torch.Tensor:
        """Converts raw NIfTI files into a unified 3-channel tensor [1, 3, H, W, D].
        
        Handles mismatched image sizes by resampling all modalities to a common
        isotropic spacing and then resizing to match the DWI dimensions.
        """
        # Load with resampling to isotropic spacing
        base_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),  # Resample to 1.5mm isotropic
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])
        
        # Load all modalities
        dwi = base_transforms(dwi_path)
        adc = base_transforms(adc_path)
        flair = base_transforms(flair_path)
        
        # Use DWI shape as reference and resize others to match
        target_shape = dwi.shape[1:]  # (H, W, D) excluding channel dim
        
        # Resize ADC and FLAIR to match DWI if needed
        if adc.shape[1:] != target_shape:
            resize_transform = Resize(spatial_size=target_shape, mode="trilinear")
            adc = resize_transform(adc)
        
        if flair.shape[1:] != target_shape:
            resize_transform = Resize(spatial_size=target_shape, mode="trilinear")
            flair = resize_transform(flair)
        
        # Convert to tensors
        dwi = torch.as_tensor(dwi).to(self.device)
        adc = torch.as_tensor(adc).to(self.device)
        flair = torch.as_tensor(flair).to(self.device)
        
        # Stack channels: 0:DWI, 1:ADC, 2:FLAIR
        stacked = torch.cat([dwi, adc, flair], dim=0).unsqueeze(0)  # [1, 3, H, W, D]
        
        print(f"[Pipeline] Preprocessed image shape: {stacked.shape}")
        return stacked

    @torch.no_grad()
    def run_inference(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs full 16-model ensemble + fusion.
        
        Pipeline:
        1. Run all 5 nnU-Net folds → average
        2. Run all 5 Swin-UNETR folds → average
        3. Run all 5 SegResNet folds → average
        4. Fuse with Adaptive Fusion Network
        """
        print("[Pipeline] Running inference...")
        
        # --- Stage 1: nnU-Net Ensemble ---
        if self.nnunets:
            print(f"  Running {len(self.nnunets)} nnU-Net folds...")
            nn_preds = []
            for i, model in enumerate(self.nnunets):
                out = sliding_window_inference(image, self.roi_size, 4, model)
                # nnU-Net outputs logits for [background, lesion]
                nn_preds.append(torch.softmax(out, dim=1)[:, 1:])
            nn_avg = torch.mean(torch.stack(nn_preds), dim=0)
        else:
            nn_avg = None
        
        # --- Stage 2: Swin-UNETR Ensemble ---
        print(f"  Running {len(self.swin_unetrs)} Swin-UNETR folds...")
        swin_preds = []
        for model in self.swin_unetrs:
            out = sliding_window_inference(image, self.roi_size, 4, model)
            swin_preds.append(torch.softmax(out, dim=1)[:, 1:])
        swin_avg = torch.mean(torch.stack(swin_preds), dim=0)
        
        # --- Stage 3: SegResNet Ensemble ---
        print(f"  Running {len(self.segresnets)} SegResNet folds...")
        seg_preds = []
        for model in self.segresnets:
            out = sliding_window_inference(image, self.roi_size, 4, model)
            seg_preds.append(torch.softmax(out, dim=1)[:, 1:])
        seg_avg = torch.mean(torch.stack(seg_preds), dim=0)

        # --- Stage 4: Adaptive Fusion ---
        print("  Running fusion network...")
        
        # Use real nnU-Net if available, otherwise fallback to average
        if nn_avg is not None:
            nnunet_pred = nn_avg
        else:
            # Fallback: use average of Swin and SegResNet
            nnunet_pred = (swin_avg + seg_avg) / 2.0
            print("  [Note] Using Swin+SegResNet average as nnU-Net proxy")
        
        # Order: [nnU-Net, Swin-UNETR, SegResNet]
        fusion_inputs = torch.cat([nnunet_pred, swin_avg, seg_avg], dim=1)
        
        fused_output = self.fusion_net(fusion_inputs, image)
        
        print("[Pipeline] Inference complete!")
        return fused_output

def run_local_case(
    dwi: str, 
    adc: str, 
    flair: str, 
    output_path: str,
    fusion_checkpoint: Optional[str] = None,
    use_nnunet: bool = True,
):
    """
    Convenience function for one-off local cases.
    
    Args:
        dwi: Path to DWI NIfTI file
        adc: Path to ADC NIfTI file
        flair: Path to FLAIR NIfTI file
        output_path: Path for output segmentation
        fusion_checkpoint: Optional path to fusion checkpoint (default: auto-detect v6/v4)
        use_nnunet: Whether to use nnU-Net (default: True)
    """
    pipeline = LocalEnsemblePipeline(
        fusion_checkpoint=fusion_checkpoint,
        use_nnunet=use_nnunet,
    )
    image = pipeline.preprocess_raw(dwi, adc, flair)
    result = pipeline.run_inference(image)
    
    # Threshold to binary
    result_binary = (result > 0.5).float()
    
    # Save as NIfTI
    result_np = result_binary.squeeze().cpu().numpy().astype(np.uint8)
    
    # We use DWI header for the final NIfTI
    dwi_img = nib.load(dwi)
    final_img = nib.Nifti1Image(result_np, dwi_img.affine, dwi_img.header)
    nib.save(final_img, output_path)
    print(f"[Pipeline] Final segmentation saved to: {output_path}")

if __name__ == "__main__":
    # Test stub
    pass
