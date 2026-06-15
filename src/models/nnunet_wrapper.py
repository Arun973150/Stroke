"""
nnU-Net Wrapper for Local Inference
====================================
Loads nnU-Net v2 checkpoints and provides inference without nnUNet CLI.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Try to import dynamic_network_architectures (installed with nnunetv2)
try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
    HAS_DNA = True
except ImportError:
    HAS_DNA = False
    print("[WARNING] dynamic_network_architectures not found. Install with: pip install nnunetv2")


class nnUNetWrapper(nn.Module):
    """
    Wrapper for nnU-Net v2 models.
    
    Loads checkpoints trained with nnUNetv2 and provides a consistent
    forward() interface matching other models in the pipeline.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        n_stages: int = 5,
        features_per_stage: Tuple[int, ...] = (32, 64, 128, 256, 320),
        conv_op: str = "torch.nn.modules.conv.Conv3d",
        kernel_sizes: Tuple = ((3,3,3), (3,3,3), (3,3,3), (3,3,3), (3,3,3)),
        strides: Tuple = ((1,1,1), (2,2,2), (2,2,2), (2,2,2), (2,2,2)),
        n_conv_per_stage: Tuple[int, ...] = (2, 2, 2, 2, 2),
        n_conv_per_stage_decoder: Tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        
        if not HAS_DNA:
            raise ImportError(
                "dynamic_network_architectures is required for nnU-Net. "
                "Install with: pip install nnunetv2"
            )
        
        self.config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'n_stages': n_stages,
            'features_per_stage': features_per_stage,
        }
        
        # Build the network using dynamic_network_architectures
        self.network = PlainConvUNet(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            conv_op=nn.Conv3d,
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            num_classes=out_channels,
            deep_supervision=False,  # We don't need deep supervision for inference
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Logits tensor [B, num_classes, H, W, D]
        """
        return self.network(x)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['nnUNetWrapper', Dict]:
        """
        Load model from nnU-Net checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint_final.pth
            device: Device to load model on
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract architecture config from init_args
        init_args = ckpt.get('init_args', {})
        plans = init_args.get('plans', {})
        config_name = init_args.get('configuration', '3d_fullres')
        config = plans.get('configurations', {}).get(config_name, {})
        arch = config.get('architecture', {}).get('arch_kwargs', {})
        
        # Get dataset info for channels
        dataset_json = init_args.get('dataset_json', {})
        channel_names = dataset_json.get('channel_names', {})
        in_channels = len(channel_names) if channel_names else 3
        
        labels = dataset_json.get('labels', {})
        out_channels = len(labels) if labels else 2
        
        # Create model with architecture from checkpoint
        model = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            n_stages=arch.get('n_stages', 5),
            features_per_stage=tuple(arch.get('features_per_stage', [32, 64, 128, 256, 320])),
            kernel_sizes=tuple(tuple(k) for k in arch.get('kernel_sizes', [[3,3,3]]*5)),
            strides=tuple(tuple(s) for s in arch.get('strides', [[1,1,1]] + [[2,2,2]]*4)),
            n_conv_per_stage=tuple(arch.get('n_conv_per_stage', [2, 2, 2, 2, 2])),
            n_conv_per_stage_decoder=tuple(arch.get('n_conv_per_stage_decoder', [2, 2, 2, 2])),
        )
        
        # Load weights
        model.network.load_state_dict(ckpt['network_weights'])
        model = model.to(device)
        model.eval()
        
        print(f"[nnUNetWrapper] Loaded from {checkpoint_path.name}")
        print(f"[nnUNetWrapper] Architecture: {arch.get('n_stages', 5)} stages, "
              f"features: {arch.get('features_per_stage', [])}")
        
        return model, ckpt
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['nnUNetWrapper', Dict]:
        """Alias for from_checkpoint for API consistency."""
        return cls.from_checkpoint(path, device)


def load_nnunet_ensemble(
    models_dir: Union[str, Path],
    device: Union[str, torch.device] = 'cuda',
    num_folds: int = 5,
) -> list:
    """
    Load all nnU-Net folds as an ensemble.
    
    Args:
        models_dir: Directory containing fold_*/checkpoint_final.pth
        device: Device to load models on
        num_folds: Number of folds (default 5)
        
    Returns:
        List of loaded nnUNetWrapper models
    """
    models_dir = Path(models_dir)
    models = []
    
    for fold in range(num_folds):
        ckpt_path = models_dir / f"fold_{fold}" / "checkpoint_final.pth"
        if ckpt_path.exists():
            model, _ = nnUNetWrapper.from_checkpoint(ckpt_path, device)
            models.append(model)
        else:
            print(f"[WARNING] Fold {fold} checkpoint not found: {ckpt_path}")
    
    print(f"[nnUNetWrapper] Loaded {len(models)}/{num_folds} folds")
    return models


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "trained_models/nnunet/fold_0/checkpoint_final.pth"
    
    print(f"Testing nnU-Net wrapper with: {ckpt_path}")
    
    try:
        model, ckpt = nnUNetWrapper.from_checkpoint(ckpt_path, device='cpu')
        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 80, 112, 128)
        with torch.no_grad():
            out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
