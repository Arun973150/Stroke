"""
SegResNet Model
===============
Wrapper for MONAI's SegResNet with checkpoint utilities.

This module provides:
- SegResNetWrapper: Model class with checkpoint loading utilities
- create_segresnet: Factory function for model creation

SegResNet is a fully convolutional residual network designed for
stability-focused segmentation with:
- Residual connections for gradient flow
- Group normalization for stable training with small batches
- Conservative predictions with fewer false positives
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from monai.networks.nets import SegResNet


class SegResNetWrapper(nn.Module):
    """
    Wrapper around MONAI's SegResNet with additional utilities.
    
    Adds:
    - Checkpoint save/load utilities
    - Configuration-based initialization
    
    SegResNet characteristics:
    - Residual blocks with group normalization
    - Encoder-decoder architecture
    - Skip connections for spatial detail preservation
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 3,
        out_channels: int = 2,
        init_filters: int = 32,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        dropout_prob: float = 0.2,
        norm: str = "GROUP",
        num_groups: int = 8,
        use_conv_final: bool = True,
        upsample_mode: str = "nontrainable",
    ):
        """
        Initialize SegResNet model.
        
        Args:
            spatial_dims: Number of spatial dimensions (3 for 3D medical imaging)
            in_channels: Number of input channels (3 for DWI, ADC, FLAIR)
            out_channels: Number of output classes (2 for background + lesion)
            init_filters: Initial number of filters (doubles at each level)
            blocks_down: Number of residual blocks at each encoder level
            blocks_up: Number of residual blocks at each decoder level
            dropout_prob: Dropout probability in residual blocks
            norm: Normalization type ("GROUP", "BATCH", "INSTANCE")
            num_groups: Number of groups for group normalization
            use_conv_final: Use convolution for final layer
            upsample_mode: Upsampling mode ("nontrainable", "deconv")
        """
        super().__init__()
        
        # Store configuration
        self.config = {
            'spatial_dims': spatial_dims,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'init_filters': init_filters,
            'blocks_down': blocks_down,
            'blocks_up': blocks_up,
            'dropout_prob': dropout_prob,
            'norm': norm,
            'num_groups': num_groups,
            'use_conv_final': use_conv_final,
            'upsample_mode': upsample_mode,
        }
        
        # Prepare norm parameter for MONAI
        # MONAI expects (norm_type, dict_of_params) for specific settings
        if norm.upper() == "GROUP":
            monai_norm = ("GROUP", {"num_groups": num_groups})
        else:
            monai_norm = norm
            
        # Create MONAI SegResNet
        self.model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            norm=monai_norm,
            use_conv_final=use_conv_final,
            upsample_mode=upsample_mode,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SegResNet] Total parameters: {total_params:,}")
        print(f"[SegResNet] Trainable parameters: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W, D]
        
        Returns:
            Output tensor of shape [B, num_classes, H, W, D]
        """
        return self.model(x)
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        best_metric: float = 0.0,
        **kwargs,
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            epoch: Current epoch
            best_metric: Best validation metric so far
            **kwargs: Additional items to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'best_metric': best_metric,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[SegResNet] Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['SegResNetWrapper', Dict]:
        """
        Load model from checkpoint.
        
        Args:
            path: Checkpoint path
            device: Device to load model to
        
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        # PyTorch 2.6+ defaults to weights_only=True, which can fail with legacy checkpoints
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older torch versions lacking weights_only arg
            checkpoint = torch.load(path, map_location=device)
        
        # Get model config from checkpoint
        model_config = checkpoint.get('model_config', {})
        
        # Create model with saved config
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"[SegResNet] Loaded checkpoint from {path}")
        print(f"[SegResNet] Epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")
        
        return model, checkpoint


def create_segresnet(config: Dict = None) -> SegResNetWrapper:
    """
    Factory function to create SegResNet model from config.
    
    Args:
        config: Configuration dictionary. If None, uses defaults.
    
    Returns:
        Initialized SegResNetWrapper model
    """
    if config is None:
        config = {}
    
    model_config = config.get('model', {})
    
    # Extract parameters with defaults
    return SegResNetWrapper(
        spatial_dims=model_config.get('spatial_dims', 3),
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 2),
        init_filters=model_config.get('init_filters', 32),
        blocks_down=tuple(model_config.get('blocks_down', [1, 2, 2, 4])),
        blocks_up=tuple(model_config.get('blocks_up', [1, 1, 1])),
        dropout_prob=model_config.get('dropout_prob', 0.2),
        norm=model_config.get('norm', 'GROUP'),
        num_groups=model_config.get('num_groups', 8),
        use_conv_final=model_config.get('use_conv_final', True),
        upsample_mode=model_config.get('upsample_mode', 'nontrainable'),
    )
