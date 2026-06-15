"""
Swin-UNETR Model
================
Wrapper for MONAI's SwinUNETR with pre-trained weights support.

This module provides:
- SwinUNETRWrapper: Model class with checkpoint loading utilities
- create_swin_unetr: Factory function for model creation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from monai.networks.nets import SwinUNETR


class SwinUNETRWrapper(nn.Module):
    """
    Wrapper around MONAI's SwinUNETR with additional utilities.
    
    Adds:
    - Pre-trained weight loading from MONAI Model Zoo
    - Checkpoint save/load utilities
    - Forward pass with optional deep supervision
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 3,
        out_channels: int = 2,
        feature_size: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        use_pretrained: bool = True,
        dropout_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_v2: bool = False,
    ):
        """
        Initialize Swin-UNETR model.
        
        Args:
            img_size: Input image size (H, W, D)
            in_channels: Number of input channels (3 for DWI, ADC, FLAIR)
            out_channels: Number of output classes (2 for background + lesion)
            feature_size: Base feature dimension (48 or 96)
            depths: Number of transformer blocks at each stage
            num_heads: Number of attention heads at each stage
            use_pretrained: Whether to load pre-trained weights
            dropout_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            use_v2: Use SwinUNETR V2 (if available)
        """
        super().__init__()
        
        # Note: img_size was removed in MONAI 1.5 - the check is now done in forward()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            drop_rate=dropout_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=True,
            use_checkpoint=False,  # Gradient checkpointing (saves memory)
            spatial_dims=3,
        )
        
        self.config = {
            'img_size': img_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feature_size': feature_size,
            'depths': depths,
            'num_heads': num_heads,
        }
        
        if use_pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights from MONAI Model Zoo."""
        try:
            # MONAI's pre-trained weights for SwinUNETR encoder
            # These are self-supervised pre-trained on large medical imaging datasets
            weight_url = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                "download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
            )
            
            print("[SwinUNETR] Loading pre-trained weights from MONAI Model Zoo...")
            
            # Download weights
            state_dict = torch.hub.load_state_dict_from_url(
                weight_url,
                progress=True,
                map_location='cpu',
            )
            
            # The pretrained weights have 'state_dict' key for encoder weights
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Get model state dict
            model_state = self.model.state_dict()
            loaded_count = 0
            
            # Try to match keys - pretrained weights use 'swinViT.' prefix
            for key, value in state_dict.items():
                # Remove 'module.' prefix if present
                if key.startswith('module.'):
                    key = key[7:]
                
                # The pretrained model uses 'swinViT' for encoder
                # Our model uses 'swinViT' directly
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        model_state[key] = value
                        loaded_count += 1
                else:
                    # Try adding 'swinViT.' prefix
                    prefixed_key = f"swinViT.{key}"
                    if prefixed_key in model_state:
                        if model_state[prefixed_key].shape == value.shape:
                            model_state[prefixed_key] = value
                            loaded_count += 1
            
            # Load the updated state dict
            self.model.load_state_dict(model_state, strict=False)
            
            total_keys = len(model_state)
            print(f"[SwinUNETR] Loaded {loaded_count}/{total_keys} pre-trained parameters")
            
            if loaded_count == 0:
                print("[SwinUNETR] Note: Pre-trained weights are for encoder only. Training from scratch.")
            
        except Exception as e:
            print(f"[SwinUNETR] Warning: Could not load pre-trained weights: {e}")
            print("[SwinUNETR] Training from scratch...")
    
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
            best_metric: Best validation metric
            **kwargs: Additional items to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'best_metric': best_metric,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        print(f"[SwinUNETR] Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['SwinUNETRWrapper', Dict]:
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
        config = checkpoint['config']
        
        # Create model with saved config
        model = cls(
            img_size=config['img_size'],
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            feature_size=config['feature_size'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            use_pretrained=False,  # Don't load pretrained, we have checkpoint
        )
        
        # Load saved weights
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"[SwinUNETR] Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 0)})")
        
        return model, checkpoint


def create_swin_unetr(config: Dict = None) -> SwinUNETRWrapper:
    """
    Factory function to create Swin-UNETR model from config.
    
    Args:
        config: Configuration dictionary. If None, uses defaults.
    
    Returns:
        Initialized SwinUNETRWrapper model
    """
    if config is None:
        config = {}
    
    model_config = config.get('model', {})
    
    model = SwinUNETRWrapper(
        img_size=tuple(model_config.get('img_size', [128, 128, 128])),
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 2),
        feature_size=model_config.get('feature_size', 48),
        depths=tuple(model_config.get('depths', [2, 2, 2, 2])),
        num_heads=tuple(model_config.get('num_heads', [3, 6, 12, 24])),
        use_pretrained=model_config.get('use_pretrained', True),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
        dropout_path_rate=model_config.get('dropout_path_rate', 0.0),
        use_v2=model_config.get('use_v2', False),
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[SwinUNETR] Model created:")
    print(f"  - Total parameters: {num_params / 1e6:.1f}M")
    print(f"  - Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  - Feature size: {model_config.get('feature_size', 48)}")
    
    return model
