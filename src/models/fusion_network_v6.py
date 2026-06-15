"""
Fusion Network v6
=================
Enhanced fusion network with Squeeze-and-Excitation attention blocks.

Key changes from v4:
- Added SE attention blocks between conv layers
- Same logit-space residual on SegResNet (unchanged)
- Same input channels and overall structure

The SE blocks learn channel-wise attention weights, allowing the network
to focus on the most informative model predictions per-voxel.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path


class SqueezeExcite3D(nn.Module):
    """
    3D Squeeze-and-Excitation block for channel attention.
    
    Learns to weight channels based on global context, helping the
    fusion network decide which base model to trust at each location.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        
        reduced_channels = max(channels // reduction, 4)
        
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Attention-weighted tensor [B, C, H, W, D]
        """
        b, c, _, _, _ = x.shape
        
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        
        # Excite: channel attention weights
        y = self.excite(y).view(b, c, 1, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ConvBlockWithSE(nn.Module):
    """
    Convolution block with SE attention.
    
    Conv3d → InstanceNorm → LeakyReLU → SE_Attention → Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.1,
        use_se: bool = True,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = SqueezeExcite3D(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        x = self.dropout(x)
        return x


class AdaptiveFusionNetworkV6(nn.Module):
    """
    v6 Fusion Network with SE attention blocks.
    
    Architecture:
    - Input: 8 channels (3 predictions + 3 images + 1 DWI-ADC coupling + 1 uncertainty)
    - ConvBlock(8→16) + SE
    - ConvBlock(16→32) + SE
    - ConvBlock(32→16) + SE (no dropout)
    - Conv1x1(16→1) → logit delta
    - Output: sigmoid(logit(SegResNet) + delta)
    
    The SE blocks learn which channels (model predictions) to trust per-voxel.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: Tuple[int, ...] = (16, 32, 16),
        out_channels: int = 1,
        dropout_prob: float = 0.1,
        use_residual: bool = True,
        use_se: bool = True,  # NEW: toggle SE attention
    ):
        """
        Initialize v6 fusion network.
        
        Args:
            in_channels: Number of input channels (3 preds + 3 imgs + 2 derived)
            hidden_channels: Feature channels at each layer
            out_channels: Output channels (1 for binary segmentation)
            dropout_prob: Dropout probability
            use_residual: Use logit-space residual on SegResNet
            use_se: Use SE attention blocks (NEW in v6)
        """
        super().__init__()
        
        self.config = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'dropout_prob': dropout_prob,
            'use_residual': use_residual,
            'use_se': use_se,
            'version': 'v6',
        }
        
        self.use_residual = use_residual
        self.use_se = use_se
        
        # Build network with SE attention
        layers = []
        prev_ch = in_channels
        
        for i, ch in enumerate(hidden_channels):
            is_last = (i == len(hidden_channels) - 1)
            layers.append(ConvBlockWithSE(
                prev_ch, ch,
                dropout_prob=0.0 if is_last else dropout_prob,
                use_se=use_se,
            ))
            prev_ch = ch
        
        self.features = nn.Sequential(*layers)
        
        # Output layer (no sigmoid - we work in logit space)
        self.output = nn.Conv3d(prev_ch, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[FusionNetworkV6] Total parameters: {total_params:,}")
        print(f"[FusionNetworkV6] SE attention: {'enabled' if use_se else 'disabled'}")
    
    def _init_weights(self):
        """Initialize with small weights for stable residual learning."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self,
        predictions: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with logit-space residual on SegResNet.
        
        Args:
            predictions: Base model predictions [B, 3, H, W, D]
                         Order: [nnunet, swin_unetr, segresnet]
            image: Original MRI sequences [B, 3, H, W, D]
                   Channels: [DWI, ADC, FLAIR]
        
        Returns:
            Fused probability map [B, 1, H, W, D]
        """
        inputs = [predictions]
        
        if image is not None:
            # Physical Prior: DWI bright + ADC dark = acute stroke
            dwi = image[:, 0:1]
            adc = image[:, 1:2]
            dwi_adc_coupling = dwi * (1.0 - adc.clamp(0, 1))
            
            # Uncertainty: disagreement between base models
            uncertainty = predictions.std(dim=1, keepdim=True)
            
            inputs.append(image)
            inputs.append(dwi_adc_coupling)
            inputs.append(uncertainty)
        
        x = torch.cat(inputs, dim=1)
        
        # Extract features with SE attention
        features = self.features(x)
        
        # Compute logit delta
        logit_delta = self.output(features)
        
        # Logit-space residual on SegResNet (UNCHANGED from v4)
        # SegResNet is predictions[:, 2:3] (index 2 in the stack)
        base_ref = predictions[:, 2:3] if predictions.shape[1] >= 3 else predictions.mean(dim=1, keepdim=True)
        
        # Safe logit conversion
        eps = 1e-6
        base_ref = base_ref.clamp(eps, 1.0 - eps)
        base_logit = torch.log(base_ref / (1.0 - base_ref))
        
        if self.use_residual:
            # v4/v6 approach: add learned delta to SegResNet logit
            fused_logit = base_logit + logit_delta
            output = torch.sigmoid(fused_logit)
        else:
            output = torch.sigmoid(logit_delta)
        
        return output
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        best_metric: float = 0.0,
        **kwargs,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'best_metric': best_metric,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[FusionNetworkV6] Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['AdaptiveFusionNetworkV6', Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model_config = checkpoint.get('model_config', {})
        # Remove version key if present (not a constructor arg)
        model_config.pop('version', None)
        
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"[FusionNetworkV6] Loaded checkpoint from {path}")
        
        return model, checkpoint


def create_fusion_network_v6(use_se: bool = True) -> AdaptiveFusionNetworkV6:
    """
    Factory function for v6 fusion network.
    
    Args:
        use_se: Enable SE attention (True for v6, False for ablation)
        
    Returns:
        Initialized AdaptiveFusionNetworkV6
    """
    return AdaptiveFusionNetworkV6(
        in_channels=8,
        hidden_channels=(16, 32, 16),
        out_channels=1,
        dropout_prob=0.1,
        use_residual=True,
        use_se=use_se,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing FusionNetworkV6...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_fusion_network_v6(use_se=True).to(device)
    
    # Test input
    predictions = torch.rand(2, 3, 32, 32, 32).to(device)
    image = torch.rand(2, 3, 32, 32, 32).to(device)
    
    # Forward pass
    output = model(predictions, image)
    
    print(f"Input predictions: {predictions.shape}")
    print(f"Input image: {image.shape}")
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("Test passed!")
