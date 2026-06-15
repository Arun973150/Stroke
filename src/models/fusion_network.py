"""
Fusion Network
==============
Learned adaptive fusion network for combining predictions from multiple
base models (nnU-Net, Swin-UNETR, SegResNet).

This module provides:
- AdaptiveFusionNetwork: Lightweight 3D CNN for learned fusion
- create_fusion_network: Factory function

The fusion network learns to:
- Weight different model predictions based on local context
- Use original image features to guide fusion decisions
- Output calibrated probability maps
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path


class AdaptiveFusionNetwork(nn.Module):
    """
    Lightweight 3D CNN for learned adaptive fusion of base model predictions.
    
    Takes as input:
    - Probability maps from 3 base models: [B, 3, H, W, D]
    - Original image sequences (DWI, ADC, FLAIR): [B, 3, H, W, D]
    - Total: 6 input channels
    
    Outputs:
    - Fused probability map: [B, 1, H, W, D]
    """
    
    def __init__(
        self,
        in_channels: int = 8,  # 3 predictions + 3 images + 1 physics + 1 uncertainty
        hidden_channels: Tuple[int, ...] = (16, 32, 16),
        out_channels: int = 1,
        dropout_prob: float = 0.1,
        use_residual: bool = True,
    ):
        """
        Initialize fusion network.
        
        Args:
            in_channels: Number of input channels (predictions + image)
            hidden_channels: Feature channels at each conv layer
            out_channels: Output channels (1 for fused probability)
            dropout_prob: Dropout probability
            use_residual: Add residual connection from average prediction
        """
        super().__init__()
        
        self.config = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'dropout_prob': dropout_prob,
            'use_residual': use_residual,
        }
        
        self.use_residual = use_residual
        
        # Build network layers
        layers = []
        prev_channels = in_channels
        
        for i, ch in enumerate(hidden_channels):
            layers.append(nn.Conv3d(prev_channels, ch, kernel_size=3, padding=1))
            layers.append(nn.InstanceNorm3d(ch))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            if i < len(hidden_channels) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout3d(dropout_prob))
            prev_channels = ch
        
        self.features = nn.Sequential(*layers)
        
        # Final output layer (No Sigmoid here, we work in Logit space for Residuals)
        self.output = nn.Conv3d(prev_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[FusionNetwork] Total parameters: {total_params:,}")
    
    def _init_weights(self):
        """Initialize weights with small values for stable learning."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        predictions: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            predictions: Base model predictions [B, 3, H, W, D] (probabilities)
            image: Original image sequences [B, 3, H, W, D] (optional)
                   Channels: 0: DWI, 1: ADC, 2: FLAIR
        
        Returns:
            Fused probability map [B, 1, H, W, D]
        """
        inputs = [predictions]
        
        if image is not None:
            # Physical Prior: Stroke is BRIGHT on DWI and DARK on ADC
            # dwi_adc_coupling = DWI * (1 - ADC)
            # This highlights areas where physics suggests a stroke
            dwi = image[:, 0:1]
            adc = image[:, 1:2]
            dwi_adc_coupling = dwi * (1.0 - adc.clamp(0, 1))
            
            # Uncertainty Map: Disagreement between the 3 experts
            uncertainty = predictions.std(dim=1, keepdim=True)
            
            # Combine all features
            inputs.append(image)
            inputs.append(dwi_adc_coupling)
            inputs.append(uncertainty)
        
        x = torch.cat(inputs, dim=1)
        
        # Extract features (the "delta" in logit space)
        logit_delta = self.output(self.features(x))
        
        # Robust Generalization: Base the result on the most stable expert (SegResNet)
        # SegResNet is predictions[:, 2:3] in our stack (Phase 3 logic)
        device = predictions.device
        base_reference = predictions[:, 2:3] if predictions.shape[1] >= 3 else predictions.mean(dim=1, keepdim=True)
        
        # Convert base prediction to logit space safely
        # Logit(p) = log(p / (1-p))
        eps = 1e-6
        base_reference = base_reference.clamp(eps, 1.0 - eps)
        base_logit = torch.log(base_reference / (1.0 - base_reference))
        
        if self.use_residual:
            # We ADD the learned delta to the base logit
            # logit(fused) = logit(base) + delta
            fused_logit = base_logit + logit_delta
            output = torch.sigmoid(fused_logit)
        else:
            # Traditional prediction
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
        print(f"[FusionNetwork] Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> Tuple['AdaptiveFusionNetwork', Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model_config = checkpoint.get('model_config', {})
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"[FusionNetwork] Loaded checkpoint from {path}")
        
        return model, checkpoint


class UncertaintyEstimator(nn.Module):
    """
    Estimate prediction uncertainty from ensemble disagreement.
    
    Higher uncertainty indicates:
    - Models disagree (ambiguous case)
    - Potential imaging artifact
    - Edge of lesion (boundary uncertainty)
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Uncertainty threshold for flagging cases
        """
        super().__init__()
        self.threshold = threshold
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from model predictions.
        
        Args:
            predictions: Stacked predictions [B, N_models, H, W, D]
        
        Returns:
            Uncertainty map [B, 1, H, W, D]
        """
        # Standard deviation across models
        uncertainty = predictions.std(dim=1, keepdim=True)
        return uncertainty
    
    def get_case_confidence(self, predictions: torch.Tensor, mask: torch.Tensor = None) -> float:
        """
        Compute case-level confidence score.
        
        Args:
            predictions: Model predictions [B, N_models, H, W, D]
            mask: Optional lesion mask to compute uncertainty over
        
        Returns:
            Confidence score (0-1, higher = more confident)
        """
        uncertainty = self.forward(predictions)
        
        if mask is not None:
            # Average uncertainty over lesion region
            masked_uncertainty = uncertainty * mask
            mean_uncertainty = masked_uncertainty.sum() / (mask.sum() + 1e-8)
        else:
            mean_uncertainty = uncertainty.mean()
        
        # Convert uncertainty to confidence
        confidence = 1.0 - mean_uncertainty.clamp(0, 1)
        return confidence.item()


def create_fusion_network(config: Dict = None) -> AdaptiveFusionNetwork:
    """
    Factory function to create fusion network from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized AdaptiveFusionNetwork
    """
    if config is None:
        config = {}
    
    fusion_config = config.get('fusion', {})
    
    return AdaptiveFusionNetwork(
        in_channels=fusion_config.get('in_channels', 8),
        hidden_channels=tuple(fusion_config.get('hidden_channels', [16, 32, 16])),
        out_channels=fusion_config.get('out_channels', 1),
        dropout_prob=fusion_config.get('dropout_prob', 0.1),
        use_residual=fusion_config.get('use_residual', True),
    )
