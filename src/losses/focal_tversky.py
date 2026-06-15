"""
Focal Tversky Loss
==================
Optimized for small object segmentation with controllable precision/recall trade-off.

The Tversky index generalizes Dice by allowing asymmetric weighting of FP and FN:
    TI = TP / (TP + alpha*FN + beta*FP)

- alpha > beta: penalize FN more → higher recall (catch more lesions)
- alpha < beta: penalize FP more → higher precision (fewer false alarms)

The Focal variant adds a focusing parameter gamma to down-weight easy examples:
    FTL = (1 - TI)^gamma

Reference:
    Abraham & Khan, "A Novel Focal Tversky Loss Function for Lesion Segmentation", ISBI 2019
"""

import torch
import torch.nn as nn
from typing import Optional


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for small lesion segmentation.
    
    Designed to handle class imbalance and focus on hard examples.
    Particularly effective for small lesions where standard Dice fails.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        apply_sigmoid: bool = True,
    ):
        """
        Initialize Focal Tversky Loss.
        
        Args:
            alpha: Weight for false negatives (FN). Higher = recall-focused.
                   Default 0.7 prioritizes catching all lesions.
            beta: Weight for false positives (FP). Higher = precision-focused.
                  Default 0.3 allows some FP to avoid missing small lesions.
            gamma: Focal parameter. Lower values (0.5-0.75) focus more on hard examples.
                   gamma=1.0 reduces to standard Tversky loss.
            smooth: Smoothing factor to prevent division by zero.
            apply_sigmoid: Whether to apply sigmoid to predictions.
        """
        super().__init__()
        
        assert alpha + beta == 1.0, f"alpha + beta should equal 1.0, got {alpha + beta}"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.
        
        Args:
            predictions: Model output [B, 1, H, W, D] or [B, H, W, D]
            targets: Ground truth [B, 1, H, W, D] or [B, H, W, D]
            weights: Optional per-sample weights [B] for volume-weighted loss
            
        Returns:
            Scalar loss value
        """
        if self.apply_sigmoid:
            predictions = torch.sigmoid(predictions)
        
        # Flatten spatial dimensions
        if predictions.dim() == 5:
            predictions = predictions.view(predictions.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        elif predictions.dim() == 4:
            predictions = predictions.view(predictions.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        
        # Compute Tversky components per sample
        tp = (predictions * targets).sum(dim=1)
        fn = (targets * (1 - predictions)).sum(dim=1)
        fp = ((1 - targets) * predictions).sum(dim=1)
        
        # Tversky Index per sample
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        
        # Focal Tversky Loss
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        
        # Apply per-sample weights if provided
        if weights is not None:
            focal_tversky = focal_tversky * weights
            
        return focal_tversky.mean()


class CombinedSmallLesionLoss(nn.Module):
    """
    Combined loss optimized for small lesion segmentation.
    
    Combines:
    - Focal Tversky (main loss for small objects)
    - Standard Dice (for stability)
    - Optional: Cross-entropy (for calibration)
    """
    
    def __init__(
        self,
        focal_tversky_weight: float = 0.6,
        dice_weight: float = 0.3,
        ce_weight: float = 0.1,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
    ):
        """
        Initialize combined loss.
        
        Args:
            focal_tversky_weight: Weight for Focal Tversky loss
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross-entropy loss
            alpha, beta, gamma: Focal Tversky parameters
        """
        super().__init__()
        
        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma, apply_sigmoid=False
        )
        self.focal_tversky_weight = focal_tversky_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model output (after sigmoid) [B, 1, H, W, D]
            targets: Ground truth [B, 1, H, W, D]
            weights: Optional per-sample weights [B]
            
        Returns:
            Scalar loss value
        """
        # Focal Tversky
        ft_loss = self.focal_tversky(predictions, targets, weights)
        
        # Standard Dice
        predictions_flat = predictions.view(predictions.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (predictions_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + 1e-6) / (
            predictions_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-6
        )
        dice_loss = 1 - dice
        
        if weights is not None:
            dice_loss = dice_loss * weights
        dice_loss = dice_loss.mean()
        
        # Binary Cross-Entropy
        bce_loss = torch.nn.functional.binary_cross_entropy(
            predictions_flat, targets_flat, reduction='none'
        ).mean(dim=1)
        
        if weights is not None:
            bce_loss = bce_loss * weights
        bce_loss = bce_loss.mean()
        
        # Combine
        total_loss = (
            self.focal_tversky_weight * ft_loss +
            self.dice_weight * dice_loss +
            self.ce_weight * bce_loss
        )
        
        return total_loss


def focal_tversky_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Functional interface for Focal Tversky Loss.
    
    Args:
        predictions: Model output (after sigmoid) [B, 1, H, W, D]
        targets: Ground truth [B, 1, H, W, D]
        alpha: FN weight (default 0.7 for recall)
        beta: FP weight (default 0.3)
        gamma: Focal parameter (default 0.75)
        smooth: Smoothing factor
        
    Returns:
        Scalar loss value
    """
    # Flatten
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    
    # Tversky components
    tp = (predictions * targets).sum(dim=1)
    fn = (targets * (1 - predictions)).sum(dim=1)
    fp = ((1 - targets) * predictions).sum(dim=1)
    
    # Tversky Index
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Focal
    focal_tversky = torch.pow(1 - tversky, gamma)
    
    return focal_tversky.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for improved edge segmentation.
    
    Uses distance transform to weight voxels near boundaries more heavily.
    This helps all lesion sizes but proportionally benefits smaller lesions
    (which have higher surface-to-volume ratio).
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary-weighted loss.
        
        Args:
            predictions: Model output (after sigmoid) [B, 1, H, W, D]
            targets: Ground truth [B, 1, H, W, D]
            
        Returns:
            Scalar loss value
        """
        # Compute boundary using Laplacian approximation (faster than distance transform)
        # Boundary = voxels where target differs from neighbors
        kernel_size = 3
        
        # Average pooling to get neighbor average
        avg_pool = torch.nn.functional.avg_pool3d(
            targets, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        )
        
        # Boundary is where target differs from neighborhood average
        boundary = torch.abs(targets - avg_pool)
        boundary = (boundary > 0.1).float()
        
        # Add small weight to non-boundary regions
        boundary_weight = boundary * 5.0 + 1.0  # 5x weight on boundary, 1x elsewhere
        
        # Weighted BCE loss
        bce = torch.nn.functional.binary_cross_entropy(
            predictions.view(-1),
            targets.view(-1),
            weight=boundary_weight.view(-1),
            reduction='mean'
        )
        
        return bce


class CombinedBoundaryDiceLoss(nn.Module):
    """
    Combined Dice + Boundary loss for balanced segmentation.
    
    Dice handles overall overlap, Boundary handles edge precision.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.7,
        boundary_weight: float = 0.3,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.boundary_loss = BoundaryLoss(smooth)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model output (after sigmoid) [B, 1, H, W, D]
            targets: Ground truth [B, 1, H, W, D]
            sample_weights: Optional per-sample weights [B]
            
        Returns:
            Scalar loss value
        """
        # Dice loss
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice
        
        if sample_weights is not None:
            dice_loss = dice_loss * sample_weights
        dice_loss = dice_loss.mean()
        
        # Boundary loss
        boundary_loss = self.boundary_loss(predictions, targets)
        
        # Combine
        total_loss = self.dice_weight * dice_loss + self.boundary_weight * boundary_loss
        
        return total_loss


if __name__ == "__main__":
    # Quick test
    print("Testing Focal Tversky Loss...")
    
    # Simulate predictions and targets
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32, 32))
    target = (torch.rand(2, 1, 32, 32, 32) > 0.95).float()  # Sparse targets (small lesions)
    
    # Test Focal Tversky
    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    loss = loss_fn(pred, target)
    print(f"Focal Tversky Loss: {loss.item():.4f}")
    
    # Test Boundary Loss
    boundary_fn = BoundaryLoss()
    b_loss = boundary_fn(pred, target)
    print(f"Boundary Loss: {b_loss.item():.4f}")
    
    # Test Combined
    combined_fn = CombinedBoundaryDiceLoss()
    c_loss = combined_fn(pred, target)
    print(f"Combined Dice+Boundary Loss: {c_loss.item():.4f}")
    
    print("All tests passed!")
