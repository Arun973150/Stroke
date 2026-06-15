#!/usr/bin/env python3
"""
11. SegResNet Training Script
==============================
Train SegResNet model for stroke lesion segmentation.

This script:
1. Loads ISLES data from nnU-Net preprocessed format
2. Trains SegResNet with stability-focused configuration
3. Uses DiceCE loss with higher Dice weight for precision
4. Uses SGD + momentum with step LR decay
5. Performs sliding window validation
6. Saves best and periodic checkpoints

Usage:
    python scripts/11_train_segresnet.py --config configs/segresnet_config.yaml --fold 0
    
For debugging (5 epochs):
    python scripts/11_train_segresnet.py --config configs/segresnet_config.yaml --fold 0 --epochs 5
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
from tqdm import tqdm

# MONAI imports
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segresnet import create_segresnet, SegResNetWrapper
from src.data.isles_dataset import get_train_val_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    print_header, print_step, print_success, print_warning, print_error, print_info
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict,
) -> float:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    step = 0
    
    gradient_clip = config.get('training', {}).get('gradient_clip', 1.0)
    use_amp = config.get('training', {}).get('use_amp', True)
    
    # Clear cache before training to maximize available memory
    torch.cuda.empty_cache()
    
    # Use non-persistent progress bar for cleaner logs
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, disable=not sys.stdout.isatty())
    for batch_data in train_pbar:
        step += 1
        
        # Move data to device
        inputs = batch_data['image'].to(device)
        labels = batch_data['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        train_pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / step


def validate(
    model: nn.Module,
    val_loader,
    dice_metric: DiceMetric,
    post_pred,
    post_label,
    device: torch.device,
    config: dict,
) -> float:
    """Validate model using sliding window inference."""
    model.eval()
    
    roi_size = tuple(config.get('inference', {}).get('roi_size', [96, 96, 96]))
    sw_batch_size = config.get('inference', {}).get('sw_batch_size', 2)
    overlap = config.get('inference', {}).get('overlap', 0.5)
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", leave=False, disable=sys.stdout.isatty() is False)
        for batch_data in val_pbar:
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            
            # Sliding window inference
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode='gaussian',
            )
            
            # Post-process predictions
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            labels = [post_label(i) for i in decollate_batch(labels)]
            
            # Compute Dice metric
            dice_metric(y_pred=outputs, y=labels)
    
    # Aggregate metric
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return mean_dice


def train_fold(
    config: dict,
    fold: int = 0,
    max_epochs: int = None,
    resume: str = None,
) -> dict:
    """
    Train a single fold of SegResNet.
    
    Args:
        config: Configuration dictionary
        fold: Fold number (0-4)
        max_epochs: Override max epochs (useful for debugging)
        resume: Path to checkpoint to resume from
    
    Returns:
        Dictionary with training results
    """
    # Setup device
    device = torch.device(f"cuda:{config['gpu']['device_id']}" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    # Get training config
    train_config = config.get('training', {})
    num_epochs = max_epochs or train_config.get('num_epochs', 500)
    batch_size = train_config.get('batch_size', 2)
    val_batch_size = train_config.get('val_batch_size', 1)
    learning_rate = train_config.get('learning_rate', 0.01)
    momentum = train_config.get('momentum', 0.9)
    weight_decay = train_config.get('weight_decay', 1e-4)
    val_interval = train_config.get('val_interval', 10)
    
    # Create model
    print_info("Creating SegResNet model...")
    model = create_segresnet(config)
    model = model.to(device)
    
    # Create transforms
    patch_size = tuple(config.get('dataset', {}).get('patch_size', [128, 128, 128]))
    samples_per_volume = config.get('dataset', {}).get('samples_per_volume', 4)
    
    train_transforms = get_train_transforms(
        patch_size=patch_size,
        samples_per_volume=samples_per_volume,
        config=config,
    )
    val_transforms = get_val_transforms(patch_size=patch_size)
    
    # Create dataloaders
    print_info("Loading data...")
    data_dir = config['paths']['nnunet_preprocessed']
    
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        fold=fold,
        train_transform=train_transforms,
        val_transform=val_transforms,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=config['gpu']['num_workers'],
        pin_memory=config['gpu']['pin_memory'],
    )
    
    # Loss function - weighted DiceCE for precision focus
    dice_weight = train_config.get('dice_weight', 0.6)
    ce_weight = train_config.get('ce_weight', 0.4)
    
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        lambda_dice=dice_weight,
        lambda_ce=ce_weight,
    )
    
    # Optimizer - SGD with momentum (more stable for residual networks)
    optimizer_type = train_config.get('optimizer', 'sgd').lower()
    
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    
    # Learning rate scheduler - Step decay
    scheduler_type = train_config.get('scheduler', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = train_config.get('step_size', 150)
        gamma = train_config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    else:
        # Cosine annealing fallback
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=train_config.get('use_amp', True))
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    # Tensorboard
    log_dir = Path(config['paths']['logs']) / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Checkpoint directory
    checkpoint_dir = Path(config['paths']['checkpoints']) / f"fold_{fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    
    if resume:
        print_info(f"Resuming from checkpoint: {resume}")
        model, checkpoint = SegResNetWrapper.load_checkpoint(resume, device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
    
    # Early stopping
    early_stop_config = train_config.get('early_stopping', {})
    early_stop_enabled = early_stop_config.get('enabled', True)
    patience = early_stop_config.get('patience', 80)
    min_delta = early_stop_config.get('min_delta', 0.001)
    epochs_without_improvement = 0
    
    # Training loop
    print_header(f"Training SegResNet - Fold {fold}")
    print_info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print_info(f"Optimizer: {optimizer_type.upper()}, Scheduler: {scheduler_type}")
    
    train_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch, config
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log training metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)
        
        # Validate
        if (epoch + 1) % val_interval == 0:
            val_dice = validate(
                model, val_loader, dice_metric, post_pred, post_label, device, config
            )
            
            writer.add_scalar('val/dice', val_dice, epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_dice > best_metric + min_delta:
                best_metric = val_dice
                epochs_without_improvement = 0
                model.save_checkpoint(
                    checkpoint_dir / "checkpoint_best.pth",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                )
                print_success(f"  New best Dice: {best_metric:.4f}")
            else:
                epochs_without_improvement += val_interval
            
            # Early stopping check
            if early_stop_enabled and epochs_without_improvement >= patience:
                print_warning(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s", flush=True)
        
        # Save checkpoint periodically
        save_interval = config.get('monitoring', {}).get('save_interval', 50)
        if (epoch + 1) % save_interval == 0:
            model.save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
            )
    
    # Save final checkpoint
    model.save_checkpoint(
        checkpoint_dir / "checkpoint_final.pth",
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_metric=best_metric,
    )
    
    total_time = time.time() - train_start_time
    writer.close()
    
    print_header("Training Complete!")
    print_success(f"Best Dice: {best_metric:.4f}")
    print_info(f"Total time: {total_time / 3600:.2f} hours")
    print_info(f"Checkpoints saved to: {checkpoint_dir}")
    
    return {
        'best_dice': best_metric,
        'total_time': total_time,
        'checkpoint_dir': str(checkpoint_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SegResNet for stroke segmentation")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--epochs', type=int, default=None, help='Override max epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    results = train_fold(
        config=config,
        fold=args.fold,
        max_epochs=args.epochs,
        resume=args.resume,
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
