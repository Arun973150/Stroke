"""
Stage 1: Lesion Detection Network (Phase 2)
=============================================
Trains a lightweight 3D SegResNet at low resolution (2mm isotropic)
to detect approximate lesion locations with high recall.

Input:  Downsampled full-brain (96x96x96), 3 channels (TRACE+ADC+FLAIR)
Output: Coarse probability heatmap -> bounding boxes around detected regions

Loss: Focal Loss (alpha=0.9) optimized for recall, NOT Dice.
Target: >95% lesion detection rate at case level.

Usage:
  python scripts/04_train_stage1_detection.py --config configs/soop_config.yaml
  python scripts/04_train_stage1_detection.py --config configs/soop_config.yaml --fold 0
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, RandAffined, RandGaussianNoised,
    RandFlipd, RandScaleIntensityd, ToTensord,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss with high alpha to prioritize recall on minority (lesion) class."""

    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float().clamp(-20, 20)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = (1 - p_t).clamp(0, 1) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ── Data Loading ─────────────────────────────────────────────────────────────

class Stage1Dataset(torch.utils.data.Dataset):
    """Loads preprocessed .npz and downsamples to low resolution for detection."""

    def __init__(self, subject_ids: list, preproc_dir: Path,
                 input_shape: tuple = (96, 96, 96), augment: bool = False):
        self.subject_ids = subject_ids
        self.preproc_dir = preproc_dir
        self.input_shape = input_shape
        self.augment = augment

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = np.load(self.preproc_dir / f"{sid}.npz")

        image = data["image"].astype(np.float32)   # (3, D, H, W)
        mask = data["mask"].astype(np.float32)      # (D, H, W)

        # Downsample to input_shape using trilinear for image, nearest for mask
        image_t = torch.from_numpy(image).unsqueeze(0)  # (1, 3, D, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        image_t = F.interpolate(image_t, size=self.input_shape, mode="trilinear",
                                align_corners=False).squeeze(0)  # (3, d, h, w)
        mask_t = F.interpolate(mask_t, size=self.input_shape, mode="nearest"
                               ).squeeze(0)  # (1, d, h, w)

        # Simple augmentation
        if self.augment:
            # Random flip
            for axis in [1, 2, 3]:
                if np.random.random() > 0.5:
                    image_t = torch.flip(image_t, [axis])
                    mask_t = torch.flip(mask_t, [axis])

            # Random noise
            if np.random.random() > 0.5:
                noise = torch.randn_like(image_t) * 0.05
                image_t = image_t + noise

            # Random intensity scale
            if np.random.random() > 0.5:
                scale = 0.9 + np.random.random() * 0.2
                image_t = image_t * scale

        return {
            "image": image_t,
            "mask": mask_t,
            "subject_id": sid,
            "has_lesion": float(mask.sum() > 0),
        }


# ── Training ─────────────────────────────────────────────────────────────────

def compute_detection_recall(preds: torch.Tensor, targets: torch.Tensor,
                             threshold: float = 0.2) -> float:
    """Case-level detection recall: fraction of lesion cases where any voxel > threshold."""
    batch_size = preds.shape[0]
    detected = 0
    total_positive = 0

    for i in range(batch_size):
        has_lesion = targets[i].sum() > 0
        if has_lesion:
            total_positive += 1
            pred_binary = (torch.sigmoid(preds[i]) > threshold).float()
            if pred_binary.sum() > 0:
                detected += 1

    return detected / max(total_positive, 1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_recall = 0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recall += compute_detection_recall(outputs.detach(), masks)
        n_batches += 1

    return total_loss / n_batches, total_recall / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.2):
    model.eval()
    total_loss = 0
    total_recall = 0
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_recall += compute_detection_recall(outputs, masks, threshold)

        # Dice — use 2-channel one-hot for proper computation
        pred_binary = (torch.sigmoid(outputs) > 0.5).float()
        pred_oh = torch.cat([1 - pred_binary, pred_binary], dim=1)
        mask_oh = torch.cat([1 - masks, masks], dim=1)
        dice_metric(pred_oh, mask_oh)
        n_batches += 1

    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return total_loss / n_batches, total_recall / n_batches, dice


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 detection model")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (optional)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stage1_cfg = config["training"]["stage1"]
    input_shape = tuple(stage1_cfg["input_shape"])
    epochs = args.epochs or stage1_cfg["epochs"]
    batch_size = stage1_cfg["batch_size"]
    lr = stage1_cfg["lr"]
    focal_alpha = stage1_cfg["focal_alpha"]
    det_threshold = stage1_cfg["detection_threshold"]

    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"]) / "stage1_detection"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["paths"]["logs"]) / "stage1_detection"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(splits_dir / "train.json") as f:
        train_ids = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_ids = json.load(f)

    print(f"Train subjects: {len(train_ids)}")
    print(f"Val subjects:   {len(val_ids)}")
    print(f"Input shape:    {input_shape}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Focal alpha:    {focal_alpha}")

    # Datasets
    train_ds = Stage1Dataset(train_ids, preproc_dir, input_shape, augment=True)
    val_ds = Stage1Dataset(val_ids, preproc_dir, input_shape, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=config["gpu"]["num_workers"],
                              pin_memory=config["gpu"]["pin_memory"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=config["gpu"]["num_workers"],
                            pin_memory=config["gpu"]["pin_memory"])

    # Model: lightweight SegResNet
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=3,
        out_channels=1,
        dropout_prob=0.2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir))

    # Resume
    start_epoch = 0
    best_recall = 0.0
    best_dice = 0.0
    patience_counter = 0
    patience = 50  # early stopping patience
    if args.resume:
        ckpt_path = ckpt_dir / "checkpoint_latest.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_recall = ckpt.get("best_recall", 0.0)
            best_dice = ckpt.get("best_dice", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"Resumed from epoch {start_epoch}, best_recall={best_recall:.3f}, best_dice={best_dice:.3f}")

    # Training loop
    print(f"\nStarting training...")
    print(f"{'Epoch':>7} | {'TrLoss':>8} {'VlLoss':>8} | {'TrRec':>6} {'VlRec':>6} | {'VlDice':>7} | {'LR':>10} | {'Time':>5} | {'Status'}")
    print("-" * 95)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_recall = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_recall, val_dice = validate(model, val_loader, criterion, device, det_threshold)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Tensorboard logging
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("recall", {"train": train_recall, "val": val_recall}, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("lr", lr, epoch)

        # Determine save status
        status = ""

        # Save best by recall
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_recall": best_recall,
                "best_dice": best_dice,
                "val_dice": val_dice,
                "config": stage1_cfg,
            }, ckpt_dir / "checkpoint_best_recall.pth")
            status += " *best_recall"

        # Save best by dice
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_recall": best_recall,
                "best_dice": best_dice,
                "val_dice": val_dice,
                "config": stage1_cfg,
            }, ckpt_dir / "checkpoint_best_dice.pth")
            status += " *best_dice"
        else:
            patience_counter += 1

        # Print every epoch
        print(f"  {epoch+1:3d}/{epochs} | {train_loss:8.4f} {val_loss:8.4f} | "
              f"{train_recall:6.3f} {val_recall:6.3f} | {val_dice:7.4f} | "
              f"{lr:10.6f} | {elapsed:5.1f}s |{status}")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_recall": best_recall,
                "best_dice": best_dice,
                "patience_counter": patience_counter,
                "config": stage1_cfg,
            }, ckpt_dir / "checkpoint_latest.pth")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no dice improvement for {patience} epochs)")
            break

    writer.close()

    print(f"\n{'=' * 50}")
    print("STAGE 1 TRAINING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Best validation recall: {best_recall:.3f}")
    print(f"  Best validation dice:   {best_dice:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"    - checkpoint_best_recall.pth")
    print(f"    - checkpoint_best_dice.pth")
    print(f"    - checkpoint_latest.pth")
    print(f"  Logs: {log_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
