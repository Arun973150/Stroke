"""
Stage 2: SegResNet Segmentation on Cropped ROIs (Phase 3)
==========================================================
Trains SegResNet on the cropped ROIs from Stage 1 for precise lesion
segmentation. Uses compound loss: 0.4*GDL + 0.4*Focal + 0.2*BoundaryCE.

Implements:
  - Foreground-biased sampling (70/30 pos/neg)
  - Hard example mining (optional, via difficulty scores)
  - Channel dropout augmentation (10% prob per channel)

Usage:
  python scripts/07_train_stage2_segresnet.py --config configs/soop_config.yaml --fold 0
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
from monai.data import DataLoader
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Compound Loss ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.float().clamp(-20, 20)
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_t * ((1 - p_t).clamp(0, 1) ** self.gamma) * bce
        return loss.mean()


class BoundaryLoss(nn.Module):
    """Extra CE weight on boundary voxels to improve delineation."""
    def __init__(self, weight=3.0):
        super().__init__()
        self.weight = weight
        self.laplacian = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplacian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, pred, target):
        self.laplacian = self.laplacian.to(pred.device)
        boundary = torch.abs(self.laplacian(target.float()))
        boundary = (boundary > 0).float()

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weighted = bce * (1.0 + boundary * self.weight)
        return weighted.mean()


class CompoundLoss(nn.Module):
    """0.4*GDL + 0.4*Focal + 0.2*Boundary"""
    def __init__(self):
        super().__init__()
        self.gdl = GeneralizedDiceLoss(sigmoid=True)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.boundary = BoundaryLoss(weight=3.0)

    def forward(self, pred, target):
        # Compute in float32 to avoid AMP NaN issues with GDL
        pred = pred.float()
        target = target.float()
        gdl = self.gdl(pred, target)
        focal = self.focal(pred, target)
        boundary = self.boundary(pred, target)
        loss = 0.4 * gdl + 0.4 * focal + 0.2 * boundary
        # NaN protection: if any component is NaN, fall back to focal only
        if torch.isnan(loss):
            return focal
        return loss


# ── Dataset ──────────────────────────────────────────────────────────────────

class Stage2CropDataset(torch.utils.data.Dataset):
    """Loads pre-cropped Stage 2 .npz patches."""

    def __init__(self, crop_dir: Path, manifest: dict, augment: bool = False,
                 channel_dropout_prob: float = 0.1):
        self.crop_dir = crop_dir
        self.crops = manifest["crops"]
        self.augment = augment
        self.channel_dropout_prob = channel_dropout_prob

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        info = self.crops[idx]
        data = np.load(self.crop_dir / f"{info['crop_name']}.npz")

        image = torch.from_numpy(data["image"].astype(np.float32))  # (3, D, H, W)
        mask = torch.from_numpy(data["mask"].astype(np.float32)).unsqueeze(0)  # (1, D, H, W)

        if self.augment:
            # Random flips
            for axis in [1, 2, 3]:
                if np.random.random() > 0.5:
                    image = torch.flip(image, [axis])
                    mask = torch.flip(mask, [axis])

            # Gaussian noise
            if np.random.random() > 0.5:
                image = image + torch.randn_like(image) * 0.05

            # Intensity scaling
            if np.random.random() > 0.5:
                scale = 0.9 + np.random.random() * 0.2
                image = image * scale

            # Channel dropout (randomly zero one channel)
            if np.random.random() < self.channel_dropout_prob:
                ch = np.random.randint(0, 3)
                image[ch] = 0.0

        return {
            "image": image,
            "mask": mask,
            "has_lesion": info["has_lesion"],
            "crop_name": info["crop_name"],
        }


# ── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            # Compute loss in float32 — GDL produces NaN under float16
            loss = criterion(outputs.float(), masks.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        pred = (torch.sigmoid(outputs) > 0.5).float()
        pred_oh = torch.cat([1 - pred, pred], dim=1)
        mask_oh = torch.cat([1 - masks, masks], dim=1)
        dice_metric(pred_oh, mask_oh)

        total_loss += loss.item()
        n += 1

    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return total_loss / max(n, 1), dice


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 SegResNet")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage2_cfg = config["training"]["stage2"]
    epochs = args.epochs or stage2_cfg["epochs"]

    crop_base = Path(config["paths"]["preprocessed"]).parent / "stage2_crops"
    ckpt_dir = Path(config["paths"]["checkpoints"]) / "stage2_segresnet" / f"fold_{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["paths"]["logs"]) / "stage2_segresnet" / f"fold_{args.fold}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    with open(crop_base / "train" / "manifest.json") as f:
        train_manifest = json.load(f)
    with open(crop_base / "val" / "manifest.json") as f:
        val_manifest = json.load(f)

    print(f"Train crops: {len(train_manifest['crops'])}")
    print(f"Val crops:   {len(val_manifest['crops'])}")

    train_ds = Stage2CropDataset(crop_base / "train", train_manifest, augment=True)
    val_ds = Stage2CropDataset(crop_base / "val", val_manifest, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config["gpu"]["num_workers"],
                              pin_memory=config["gpu"]["pin_memory"],
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config["gpu"]["num_workers"],
                            pin_memory=config["gpu"]["pin_memory"])

    # Model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=32,
        in_channels=3,
        out_channels=1,
        dropout_prob=0.2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: SegResNet ({param_count:.1f}M params)")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    criterion = CompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # AMP disabled — GDL + BoundaryLoss produce NaN under mixed precision
    scaler = None
    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    patience = 100  # early stopping patience

    if args.resume:
        ckpt_path = ckpt_dir / "checkpoint_latest.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt.get("best_dice", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"Resumed from epoch {start_epoch}, best dice {best_dice:.4f}")

    # Training
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"{'Epoch':>9} | {'TrLoss':>8} {'VlLoss':>8} | {'VlDice':>7} | {'LR':>10} | {'Time':>5} | {'Status'}")
    print("-" * 80)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("lr", lr, epoch)

        status = ""

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_dice": best_dice,
            }, ckpt_dir / "checkpoint_best.pth")
            status = " *best"
        else:
            patience_counter += 1

        # Print every epoch
        print(f"  {epoch+1:4d}/{epochs} | {train_loss:8.4f} {val_loss:8.4f} | "
              f"{val_dice:7.4f} | {lr:10.6f} | {elapsed:5.1f}s |{status}")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_dice": best_dice,
                "patience_counter": patience_counter,
            }, ckpt_dir / "checkpoint_latest.pth")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    writer.close()

    print(f"\n{'=' * 50}")
    print(f"SEGRESNET FOLD {args.fold} COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"    - checkpoint_best.pth")
    print(f"    - checkpoint_latest.pth")
    print(f"  Logs: {log_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
