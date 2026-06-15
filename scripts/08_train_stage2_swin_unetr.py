"""
Stage 2: Swin-UNETR Segmentation on Cropped ROIs (Phase 3)
============================================================
Trains Swin-UNETR (sensitivity-focused) on cropped ROIs.
Same data pipeline as SegResNet but with transformer architecture.

Usage:
  python scripts/08_train_stage2_swin_unetr.py --config configs/soop_config.yaml --fold 0
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
from monai.networks.nets import SwinUNETR
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# Reuse dataset and losses from segresnet script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_stage2_segresnet_imports import CompoundLoss, Stage2CropDataset
except (ImportError, ModuleNotFoundError):
    # Inline the compound loss
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
            return (alpha_t * ((1 - p_t).clamp(0, 1) ** self.gamma) * bce).mean()

    class CompoundLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.gdl = GeneralizedDiceLoss(sigmoid=True)
            self.focal = FocalLoss()

        def forward(self, pred, target):
            pred = pred.float()
            target = target.float()
            gdl = self.gdl(pred, target)
            focal = self.focal(pred, target)
            loss = 0.5 * gdl + 0.5 * focal
            if torch.isnan(loss):
                return focal
            return loss

    class Stage2CropDataset(torch.utils.data.Dataset):
        def __init__(self, crop_dir, manifest, augment=False, channel_dropout_prob=0.1):
            self.crop_dir = crop_dir
            self.crops = manifest["crops"]
            self.augment = augment
            self.channel_dropout_prob = channel_dropout_prob

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            info = self.crops[idx]
            data = np.load(self.crop_dir / f"{info['crop_name']}.npz")
            image = torch.from_numpy(data["image"].astype(np.float32))
            mask = torch.from_numpy(data["mask"].astype(np.float32)).unsqueeze(0)
            if self.augment:
                for axis in [1, 2, 3]:
                    if np.random.random() > 0.5:
                        image = torch.flip(image, [axis])
                        mask = torch.flip(mask, [axis])
                if np.random.random() > 0.5:
                    image = image + torch.randn_like(image) * 0.05
                if np.random.random() < self.channel_dropout_prob:
                    image[np.random.randint(0, 3)] = 0.0
            return {"image": image, "mask": mask, "has_lesion": info["has_lesion"],
                    "crop_name": info["crop_name"]}


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        if scaler:
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


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Swin-UNETR")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage2_cfg = config["training"]["stage2"]
    epochs = args.epochs or stage2_cfg["epochs"]

    crop_base = Path(config["paths"]["preprocessed"]).parent / "stage2_crops"
    ckpt_dir = Path(config["paths"]["checkpoints"]) / "stage2_swin_unetr" / f"fold_{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["paths"]["logs"]) / "stage2_swin_unetr" / f"fold_{args.fold}"
    log_dir.mkdir(parents=True, exist_ok=True)

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
                              pin_memory=config["gpu"]["pin_memory"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config["gpu"]["num_workers"],
                            pin_memory=config["gpu"]["pin_memory"])

    # Swin-UNETR: input must be divisible by patch_size * 2^(num_layers)
    # With 128^3 input and patch_size=(4,4,4), this works perfectly
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=3,
        out_channels=1,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_v2=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: Swin-UNETR ({param_count:.1f}M params)")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    criterion = CompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # AMP disabled — GDL + CompoundLoss produce NaN under mixed precision
    scaler = None
    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    patience = 100

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

    print(f"\nStarting training...")
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

        print(f"  {epoch+1:4d}/{epochs} | {train_loss:8.4f} {val_loss:8.4f} | "
              f"{val_dice:7.4f} | {lr:10.6f} | {elapsed:5.1f}s |{status}")

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

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    writer.close()
    print(f"\n{'=' * 50}")
    print(f"SWIN-UNETR FOLD {args.fold} COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"    - checkpoint_best.pth")
    print(f"    - checkpoint_latest.pth")
    print(f"  Logs: {log_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
