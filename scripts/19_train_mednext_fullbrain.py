"""
Path B: MedNeXt Training on Full-Brain Volumes (Dataset003)
=============================================================
Trains MedNeXt (Medical NeXt) on full-brain volumes for standalone
stroke lesion segmentation (Path B safety net).

MedNeXt uses ConvNeXt-style blocks adapted for 3D medical imaging,
with proven +2-4% Dice over standard U-Net architectures.

Architecture: MedNeXt-M (medium) with 3D kernels
Input: 3-channel (TRACE+ADC+FLAIR) at 1mm isotropic
Output: Binary lesion segmentation

Uses MONAI's sliding window inference during validation for full-brain volumes.

Usage:
  python scripts/19_train_mednext_fullbrain.py --config configs/soop_config.yaml --fold 0
  python scripts/19_train_mednext_fullbrain.py --config configs/soop_config.yaml --fold 1
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── MedNeXt Architecture ───────────────────────────────────────────────────

class MedNeXtBlock(nn.Module):
    """ConvNeXt-style block adapted for 3D medical imaging.

    Depthwise conv → LayerNorm → 1x1 conv (expand) → GELU → 1x1 conv (project)
    """
    def __init__(self, channels, expansion=4, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        expanded = channels * expansion

        self.dwconv = nn.Conv3d(channels, channels, kernel_size, padding=pad, groups=channels, bias=True)
        self.norm = nn.GroupNorm(1, channels)  # LayerNorm equivalent for conv
        self.pwconv1 = nn.Conv3d(channels, expanded, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(expanded, channels, 1)

        # Learnable scale (from ConvNeXt)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x


class MedNeXtDown(nn.Module):
    """Downsampling block: strided conv + MedNeXt blocks."""
    def __init__(self, in_ch, out_ch, n_blocks=2, kernel_size=3):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(*[MedNeXtBlock(out_ch, kernel_size=kernel_size) for _ in range(n_blocks)])

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class MedNeXtUp(nn.Module):
    """Upsampling block: transposed conv + concat skip + MedNeXt blocks."""
    def __init__(self, in_ch, skip_ch, out_ch, n_blocks=2, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv3d(in_ch + skip_ch, out_ch, 1)
        self.blocks = nn.Sequential(*[MedNeXtBlock(out_ch, kernel_size=kernel_size) for _ in range(n_blocks)])

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1x1(x)
        x = self.blocks(x)
        return x


class MedNeXt(nn.Module):
    """MedNeXt-M: U-Net with ConvNeXt-style blocks for 3D medical segmentation.

    Encoder: [32, 64, 128, 256, 512]
    Decoder mirrors encoder with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512),
                 n_blocks=(2, 2, 2, 2, 2), kernel_size=3):
        super().__init__()
        self.features = features

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], 3, padding=1, bias=False),
            nn.GroupNorm(1, features[0]),
            nn.GELU(),
        )
        self.stem_blocks = nn.Sequential(*[MedNeXtBlock(features[0], kernel_size=kernel_size) for _ in range(n_blocks[0])])

        # Encoder
        self.enc1 = MedNeXtDown(features[0], features[1], n_blocks[1], kernel_size)
        self.enc2 = MedNeXtDown(features[1], features[2], n_blocks[2], kernel_size)
        self.enc3 = MedNeXtDown(features[2], features[3], n_blocks[3], kernel_size)

        # Bottleneck
        self.bottleneck = MedNeXtDown(features[3], features[4], n_blocks[4], kernel_size)

        # Decoder
        self.dec3 = MedNeXtUp(features[4], features[3], features[3], n_blocks[3], kernel_size)
        self.dec2 = MedNeXtUp(features[3], features[2], features[2], n_blocks[2], kernel_size)
        self.dec1 = MedNeXtUp(features[2], features[1], features[1], n_blocks[1], kernel_size)
        self.dec0 = MedNeXtUp(features[1], features[0], features[0], n_blocks[0], kernel_size)

        # Output
        self.head = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        # Encoder
        s0 = self.stem_blocks(self.stem(x))
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        # Bottleneck
        b = self.bottleneck(s3)

        # Decoder
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        return self.head(d0)


# ── Loss ────────────────────────────────────────────────────────────────────

class CompoundLoss(nn.Module):
    """Tversky + Focal Tversky loss, sensitivity-focused.

    alpha=0.7 means false negatives penalized 2.3x more than false positives.
    All computation forced to float32 to avoid AMP nan issues.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    @torch.cuda.amp.custom_fwd(cast_to=torch.float32)
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        tp = (pred_flat * target_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        fp = ((1 - target_flat) * pred_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = (1 - tversky).clamp(min=0)
        focal_tversky_loss = tversky_loss ** self.gamma

        return 0.5 * tversky_loss + 0.5 * focal_tversky_loss


# ── Dataset ─────────────────────────────────────────────────────────────────

class FullBrainDataset(torch.utils.data.Dataset):
    """Loads full-brain preprocessed .npz files with random crop for training."""

    def __init__(self, subject_ids, preproc_dir, patch_size=(128, 128, 128),
                 augment=False, ensure_lesion_ratio=0.7):
        self.subject_ids = subject_ids
        self.preproc_dir = Path(preproc_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.ensure_lesion_ratio = ensure_lesion_ratio

    def __len__(self):
        return len(self.subject_ids)

    def _random_crop(self, image, mask):
        """Random crop, with bias toward lesion-containing regions."""
        d, h, w = image.shape[1:]
        pd, ph, pw = self.patch_size

        # Ensure volume is large enough
        if d < pd or h < ph or w < pw:
            # Pad if needed
            pad_d = max(pd - d, 0)
            pad_h = max(ph - h, 0)
            pad_w = max(pw - w, 0)
            image = np.pad(image, [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w)], mode="constant")
            mask = np.pad(mask, [(0, pad_d), (0, pad_h), (0, pad_w)], mode="constant")
            d, h, w = image.shape[1:]

        # Decide: crop around lesion or random
        has_lesion = mask.sum() > 0
        crop_on_lesion = has_lesion and np.random.random() < self.ensure_lesion_ratio

        if crop_on_lesion:
            # Find lesion center and crop around it with some jitter
            coords = np.where(mask > 0)
            center = [int(np.mean(c)) for c in coords]
            jitter = [np.random.randint(-pd // 4, pd // 4 + 1) for _ in range(3)]
            start = [
                np.clip(center[i] + jitter[i] - self.patch_size[i] // 2, 0, max(0, [d, h, w][i] - self.patch_size[i]))
                for i in range(3)
            ]
        else:
            start = [
                np.random.randint(0, max(1, s - p + 1))
                for s, p in zip([d, h, w], self.patch_size)
            ]

        sd, sh, sw = start
        image_crop = image[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
        mask_crop = mask[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return image_crop, mask_crop

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = np.load(self.preproc_dir / f"{sid}.npz")
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        image_crop, mask_crop = self._random_crop(image, mask)

        image_t = torch.from_numpy(image_crop.copy())
        mask_t = torch.from_numpy(mask_crop.copy()).unsqueeze(0)

        if self.augment:
            for axis in [1, 2, 3]:
                if np.random.random() > 0.5:
                    image_t = torch.flip(image_t, [axis])
                    mask_t = torch.flip(mask_t, [axis])
            if np.random.random() > 0.5:
                image_t = image_t + torch.randn_like(image_t) * 0.05
            if np.random.random() > 0.5:
                image_t = image_t * (0.9 + np.random.random() * 0.2)

        return {"image": image_t, "mask": mask_t, "subject_id": sid}


# ── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch, total_epochs):
    model.train()
    total_loss = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(images)
        n += len(images)
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / n


def validate(model, loader, criterion, device):
    """Fast validation using random crops (no sliding window)."""
    model.eval()
    total_loss = 0
    dice_sum = 0
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)

            total_loss += loss.item() * len(images)

            # Dice
            pred = (torch.sigmoid(logits) > 0.5).float()
            for i in range(len(images)):
                p = pred[i].flatten()
                g = masks[i].flatten()
                inter = (p * g).sum()
                total_vox = p.sum() + g.sum()
                dice = (2 * inter / total_vox).item() if total_vox > 0 else 1.0
                dice_sum += dice
                n += 1

    return total_loss / max(n, 1), dice_sum / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MedNeXt on full-brain volumes (Path B)")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"]) / "pathB_mednext" / f"fold_{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(splits_dir / "train.json") as f:
        all_train_ids = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_ids = json.load(f)

    # Create fold split from training data
    np.random.seed(42 + args.fold)
    indices = np.random.permutation(len(all_train_ids))
    fold_size = len(all_train_ids) // 5
    val_start = args.fold * fold_size
    val_end = val_start + fold_size

    fold_val_indices = indices[val_start:val_end]
    fold_train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

    train_ids = [all_train_ids[i] for i in fold_train_indices]
    fold_val_ids = [all_train_ids[i] for i in fold_val_indices]

    patch_size = tuple(args.patch_size)

    print(f"MedNeXt Full-Brain Training (Path B)")
    print(f"  Fold: {args.fold}")
    print(f"  Train: {len(train_ids)}, Fold Val: {len(fold_val_ids)}, Final Val: {len(val_ids)}")
    print(f"  Patch size: {patch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {device}")

    # Model
    model = MedNeXt(
        in_channels=3, out_channels=1,
        features=(32, 64, 128, 256, 512),
        n_blocks=(2, 2, 2, 2, 2),
        kernel_size=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Datasets
    train_ds = FullBrainDataset(train_ids, preproc_dir, patch_size=patch_size, augment=True)
    val_ds = FullBrainDataset(fold_val_ids, preproc_dir, patch_size=patch_size, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=3,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=3,
    )

    # Loss, optimizer, scheduler
    criterion = CompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # TensorBoard
    log_dir = Path(config["paths"]["logs"]) / f"pathB_mednext_fold{args.fold}"
    writer = SummaryWriter(log_dir)

    best_dice = 0.0
    best_epoch = 0
    patience = 50  # stop if no improvement for 50 validated epochs (= 500 actual epochs)
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, args.epochs)
        scheduler.step()

        # Validate every 10 epochs
        val_loss, val_dice = 0.0, 0.0
        if epoch % 10 == 0 or epoch == 1:
            val_loss, val_dice = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} Dice: {val_dice:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s | "
                  f"Best: {best_dice:.4f} (ep{best_epoch}) patience: {patience - no_improve}")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Dice/val", val_dice, epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                }, ckpt_dir / "checkpoint_best.pth")
                print(f"  *** New best Dice: {best_dice:.4f} ***")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} — no improvement for {patience} validations")
                break

        # Save latest every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
            }, ckpt_dir / "checkpoint_latest.pth")

    print(f"\n{'=' * 60}")
    print(f"MEDNEXT TRAINING COMPLETE (Path B)")
    print(f"{'=' * 60}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Dice:  {best_dice:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'checkpoint_best.pth'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
