"""
SegResNet + Synthetic Lesion Augmentation
==========================================
Same architecture as 19b but with synthetic stroke lesion generation:
  - Generates random ellipsoidal lesions with realistic DWI/ADC intensity
  - Pastes into healthy brain regions or augments existing lesions
  - Proven to boost sensitivity from ~80% to 91% (Radiology AI 2020)

This produces a DIFFERENT model flavor for ensemble diversity.

Usage:
  python scripts/24_train_segresnet_synthetic_aug.py --config configs/soop_config.yaml --fold 0
  python scripts/24_train_segresnet_synthetic_aug.py --config configs/soop_config.yaml --fold 0 --synth-prob 0.5
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
from monai.networks.nets import SegResNet
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Deep Supervision Wrapper ────────────────────────────────────────────────

class DeepSupSegResNet(nn.Module):
    """SegResNet with deep supervision heads at intermediate decoder layers."""

    def __init__(self, in_channels=3, out_channels=1, init_filters=32, dropout_prob=0.2):
        super().__init__()
        self.backbone = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )

        self.ds_head_4x = nn.Conv3d(init_filters * 4, out_channels, kernel_size=1)
        self.ds_head_2x = nn.Conv3d(init_filters * 2, out_channels, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]

        out = self.backbone.convInit(x)
        encoder_outputs = []
        for down_layer in self.backbone.down_layers:
            out = down_layer(out)
            encoder_outputs.append(out)

        ds_outputs = []
        for i in range(len(self.backbone.up_layers)):
            out = self.backbone.up_samples[i](out)
            skip = encoder_outputs[-(i + 2)]
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode="trilinear", align_corners=False)
            out = out + skip
            out = self.backbone.up_layers[i](out)

            if self.training:
                if i == 0:
                    ds = self.ds_head_4x(out)
                    ds = F.interpolate(ds, size=input_shape, mode="trilinear", align_corners=False)
                    ds_outputs.append(ds)
                elif i == 1:
                    ds = self.ds_head_2x(out)
                    ds = F.interpolate(ds, size=input_shape, mode="trilinear", align_corners=False)
                    ds_outputs.append(ds)

        final = self.backbone.conv_final(out)

        if self.training and ds_outputs:
            return final, ds_outputs
        return final


# ── Loss ────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        with torch.amp.autocast("cuda", enabled=False):
            pred = torch.sigmoid(pred.float())
            target = target.float()

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        tp = (pred_flat * target_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        fp = ((1 - target_flat) * pred_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky).clamp(min=0)


class DeepSupTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, ds_weights=(0.5, 0.25)):
        super().__init__()
        self.main_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.ds_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.ds_weights = ds_weights

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_out, ds_outs = outputs
        else:
            return self.main_loss(outputs, target)

        loss = self.main_loss(main_out, target)

        for i, ds_out in enumerate(ds_outs):
            w = self.ds_weights[i] if i < len(self.ds_weights) else 0.25
            ds_l = self.ds_loss(ds_out, target)
            if not torch.isnan(ds_l):
                loss = loss + w * ds_l

        return loss


# ── Synthetic Lesion Generator ─────────────────────────────────────────────

def generate_synthetic_lesion(patch_shape, brain_mask, image_crop):
    """Generate a realistic synthetic stroke lesion.

    Stroke on DWI: restricted diffusion → BRIGHT on DWI (ch0), DARK on ADC (ch1)
    Data is z-score normalized, so:
      - DWI lesion: +2 to +4 std above mean (bright)
      - ADC lesion: -1 to -3 std below mean (dark/restricted)
      - FLAIR (ch2): +1 to +3 std (hyperintense if present)

    Shape: random ellipsoid with elastic deformation for realism.
    Size: variable — biased toward small (0.5-10ml) to help sensitivity.
    """
    d, h, w = patch_shape

    # 1. Random ellipsoid center (must be in brain)
    brain_coords = np.where(brain_mask > 0)
    if len(brain_coords[0]) == 0:
        return None, None

    idx = np.random.randint(len(brain_coords[0]))
    cz, cy, cx = brain_coords[0][idx], brain_coords[1][idx], brain_coords[2][idx]

    # 2. Random radii — biased toward small lesions
    # Volume ~ 4/3 * pi * r1 * r2 * r3
    # For 1mm spacing: 1ml ≈ 1000 voxels, radius ~6 for sphere
    size_type = np.random.choice(["tiny", "small", "medium", "large"],
                                  p=[0.35, 0.35, 0.20, 0.10])
    if size_type == "tiny":
        radii = np.random.randint(2, 6, size=3)      # ~30-900 voxels
    elif size_type == "small":
        radii = np.random.randint(4, 10, size=3)     # ~250-4000 voxels
    elif size_type == "medium":
        radii = np.random.randint(8, 18, size=3)     # ~2000-24000 voxels
    else:
        radii = np.random.randint(12, 25, size=3)    # ~7000-65000 voxels

    # 3. Create ellipsoid mask
    zz, yy, xx = np.ogrid[0:d, 0:h, 0:w]
    ellipsoid = (
        ((zz - cz) / max(radii[0], 1)) ** 2 +
        ((yy - cy) / max(radii[1], 1)) ** 2 +
        ((xx - cx) / max(radii[2], 1)) ** 2
    ) <= 1.0

    # 4. Add irregularity with smooth random noise
    noise = np.random.randn(d, h, w)
    noise = gaussian_filter(noise, sigma=np.random.uniform(3, 8))
    noise = (noise - noise.mean()) / (noise.std() + 1e-8)
    # Erode/expand ellipsoid based on noise
    threshold = np.random.uniform(-0.3, 0.3)
    irregular = ellipsoid & (noise > threshold)

    # 5. Keep only within brain
    lesion_mask = irregular & (brain_mask > 0)

    if lesion_mask.sum() < 5:  # too small
        return None, None

    # 6. Optionally add a small satellite lesion (multi-focal stroke)
    if np.random.random() < 0.3:
        offset = np.random.randint(-20, 20, size=3)
        sz, sy, sx = cz + offset[0], cy + offset[1], cx + offset[2]
        sat_radii = np.random.randint(2, 5, size=3)
        satellite = (
            ((zz - sz) / max(sat_radii[0], 1)) ** 2 +
            ((yy - sy) / max(sat_radii[1], 1)) ** 2 +
            ((xx - sx) / max(sat_radii[2], 1)) ** 2
        ) <= 1.0
        lesion_mask = lesion_mask | (satellite & (brain_mask > 0))

    # 7. Generate intensity modifications (z-score space)
    intensity_mods = np.zeros((3, d, h, w), dtype=np.float32)

    # DWI/TRACE (ch0): bright — restricted diffusion
    dwi_boost = np.random.uniform(2.0, 4.5)
    # ADC (ch1): dark — restricted diffusion
    adc_drop = np.random.uniform(-3.0, -1.0)
    # FLAIR (ch2): mildly bright (if channel exists)
    flair_boost = np.random.uniform(0.5, 2.5)

    # Smooth the intensity within the lesion for realism (not flat)
    lesion_texture = gaussian_filter(np.random.randn(d, h, w), sigma=4)
    lesion_texture = 0.7 + 0.3 * (lesion_texture - lesion_texture.min()) / \
                     (lesion_texture.max() - lesion_texture.min() + 1e-8)

    intensity_mods[0][lesion_mask] = dwi_boost * lesion_texture[lesion_mask]
    intensity_mods[1][lesion_mask] = adc_drop * lesion_texture[lesion_mask]
    intensity_mods[2][lesion_mask] = flair_boost * lesion_texture[lesion_mask]

    # 8. Smooth edges for realism (penumbra-like gradient)
    for ch in range(3):
        intensity_mods[ch] = gaussian_filter(intensity_mods[ch], sigma=1.5)

    return lesion_mask, intensity_mods


# ── Dataset ─────────────────────────────────────────────────────────────────

class SyntheticAugDataset(torch.utils.data.Dataset):
    """Full-brain dataset with synthetic lesion augmentation.

    Augmentation pipeline:
      1. Random crop (biased toward lesion center if present)
      2. Copy-paste from real donor (30% prob)
      3. Synthetic lesion generation (configurable prob, default 40%)
      4. Standard spatial/intensity augmentations
    """

    def __init__(self, subject_ids, preproc_dir, patch_size=(128, 128, 128),
                 augment=False, ensure_lesion_ratio=0.7,
                 copy_paste_prob=0.3, synth_prob=0.4):
        self.subject_ids = subject_ids
        self.preproc_dir = Path(preproc_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.ensure_lesion_ratio = ensure_lesion_ratio
        self.copy_paste_prob = copy_paste_prob
        self.synth_prob = synth_prob

        self.lesion_subjects = []
        if augment:
            print("  Indexing lesion donors for copy-paste...")
            for sid in tqdm(subject_ids, desc="  Scanning", leave=False):
                npz = np.load(self.preproc_dir / f"{sid}.npz", mmap_mode="r")
                if npz["mask"].any():
                    self.lesion_subjects.append(sid)
            print(f"  Copy-paste donors: {len(self.lesion_subjects)} subjects with lesions")

    def __len__(self):
        return len(self.subject_ids)

    def _random_crop(self, image, mask):
        d, h, w = image.shape[1:]
        pd, ph, pw = self.patch_size

        if d < pd or h < ph or w < pw:
            pad_d = max(pd - d, 0)
            pad_h = max(ph - h, 0)
            pad_w = max(pw - w, 0)
            image = np.pad(image, [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w)], mode="constant")
            mask = np.pad(mask, [(0, pad_d), (0, pad_h), (0, pad_w)], mode="constant")
            d, h, w = image.shape[1:]

        has_lesion = mask.sum() > 0
        crop_on_lesion = has_lesion and np.random.random() < self.ensure_lesion_ratio

        if crop_on_lesion:
            coords = np.where(mask > 0)
            rand_idx = np.random.randint(len(coords[0]))
            center = [int(coords[dim][rand_idx]) for dim in range(3)]
            jitter = [np.random.randint(-pd // 6, pd // 6 + 1) for _ in range(3)]
            start = [
                np.clip(center[i] + jitter[i] - self.patch_size[i] // 2, 0,
                        max(0, [d, h, w][i] - self.patch_size[i]))
                for i in range(3)
            ]
        else:
            start = [
                np.random.randint(0, max(1, s - p + 1))
                for s, p in zip([d, h, w], self.patch_size)
            ]

        sd, sh, sw = start
        return image[:, sd:sd+pd, sh:sh+ph, sw:sw+pw], mask[sd:sd+pd, sh:sh+ph, sw:sw+pw]

    def _elastic_deform(self, image_crop, mask_crop, alpha=50, sigma=8):
        shape = image_crop.shape[1:]
        dz = gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="constant")
        dy = gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="constant")
        dx = gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="constant")

        z, y, x = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing="ij"
        )
        coords = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1),
        ]

        for ch in range(image_crop.shape[0]):
            image_crop[ch] = map_coordinates(image_crop[ch], coords, order=1, mode="reflect")
        mask_crop = map_coordinates(mask_crop, coords, order=0, mode="reflect")

        return image_crop, mask_crop

    def _copy_paste_lesion(self, image_crop, mask_crop):
        donor_sid = self.lesion_subjects[np.random.randint(len(self.lesion_subjects))]
        donor_data = np.load(self.preproc_dir / f"{donor_sid}.npz")
        donor_image = donor_data["image"].astype(np.float32)
        donor_mask = donor_data["mask"].astype(np.float32)

        old_ratio = self.ensure_lesion_ratio
        self.ensure_lesion_ratio = 1.0
        donor_crop, donor_mask_crop = self._random_crop(donor_image, donor_mask)
        self.ensure_lesion_ratio = old_ratio

        lesion_voxels = donor_mask_crop > 0
        if lesion_voxels.sum() == 0:
            return image_crop, mask_crop

        for ch in range(image_crop.shape[0]):
            image_crop[ch][lesion_voxels] = donor_crop[ch][lesion_voxels]
        mask_crop[lesion_voxels] = 1.0

        return image_crop, mask_crop

    def _add_synthetic_lesion(self, image_crop, mask_crop):
        """Generate and paste a synthetic lesion into the crop."""
        # Brain mask: non-zero voxels in TRACE channel
        brain_mask = np.abs(image_crop[0]) > 0.1

        lesion_mask, intensity_mods = generate_synthetic_lesion(
            image_crop.shape[1:], brain_mask, image_crop
        )

        if lesion_mask is None:
            return image_crop, mask_crop

        # Add intensity modifications (not replace — add to existing)
        image_crop = image_crop + intensity_mods

        # Update mask — union with existing
        mask_crop = np.maximum(mask_crop, lesion_mask.astype(np.float32))

        return image_crop, mask_crop

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = np.load(self.preproc_dir / f"{sid}.npz")
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        image_crop, mask_crop = self._random_crop(image, mask)

        if self.augment:
            # Copy-paste augmentation
            if np.random.random() < self.copy_paste_prob:
                image_crop, mask_crop = self._copy_paste_lesion(image_crop, mask_crop)

            # Synthetic lesion augmentation
            if np.random.random() < self.synth_prob:
                image_crop, mask_crop = self._add_synthetic_lesion(image_crop, mask_crop)

        image_t = torch.from_numpy(image_crop.copy())
        mask_t = torch.from_numpy(mask_crop.copy()).unsqueeze(0)

        if self.augment:
            # Elastic deformation (20% prob)
            if np.random.random() < 0.2:
                image_crop, mask_crop = self._elastic_deform(image_crop, mask_crop)
                image_t = torch.from_numpy(image_crop.copy())
                mask_t = torch.from_numpy(mask_crop.copy()).unsqueeze(0)

            # Spatial flips
            for axis in [1, 2, 3]:
                if np.random.random() > 0.5:
                    image_t = torch.flip(image_t, [axis])
                    mask_t = torch.flip(mask_t, [axis])
            # Gaussian noise
            if np.random.random() > 0.5:
                image_t = image_t + torch.randn_like(image_t) * 0.05
            # Intensity scaling
            if np.random.random() > 0.5:
                image_t = image_t * (0.9 + np.random.random() * 0.2)
            # Contrast augmentation
            if np.random.random() > 0.5:
                factor = 0.8 + np.random.random() * 0.4
                mean = image_t.mean(dim=[1, 2, 3], keepdim=True)
                image_t = (image_t - mean) * factor + mean
            # Channel dropout
            if np.random.random() < 0.1:
                ch = np.random.randint(0, 3)
                image_t[ch] = 0.0

        return {"image": image_t, "mask": mask_t, "subject_id": sid}


# ── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(images)
        n += len(images)
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_sum = 0
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * len(images)

            pred = (torch.sigmoid(outputs) > 0.5).float()
            for i in range(len(images)):
                p = pred[i].flatten()
                g = masks[i].flatten()
                inter = (p * g).sum()
                total_vox = p.sum() + g.sum()
                dice = (2 * inter / total_vox).item() if total_vox > 0 else 1.0
                dice_sum += dice
                n += 1

    return total_loss / max(n, 1), dice_sum / max(n, 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SegResNet + Synthetic Lesion Augmentation")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--synth-prob", type=float, default=0.4,
                        help="Probability of adding synthetic lesion per sample (default 0.4)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Tversky alpha (FN weight). Use 0.8 for more aggressive sensitivity")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])

    # Save to separate checkpoint dir to avoid overwriting existing models
    model_name = f"pathB_segresnet_synth_a{args.alpha}"
    ckpt_dir = Path(config["paths"]["checkpoints"]) / model_name / f"fold_{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(splits_dir / "train.json") as f:
        all_train_ids = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_ids = json.load(f)

    # Create fold split (same seed scheme as 19b for consistency)
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

    print(f"SegResNet + DeepSup + Synthetic Lesion Augmentation")
    print(f"  Fold: {args.fold}")
    print(f"  Train: {len(train_ids)}, Fold Val: {len(fold_val_ids)}")
    print(f"  Patch size: {patch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Synth prob: {args.synth_prob}")
    print(f"  Tversky alpha: {args.alpha}")
    print(f"  Device: {device}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Loss: Tversky (alpha={args.alpha}) + Deep Supervision")

    # Model
    model = DeepSupSegResNet(
        in_channels=3, out_channels=1, init_filters=32, dropout_prob=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Dataset with synthetic augmentation
    train_ds = SyntheticAugDataset(
        train_ids, preproc_dir, patch_size=patch_size, augment=True,
        synth_prob=args.synth_prob,
    )
    val_ds = SyntheticAugDataset(
        fold_val_ids, preproc_dir, patch_size=patch_size, augment=False,
    )

    # Oversample tiny/small lesions
    print("  Computing sample weights for lesion size oversampling...")
    sample_weights = []
    spacing = tuple(config["preprocessing"]["target_spacing"])
    voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    for sid in tqdm(train_ids, desc="  Weights", leave=False):
        npz = np.load(preproc_dir / f"{sid}.npz", mmap_mode="r")
        vol_ml = float(npz["mask"].sum()) * voxel_vol_ml
        if vol_ml <= 0:
            sample_weights.append(1.0)
        elif vol_ml < 1:
            sample_weights.append(3.0)
        elif vol_ml < 5:
            sample_weights.append(2.0)
        else:
            sample_weights.append(1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_ids), replacement=True
    )
    print(f"  Oversampling: tiny 3x, small 2x, medium/large/none 1x")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=3,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=3,
    )

    # Loss, optimizer, scheduler
    criterion = DeepSupTverskyLoss(
        alpha=args.alpha, beta=1.0 - args.alpha, ds_weights=(0.5, 0.25)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # TensorBoard
    log_dir = Path(config["paths"]["logs"]) / f"{model_name}_fold{args.fold}"
    writer = SummaryWriter(log_dir)

    best_dice = 0.0
    best_epoch = 0
    patience = 50
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
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
                    "model_state_dict": model.backbone.state_dict(),
                    "full_model_state_dict": model.state_dict(),
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
                "model_state_dict": model.backbone.state_dict(),
                "full_model_state_dict": model.state_dict(),
                "val_dice": val_dice,
            }, ckpt_dir / "checkpoint_latest.pth")

    writer.close()

    print(f"\n{'=' * 60}")
    print(f"SEGRESNET SYNTHETIC AUG TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Dice:  {best_dice:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'checkpoint_best.pth'}")
    print(f"  Synth prob: {args.synth_prob}")
    print(f"  Alpha:      {args.alpha}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
