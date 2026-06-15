"""
Binary Classifier: Lesion Present Yes/No (FP Suppression Gate)
================================================================
Trains a lightweight 3D classifier on downsampled full-brain volumes
to predict whether a stroke lesion is present or not.

Used as a gate BEFORE segmentation: if classifier says "no lesion" with
high confidence → skip segmentation → eliminates false positives.

Architecture: 3D ResNet-18 style (small, fast inference)
Input: 3-channel (TRACE+ADC+FLAIR) at 2mm isotropic (96³)
Output: Binary probability (lesion present)

Usage:
  python scripts/18_train_binary_classifier.py --config configs/soop_config.yaml
  python scripts/18_train_binary_classifier.py --config configs/soop_config.yaml --epochs 100
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


# ── Model ───────────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_ch, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.leaky_relu(out, 0.01)


class BrainLesionClassifier(nn.Module):
    """Lightweight 3D ResNet classifier for lesion detection."""

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(32, affine=True)

        self.layer1 = nn.Sequential(ResBlock3D(32, 32), ResBlock3D(32, 32))
        self.layer2 = nn.Sequential(ResBlock3D(32, 64, stride=2), ResBlock3D(64, 64))
        self.layer3 = nn.Sequential(ResBlock3D(64, 128, stride=2), ResBlock3D(128, 128))
        self.layer4 = nn.Sequential(ResBlock3D(128, 256, stride=2), ResBlock3D(256, 256))

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ── Dataset ─────────────────────────────────────────────────────────────────

class BrainClassifierDataset(torch.utils.data.Dataset):
    """Loads preprocessed .npz, downsamples, returns binary label."""

    def __init__(self, subject_ids, preproc_dir, input_shape=(96, 96, 96), augment=False):
        self.subject_ids = subject_ids
        self.preproc_dir = Path(preproc_dir)
        self.input_shape = input_shape
        self.augment = augment

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = np.load(self.preproc_dir / f"{sid}.npz")

        image = data["image"].astype(np.float32)  # (3, D, H, W)
        mask = data["mask"].astype(np.float32)     # (D, H, W)

        # Downsample to 96³
        image_t = torch.from_numpy(image).unsqueeze(0)
        image_t = F.interpolate(image_t, size=self.input_shape, mode="trilinear",
                                align_corners=False).squeeze(0)

        # Binary label
        has_lesion = 1.0 if mask.sum() > 0 else 0.0

        # Augmentation
        if self.augment:
            for axis in [1, 2, 3]:
                if np.random.random() > 0.5:
                    image_t = torch.flip(image_t, [axis])
            if np.random.random() > 0.5:
                image_t = image_t + torch.randn_like(image_t) * 0.05
            if np.random.random() > 0.5:
                image_t = image_t * (0.9 + np.random.random() * 0.2)

        return {
            "image": image_t,
            "label": torch.tensor([has_lesion], dtype=torch.float32),
            "subject_id": sid,
        }


# ── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(images)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_sids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(images)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_sids.extend(batch["subject_id"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Metrics at threshold 0.5
    preds = (all_probs > 0.5).astype(float)
    accuracy = (preds == all_labels).mean()

    # Sensitivity (recall on positive cases)
    pos_mask = all_labels == 1
    sensitivity = preds[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

    # Specificity (recall on negative cases)
    neg_mask = all_labels == 0
    specificity = (1 - preds[neg_mask]).mean() if neg_mask.sum() > 0 else 0.0

    # FP rate on no-lesion cases
    fp_rate = preds[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fp_rate": fp_rate,
        "n_positive": int(pos_mask.sum()),
        "n_negative": int(neg_mask.sum()),
        "probs": all_probs,
        "labels": all_labels,
        "sids": all_sids,
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary lesion classifier")
    parser.add_argument("--config", default="configs/soop_config.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preproc_dir = Path(config["paths"]["preprocessed"])
    splits_dir = Path(config["paths"]["splits"])
    ckpt_dir = Path(config["paths"]["checkpoints"]) / "binary_classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(splits_dir / "train.json") as f:
        train_ids = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_ids = json.load(f)

    # Count class balance
    n_pos_train = sum(1 for sid in train_ids
                      if np.load(preproc_dir / f"{sid}.npz")["mask"].sum() > 0)
    n_neg_train = len(train_ids) - n_pos_train

    print(f"Binary Lesion Classifier Training")
    print(f"  Train: {len(train_ids)} ({n_pos_train} positive, {n_neg_train} negative)")
    print(f"  Val:   {len(val_ids)}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")

    # Datasets
    train_ds = BrainClassifierDataset(train_ids, preproc_dir, augment=True)
    val_ds = BrainClassifierDataset(val_ids, preproc_dir, augment=False)

    # Balanced sampler: oversample no-lesion cases so each batch is ~50/50
    sample_weights = []
    for sid in train_ids:
        has_lesion = np.load(preproc_dir / f"{sid}.npz")["mask"].sum() > 0
        # Weight inversely proportional to class size
        sample_weights.append(1.0 / n_pos_train if has_lesion else 1.0 / max(n_neg_train, 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_ids), replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = BrainLesionClassifier(in_channels=3, num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # BCE loss — sampler handles class balance, so use equal weight
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    log_dir = Path(config["paths"]["logs"]) / "binary_classifier"
    writer = SummaryWriter(log_dir)

    best_specificity = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_results['loss']:.4f} Acc: {val_results['accuracy']:.3f} | "
              f"Sens: {val_results['sensitivity']:.3f} Spec: {val_results['specificity']:.3f} | "
              f"FP: {val_results['fp_rate']:.3f} | {elapsed:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_results["loss"], epoch)
        writer.add_scalar("Accuracy/val", val_results["accuracy"], epoch)
        writer.add_scalar("Sensitivity/val", val_results["sensitivity"], epoch)
        writer.add_scalar("Specificity/val", val_results["specificity"], epoch)

        # Save best by specificity (while maintaining high sensitivity)
        if (val_results["specificity"] > best_specificity and
                val_results["sensitivity"] > 0.95):
            best_specificity = val_results["specificity"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "specificity": val_results["specificity"],
                "sensitivity": val_results["sensitivity"],
                "accuracy": val_results["accuracy"],
                "fp_rate": val_results["fp_rate"],
            }, ckpt_dir / "checkpoint_best.pth")
            print(f"  *** New best: Spec {best_specificity:.3f} (Sens {val_results['sensitivity']:.3f}) ***")

        # Save latest
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, ckpt_dir / "checkpoint_latest.pth")

    # Final evaluation with best model
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best specificity: {best_specificity:.4f}")

    # Load best and do threshold sweep
    ckpt = torch.load(ckpt_dir / "checkpoint_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_results = validate(model, val_loader, criterion, device)

    print(f"\n  Threshold sweep on best model:")
    print(f"  {'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'FP Rate':>10} {'Accuracy':>10}")
    print(f"  {'-' * 58}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (val_results["probs"] > thresh).astype(float)
        labels = val_results["labels"]

        pos_mask = labels == 1
        neg_mask = labels == 0

        sens = preds[pos_mask].mean() if pos_mask.sum() > 0 else 0
        spec = (1 - preds[neg_mask]).mean() if neg_mask.sum() > 0 else 0
        fp = preds[neg_mask].mean() if neg_mask.sum() > 0 else 0
        acc = (preds == labels).mean()

        marker = " <-- recommended" if sens > 0.98 and spec > best_specificity * 0.9 else ""
        print(f"  {thresh:>10.1f} {sens:>12.4f} {spec:>12.4f} {fp:>10.4f} {acc:>10.4f}{marker}")

    # Save per-subject predictions for integration
    pred_output = {
        "subjects": [],
        "threshold_recommended": 0.5,
    }
    for sid, prob, label in zip(val_results["sids"], val_results["probs"], val_results["labels"]):
        pred_output["subjects"].append({
            "subject_id": sid,
            "lesion_probability": round(float(prob), 4),
            "gt_has_lesion": bool(label > 0.5),
        })

    pred_path = ckpt_dir / "val_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(pred_output, f, indent=2)

    print(f"\n  Checkpoint: {ckpt_dir / 'checkpoint_best.pth'}")
    print(f"  Val predictions: {pred_path}")
    print(f"  Next: integrate into cascade inference (gate before Stage 2)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
