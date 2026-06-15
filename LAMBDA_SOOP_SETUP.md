# Lambda Labs Setup Guide - Sentinel Stroke v2 (SOOP)

## Quick Reference

| Item | Value |
|------|-------|
| **GPU** | 1x A100 SXM4 40GB (gpu_1x_a100_sxm4) |
| **Instance IP** | `129.213.93.64` |
| **Region** | us-east-1 |
| **SSH** | `ssh ubuntu@129.213.93.64` |
| **SSH Key** | prototype |
| **Dataset** | SOOP (ds004889) - 1715 subjects |
| **Stroke-confirmed** | 1449 subjects |
| **Project folder** | `~/sentinel_stroke/` |
| **Data folder** | `~/data/SOOP_Dataset/` |
| **Config** | `configs/soop_config.yaml` |
| **Venv** | `~/sentinel_v2/` |
| **Estimated total time** | ~3-4 days (full pipeline) |

## Current Progress (2026-04-05)

| Step | Status | Result |
|------|--------|--------|
| Dataset download | DONE | 1715 subjects |
| Audit | DONE | 1451 acute masks, 203 chronic |
| Preprocessing | DONE | 1715/1715 processed (Otsu skull strip) |
| QC | DONE | 0 failures, 1033 warn (trace_adc_consistency — expected) |
| Split | DONE | Train 1199 / Val 258 / Test 258 |
| Stage 1 training | DONE | Dice 0.842, Recall 1.000 (early stop epoch 243) |
| Stage 1 ROIs | DONE | 100% detection recall (1015/1015 train, 218/218 val, 218/218 test) |
| Stage 2 crops | DONE | Train: 1501 crops, Val/Test: generated |
| Stage 2 SegResNet | IN PROGRESS | fold 0 |
| Stage 2 nnU-Net | PENDING | |
| Stage 2 Swin-UNETR | PENDING | |
| Cascade inference | PENDING | |
| Evaluation | PENDING | |

### Bug Fixes Applied

| Bug | File | Fix |
|-----|------|-----|
| Wrong derivatives path | `01_preprocess_soop.py:431` | `raw_dir / "derivatives"` → read from config |
| Focal Loss overflow | `04, 07, 08` | Clamp pred to [-20,20], p_t via sigmoid |
| Broken import | `08_swin_unetr.py:33` | `except NameError` → `except (ImportError, ModuleNotFoundError)` |
| DiceMetric NaN | `04, 07` | `include_background=False` → True with 2-channel one-hot |

---

## Phase 0: Connect & Upload Code

### Step 1: SSH into Lambda

```powershell
ssh ubuntu@129.213.93.64
```

### Step 2: Upload project code (from Windows PowerShell)

```powershell
cd "C:\Users\ADMIN\OneDrive\Desktop\stroke_mvp"

# Compress project (use .\ paths for Windows tar)
tar -cvzf stroke_v2.tar.gz --exclude="v1" --exclude="*.zip" --exclude="__pycache__" .\scripts\ .\src\ .\configs\ .\requirements.txt

# Upload to Lambda
scp stroke_v2.tar.gz ubuntu@129.213.93.64:~/
```

### Step 3: Extract on Lambda

```bash
ssh ubuntu@129.213.93.64 "mkdir -p ~/sentinel_stroke ~/data/SOOP_Dataset && cd ~/sentinel_stroke && tar -xvzf ~/stroke_v2.tar.gz && rm ~/stroke_v2.tar.gz"
```

---

## Phase 1: Environment Setup

### Step 4: Install dependencies

```bash
# Lambda comes with PyTorch + CUDA pre-installed. Verify first:
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Create virtual environment (Lambda has no conda)
python3 -m venv ~/sentinel_v2
source ~/sentinel_v2/bin/activate

# PyTorch is system-installed on Lambda — install with matching CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
cd ~/sentinel_stroke
pip install -r requirements.txt

# SynthStrip for skull stripping — NOT available via pip
# Falls back to Otsu threshold (acceptable for stroke segmentation)
# To install manually: pip install surfa, then download model weights
# See README.md for details

# Install AWS CLI for dataset download
pip install awscli
```

> **Tip:** Add `source ~/sentinel_v2/bin/activate` to your `~/.bashrc` so it activates on every SSH login:
> ```bash
> echo 'source ~/sentinel_v2/bin/activate' >> ~/.bashrc
> ```

### Step 5: Set nnU-Net environment variables

```bash
# Add to ~/.bashrc
echo 'export nnUNet_raw="/home/ubuntu/nnUNet/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/home/ubuntu/nnUNet/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/home/ubuntu/nnUNet/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc

# Create nnU-Net directories
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

---

## Phase 2: Download SOOP Dataset

### Step 6: Download from OpenNeuro (~2-4 hours)

```bash
# Use screen so download survives SSH disconnect
screen -S download

cd ~/sentinel_stroke
python3 scripts/00_download_soop.py --data-dir /home/ubuntu/data/SOOP_Dataset

# Detach: Ctrl+A then D
# Reattach: screen -r download
```

This downloads:
- All 1715 subjects (TRACE, ADC, FLAIR per subject)
- Derivatives (acute, chronic, combined lesion masks)
- `participants.tsv` (demographics, NIHSS scores)

### Step 7: Verify download

```bash
# Check subject count
ls /home/ubuntu/data/SOOP_Dataset/raw/ | grep "sub-" | wc -l
# Expected: 1715

# Check derivatives
ls /home/ubuntu/data/SOOP_Dataset/derivatives/ | head -5

# Check a sample subject
ls -R /home/ubuntu/data/SOOP_Dataset/raw/sub-1/
# Should show: anat/sub-1_FLAIR.nii.gz, dwi/sub-1_rec-TRACE_dwi.nii.gz, etc.
```

---

## Phase 3: Run the Pipeline

**Important:** Always use `screen` for long-running steps. Each script logs progress so you can monitor.

### Step 8: Audit the dataset (~30 min)

```bash
screen -S pipeline
cd ~/sentinel_stroke
source ~/sentinel_v2/bin/activate

python3 scripts/00_audit_soop.py
```

Produces a report of: missing modalities, voxel spacings, lesion volumes, flagged subjects.

### Step 9: Preprocessing (~6-12 hours)

```bash
python3 scripts/01_preprocess_soop.py
```

This handles: co-registration (FLAIR -> TRACE), skull stripping, intensity normalization, resampling to 1mm isotropic, cropping/padding.

### Step 10: Quality control

```bash
python3 scripts/02_quality_control.py
```

Review the QC report. Fix or exclude any flagged subjects before proceeding.

### Step 11: Split dataset

```bash
python3 scripts/03_split_dataset.py
```

Creates stratified train/val/test split (70/15/15) saved as JSON.

### Step 12: Train Stage 1 - Detection (~12-24 hours)

```bash
python3 scripts/04_train_stage1_detection.py
```

Trains a lightweight SegResNet at 2mm resolution optimized for recall (>95% detection rate).

### Step 13: Generate Stage 1 ROIs (~1-2 hours)

```bash
python3 scripts/05_generate_stage1_rois.py
```

Runs Stage 1 inference on training set to produce bounding boxes for Stage 2.

### Step 14: Prepare Stage 2 crops (~1-2 hours)

```bash
# IMPORTANT: Must run for all three splits separately
python3 scripts/06_prepare_stage2_crops.py --config configs/soop_config.yaml --split train
python3 scripts/06_prepare_stage2_crops.py --config configs/soop_config.yaml --split val
python3 scripts/06_prepare_stage2_crops.py --config configs/soop_config.yaml --split test
```

Crops full-resolution patches around detected ROIs with adaptive sizing. The `--split` flag defaults to `train` only — you must run all three or Stage 2 training will fail with missing `manifest.json`.

### Step 15: Train Stage 2 - Segmentation models

Option A - SegResNet (~24-48 hours per fold):

```bash
python3 scripts/07_train_stage2_segresnet.py --fold 0
```

Option B - nnU-Net:

```bash
python3 scripts/06b_convert_crops_to_nnunet.py
python3 scripts/07b_train_stage2_nnunet.py --fold 0
```

Option C - Swin-UNETR (~24-48 hours per fold):

```bash
python3 scripts/08_train_stage2_swin_unetr.py --fold 0
```

Option D - All folds for all models (automated):

```bash
python3 scripts/09_train_stage2_all_folds.py
```

### Step 16: Cascade inference & evaluation

```bash
python3 scripts/10_cascade_inference.py
python3 scripts/11_evaluate_cascade.py
```

---

## Monitoring & Management

### GPU monitoring (new terminal)

```bash
ssh ubuntu@129.213.93.64
watch -n 2 nvidia-smi
```

### Check training logs

```bash
# TensorBoard
tensorboard --logdir ~/sentinel_stroke/logs/tensorboard --port 6006 &

# From your local machine, tunnel the port:
# ssh -L 6006:localhost:6006 ubuntu@129.213.93.64
# Then open http://localhost:6006
```

### Screen session management

```bash
screen -ls              # List sessions
screen -r pipeline      # Reattach
# Ctrl+A then D         # Detach
```

---

## Download Results (After Training)

From Windows PowerShell:

```powershell
# Download checkpoints
scp -r ubuntu@129.213.93.64:~/sentinel_stroke/checkpoints/ ./checkpoints_v2/

# Download nnU-Net results
scp -r ubuntu@129.213.93.64:~/nnUNet/nnUNet_results/ ./nnunet_results_v2/

# Download evaluation reports
scp -r ubuntu@129.213.93.64:~/data/SOOP_Dataset/reports/ ./soop_reports/

# Download logs
scp -r ubuntu@129.213.93.64:~/sentinel_stroke/logs/ ./logs_v2/
```

---

## Cost Estimate (A100 40GB @ $1.10/hr)

| Phase | Time | Cost |
|-------|------|------|
| Setup + Download | ~4 hours | ~$5 |
| Audit + Preprocess + QC | ~12 hours | ~$15 |
| Stage 1 Detection | ~24 hours | ~$31 |
| Stage 1 ROI generation | ~2 hours | ~$3 |
| Stage 2 (1 model, 1 fold) | ~36 hours | ~$46 |
| Stage 2 (all 3 models, 5 folds) | ~540 hours | ~$697 |
| Cascade inference + eval | ~4 hours | ~$5 |
| **Total (1 model, 1 fold)** | **~78 hours** | **~$100** |
| **Total (full pipeline)** | **~586 hours** | **~$756** |

**Strategy:** Train 1 fold of SegResNet first to validate the pipeline end-to-end (~$100). Then scale up.

---

## Troubleshooting

### Out of Memory
```bash
# Check GPU memory
nvidia-smi
# Reduce batch_size in configs/soop_config.yaml
# Stage 1: try batch_size 2
# Stage 2: use smaller patch size
```

### Download interrupted
```bash
# 00_download_soop.py uses aws s3 sync — just re-run it
# It will skip already-downloaded files
python3 scripts/00_download_soop.py --data-dir /home/ubuntu/data/SOOP_Dataset
```

### Training interrupted
```bash
# Most scripts auto-resume from latest checkpoint
# Just re-run the same command
```

### Disk space check
```bash
df -h /home/ubuntu
# SOOP raw data: ~150-200 GB
# Preprocessed: ~100-150 GB
# nnU-Net data: ~50-100 GB
# Checkpoints: ~20-50 GB
# Total needed: ~400-500 GB minimum
```

### SSH disconnected during training
```bash
# If you used screen, just reattach:
screen -r pipeline
# Training continues in background
```
