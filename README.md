# Sentinel Stroke v2 - Clinical Stroke Lesion Segmentation

A clinical-grade pipeline for ischemic stroke lesion detection and segmentation from MRI (TRACE/DWI, ADC, FLAIR). Designed for hospital deployment with dual-path inference for patient safety.

## Clinical Architecture

```
                         ┌──────────────┐
                         │  MRI Scan    │
                         │  TRACE, ADC  │
                         │  FLAIR       │
                         └──────┬───────┘
                                │
                     ┌──────────▼──────────┐
                     │    PREPROCESSING     │
                     │  Co-register FLAIR   │
                     │  Normalize intensity │
                     │  Resample to 1mm     │
                     │  Crop/pad to 192^3   │
                     └──────────┬──────────┘
                                │
               ┌────────────────┼────────────────┐
               │                                 │
    ═══ PATH A: CASCADE ═══          ═══ PATH B: SAFETY NET ═══
               │                                 │
    ┌──────────▼──────────┐          ┌───────────▼───────────┐
    │   STAGE 1            │          │   nnU-Net              │
    │   Detection          │          │   Full-Volume          │
    │   SegResNet @ 2mm    │          │   Sliding Window       │
    │   96^3 input         │          │   192^3 input          │
    │                      │          │   5-fold ensemble      │
    │   "Where is the      │          │                        │
    │    lesion?"           │          │   Direct full-brain    │
    └──────────┬───────────┘          │   segmentation         │
               │                      └───────────┬────────────┘
    ┌──────────▼──────────┐                       │
    │   ROI CROPPING       │                       │
    │   Bounding box       │                       │
    │   + 50% margin       │                       │
    │   128^3 crop @ 1mm   │                       │
    └──────────┬──────────┘                       │
               │                                   │
    ┌──────────▼──────────┐                       │
    │   STAGE 2            │                       │
    │   Segmentation       │                       │
    │   1mm resolution     │                       │
    │                      │                       │
    │   SegResNet  x5 folds│                       │
    │   nnU-Net    x5 folds│                       │
    │   Swin-UNETR x5 folds│                       │
    │         │             │                       │
    │     Ensemble          │                       │
    └─────────┬────────────┘                       │
              │                                    │
              └──────────┬─────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     COMPARATOR       │
              │                      │
              │  Both agree    → Report          │
              │  Disagree      → Flag for        │
              │                  radiologist     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   CLINICAL OUTPUT    │
              │                      │
              │  Lesion mask         │
              │  Volume (ml)         │
              │  Location            │
              │  Confidence level    │
              │  Heatmap overlay     │
              └─────────────────────┘
```

## Confidence Levels

| Scenario | Confidence | Action |
|----------|-----------|--------|
| Path A + Path B agree on lesion | **HIGH** | Auto-report with mask overlay |
| Both find lesion, boundaries differ | **MEDIUM** | Report with both masks shown |
| Only one path finds lesion | **LOW** | Flag for mandatory radiologist review |
| Both find no lesion | **CLEAR** | Report no stroke detected |
| Disagree on lesion count | **REVIEW** | Flag with both outputs |

## Dataset

| Item | Value |
|------|-------|
| **Dataset** | SOOP (OpenNeuro ds004889 v1.1.2) |
| **Total subjects** | 1,715 |
| **Stroke-confirmed** | 1,449 (with acute lesion masks) |
| **Negative controls** | 266 (no confirmed stroke) |
| **Chronic lesions** | 203 subjects |
| **Modalities** | TRACE (DWI), ADC, FLAIR |
| **Mask types** | Acute, Chronic, Combined (all in TRACE space) |

### Lesion Volume Distribution

| Bin | Volume | Count |
|-----|--------|-------|
| Tiny | < 1 ml | 243 |
| Small | 1-5 ml | 380 |
| Medium | 5-50 ml | 560 |
| Large | > 50 ml | 268 |
| None | No mask | 264 |

## Pipeline Scripts

```
Script                              │ Phase   │ What it does
────────────────────────────────────┼─────────┼──────────────────────────────────────
00_download_soop.py                 │ Setup   │ Download SOOP from OpenNeuro
00_audit_soop.py                    │ Setup   │ Audit modalities, masks, spacing, flags
                                    │         │
01_preprocess_soop.py               │ Phase 1 │ Co-register, normalize, resample, crop
02_quality_control.py               │ Phase 1 │ Automated QC checks + visual montages
03_split_dataset.py                 │ Phase 1 │ Stratified train/val/test (70/15/15)
                                    │         │
04_train_stage1_detection.py        │ Phase 2 │ SegResNet @ 2mm, focal loss, recall>95%
05_generate_stage1_rois.py          │ Phase 2 │ Generate ROI bounding boxes
06_prepare_stage2_crops.py          │ Phase 2 │ Crop ROIs at 1mm for Stage 2
                                    │         │
07_train_stage2_segresnet.py        │ Phase 3 │ SegResNet segmentation (5-fold)
06b_convert_crops_to_nnunet.py      │ Phase 3 │ Convert crops to nnU-Net format
07b_train_stage2_nnunet.py          │ Phase 3 │ nnU-Net segmentation (5-fold)
08_train_stage2_swin_unetr.py       │ Phase 3 │ Swin-UNETR segmentation (5-fold)
09_train_stage2_all_folds.py        │ Phase 3 │ Orchestrate all folds/models
                                    │         │
07c_export_nnunet_predictions.py    │ Phase 4 │ Export nnU-Net predictions
10_cascade_inference.py             │ Phase 4 │ Full cascade: detect → crop → segment
11_evaluate_cascade.py              │ Phase 4 │ Evaluate Dice, HD95, per-volume-bin
```

## Preprocessing Pipeline

For each of the 1,715 subjects:

1. **BIDS File Discovery** - Locate TRACE, ADC, FLAIR, and lesion masks
2. **Co-registration** - Rigid registration of FLAIR → TRACE space (mutual information)
3. **Skull Stripping** - SynthStrip (falls back to Otsu threshold)
4. **Intensity Normalization**
   - TRACE/FLAIR: Z-score normalize using brain-only voxels, clip [p1, p99]
   - ADC: Auto-detect units, clip [0, 3000] and scale to [0,1] or z-score
5. **Resampling** - Resample to 1.0 x 1.0 x 1.0 mm isotropic
6. **Crop/Pad** - Center crop or pad to 192 x 192 x 192

**Output:** `<subject_id>.npz` with keys: `image (3,192,192,192)`, `mask (192,192,192)`, `mask_chronic`, `spacing`

## Model Architectures

### Stage 1: Detection (SegResNet)
- **Purpose:** Detect IF and WHERE a stroke lesion exists
- **Input:** Full brain downsampled to 96x96x96 @ 2mm, 3 channels
- **Output:** Coarse probability heatmap → bounding boxes
- **Loss:** Focal Loss (alpha=0.9, gamma=2.0)
- **Target:** >95% case-level detection recall
- **Parameters:** 4.7M

### Stage 2: Segmentation (3-model ensemble)

| Model | Parameters | Strengths |
|-------|-----------|-----------|
| **SegResNet** | ~18M | Fast, lightweight, good on small lesions |
| **nnU-Net** | Self-configured | Self-configuring, usually top performer |
| **Swin-UNETR** | ~62M | Transformer-based, long-range context |

- **Input:** Cropped ROI at 128x128x128 @ 1mm, 3 channels
- **Loss:** Compound (0.4 GDL + 0.4 Focal + 0.2 Boundary)
- **Training:** 5-fold cross-validation per model

### Ensemble Methods
- Simple averaging
- Majority voting
- Learned fusion network

## Training Configuration

```yaml
preprocessing:
  target_spacing: [1.0, 1.0, 1.0]
  target_shape: [192, 192, 192]
  skull_strip: synthstrip (otsu fallback)

stage1:
  architecture: segresnet_small
  input_shape: [96, 96, 96]
  resolution: 2mm isotropic
  loss: focal (alpha=0.9)
  epochs: 300 (early stopping patience=50)
  batch_size: 4

stage2:
  models: [segresnet, nnunet, swin_unetr]
  input_shape: [128, 128, 128]
  resolution: 1mm isotropic
  loss: compound (GDL + Focal + Boundary)
  epochs: 1000 (early stopping patience=100)
  folds: 5

data_split:
  train: 70% (1199 subjects)
  val: 15% (258 subjects)
  test: 15% (258 subjects)
  stratified_by: [lesion_volume_bin, has_chronic_lesion]
```

## Expected Performance

| Component | Metric | Expected |
|-----------|--------|----------|
| Stage 1 Detection | Case-level recall | >95% |
| Stage 1 Detection | Coarse Dice | 0.80-0.85 |
| Stage 2 Single Model | Dice | 0.55-0.70 |
| Stage 2 Ensemble | Dice | 0.60-0.75 |
| Final (large lesions) | Dice | 0.80+ |
| Final (tiny lesions) | Dice | 0.30-0.45 |

*Note: Stroke segmentation is inherently difficult. ISLES 2022 challenge winner achieved ~0.65 mean Dice.*

### Sensitivity by Lesion Size

| Lesion Size | Expected Sensitivity |
|-------------|---------------------|
| Large (>50ml) | 95-98% |
| Medium (5-50ml) | 90-95% |
| Small (1-5ml) | 80-90% |
| Tiny (<1ml) | 50-70% |

## Inference Timeline

```
Time:  0s ─────── 5s ─────── 15s ─────── 25s ─────── 30s
       │          │           │           │           │
       │Preprocess│  Stage 1  │   Stage 2 │  Compare  │
       │          │  Detect   │  Segment  │  & Report │
       │          │           │           │           │
       │Preprocess│   nnU-Net safety net  │  Compare  │
       │          │   (sliding window)    │  & Report │
                                                 Total: ~30s
```

## Project Structure

```
sentinel_stroke/
├── configs/
│   └── soop_config.yaml           # All configuration
├── scripts/
│   ├── 00_download_soop.py        # Data download
│   ├── 00_audit_soop.py           # Dataset audit
│   ├── 01_preprocess_soop.py      # Preprocessing pipeline
│   ├── 02_quality_control.py      # QC checks
│   ├── 03_split_dataset.py        # Train/val/test split
│   ├── 04_train_stage1_detection.py
│   ├── 05_generate_stage1_rois.py
│   ├── 06_prepare_stage2_crops.py
│   ├── 06b_convert_crops_to_nnunet.py
│   ├── 07_train_stage2_segresnet.py
│   ├── 07b_train_stage2_nnunet.py
│   ├── 07c_export_nnunet_predictions.py
│   ├── 08_train_stage2_swin_unetr.py
│   ├── 09_train_stage2_all_folds.py
│   ├── 10_cascade_inference.py
│   └── 11_evaluate_cascade.py
├── src/
│   ├── data/                      # Data utilities
│   ├── inference/                 # Inference modules
│   ├── losses/                    # Custom loss functions
│   ├── models/                    # Model definitions
│   └── utils/                     # General utilities
├── checkpoints/                   # Saved model weights
├── logs/                          # TensorBoard logs
├── requirements.txt
├── ARCHITECTURE.md                # Detailed architecture doc
├── LAMBDA_SOOP_SETUP.md           # Server setup guide
└── README.md
```

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB (V100) | 40 GB (A100) |
| RAM | 64 GB | 128 GB |
| Disk | 400 GB | 500 GB+ |
| Training time | ~3-4 days (A100) | - |

## Current Training Infrastructure

| Item | Value |
|------|-------|
| GPU | 1x A100 SXM4 40GB |
| Provider | Lambda Labs |
| Instance | `129.213.93.64` |
| SSH | `ssh ubuntu@129.213.93.64` |
| Project | `~/sentinel_stroke/` |
| Data | `~/data/SOOP_Dataset/` |
| Venv | `~/sentinel_v2/` |

## Bug Fixes Applied

| Bug | File | Impact | Fix |
|-----|------|--------|-----|
| Wrong derivatives path | `01_preprocess_soop.py:431` | All masks saved as zeros | Read path from config |
| Focal Loss overflow | `04, 07, 08` | Training loss explodes to NaN | Clamp predictions, compute p_t via sigmoid |
| Broken import | `08_train_stage2_swin_unetr.py:33` | Script crashes on startup | Catch ImportError/ModuleNotFoundError |
| DiceMetric NaN | `04, 07` | Validation Dice always NaN | Convert to 2-channel one-hot |

## Key References

1. **SOOP Dataset:** Absher et al. (2024). "The stroke outcome optimization project." Scientific Data, 11(1), 839.
2. **DeepISLES:** de la Rosa et al. (2025). "DeepISLES: a clinically validated ischemic stroke segmentation model." Nature Communications 16, 7357.
3. **ISLES'24 Winner:** Heras Rivera et al. (2025). "How We Won the ISLES'24 Challenge by Preprocessing." MIDL 2025.
4. **nnU-Net:** Isensee et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods 18, 203-211.
5. **Swin-UNETR:** Hatamizadeh et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."

## License

[Your License Here]
