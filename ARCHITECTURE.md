# Sentinel Stroke v2 - Clinical Architecture

## Overview

Sentinel Stroke is a dual-path clinical pipeline for ischemic stroke lesion segmentation. It runs two independent inference paths in parallel and compares results to provide confidence-rated outputs suitable for hospital deployment.

## Why Dual-Path?

A single cascade pipeline has a critical failure mode: if Stage 1 (detection) misses a lesion, Stage 2 (segmentation) never sees it, resulting in a **silent false negative**. In a hospital setting, this means a missed stroke.

The dual-path design eliminates this blind spot:
- **Path A (Cascade):** Fast, focused, high-resolution segmentation via detect-then-crop
- **Path B (Safety Net):** Full-volume nnU-Net sliding window — no cropping, no detection dependency

If both paths agree, confidence is high. If they disagree, the case is flagged for radiologist review.

## Path A: Cascade Pipeline

### Stage 1 - Detection

```
Input:  Full brain volume, downsampled to 2mm (96x96x96), 3 channels
Model:  SegResNet (4.7M parameters)
Loss:   Focal Loss (alpha=0.9, gamma=2.0)
Output: Probability heatmap → bounding boxes via connected components
```

**Design rationale:**
- Low resolution (2mm) allows processing the entire brain in one forward pass
- Focal loss with high alpha prioritizes recall over precision — we'd rather have false positives (Stage 2 handles them) than false negatives (missed strokes)
- Detection threshold is set low (0.2) to maximize sensitivity
- Achieved: 100% detection recall on all splits (1015/1015 train, 218/218 val, 218/218 test)
- Dice: 0.842 (coarse, but sufficient for bounding box generation)

### ROI Cropping

```
Input:  Stage 1 heatmap + original full-resolution (1mm) volume
Process:
  1. Threshold heatmap at 0.2
  2. Connected component analysis
  3. Compute bounding box per component
  4. Expand each box by 50% margin (roi_margin_expand: 0.5)
  5. Crop from original 1mm volume
Output: 128x128x128 crops at 1mm resolution
```

**Why crop instead of processing full volume?**
- Full brain at 1mm = 192x192x192 = 7M voxels — too large for powerful models on GPU
- Cropped ROI = 128x128x128 = 2M voxels — allows bigger, more accurate models
- Class balance improves dramatically (lesion is larger fraction of crop)
- Negative ROIs (false positive detections) get rejected by Stage 2 naturally

### Stage 2 - Segmentation (3-Model Ensemble)

```
Input:  128x128x128 crop at 1mm, 3 channels (TRACE, ADC, FLAIR)
Models: SegResNet + nnU-Net + Swin-UNETR (each trained with 5-fold CV)
Loss:   Compound = 0.4*GDL + 0.4*Focal + 0.2*Boundary
Output: Voxel-level lesion probability map
```

**SegResNet (MONAI)**
- Residual encoder-decoder with deep supervision
- Fast inference, lightweight (~18M params)
- Good baseline, especially for small lesions
- Uses init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1]

**nnU-Net v2**
- Self-configuring: automatically determines patch size, spacing, architecture, augmentation
- Consistently wins medical segmentation challenges
- Trained via native nnU-Net pipeline on crops converted to nnU-Net format
- Most robust single model

**Swin-UNETR (MONAI)**
- Transformer-based encoder (Swin Transformer) with CNN decoder
- Captures long-range spatial dependencies that CNNs miss
- Largest model (~62M params), slowest but most expressive
- feature_size=48, uses Swin V2

**Ensemble strategy:**
- Simple averaging of probability maps from all 15 models (3 architectures x 5 folds)
- Majority voting as alternative
- Optional learned fusion network

### Stage 2 - Compound Loss Function

```
CompoundLoss = 0.4 * GeneralizedDiceLoss + 0.4 * FocalLoss + 0.2 * BoundaryLoss
```

- **Generalized Dice Loss (GDL):** Weights each class inversely by volume. Handles extreme class imbalance (tiny lesions). Uses sigmoid activation.
- **Focal Loss (alpha=0.75, gamma=2.0):** Down-weights easy negatives, focuses gradient on hard examples (boundary voxels, tiny lesions). Predictions clamped to [-20, 20] for numerical stability.
- **Boundary Loss:** Extra BCE weight (3x) on voxels detected as boundaries via 3D Laplacian convolution. Improves boundary delineation for small lesions.

## Path B: Safety Net (Full-Volume nnU-Net)

```
Input:  Full brain volume at 1mm (192x192x192), 3 channels
Model:  nnU-Net with sliding window inference
Output: Full-brain segmentation mask
```

**Why nnU-Net for the safety net?**
- Self-configuring — handles full volumes with automatic patch-based sliding window
- No dependency on Stage 1 detection — processes everything
- Battle-tested across hundreds of medical segmentation challenges
- If the cascade misses a lesion, nnU-Net's sliding window likely catches it

**Status:** To be trained after Stage 2 models complete.

## Comparator Module

After both paths complete inference, the comparator evaluates agreement:

```python
# Pseudocode
cascade_mask = path_a_inference(scan)
safety_mask  = path_b_inference(scan)

cascade_has_lesion = cascade_mask.sum() > 0
safety_has_lesion  = safety_mask.sum() > 0
overlap_dice       = dice(cascade_mask, safety_mask)

if not cascade_has_lesion and not safety_has_lesion:
    confidence = "CLEAR"
    action = "Report no stroke"
elif cascade_has_lesion and safety_has_lesion and overlap_dice > 0.7:
    confidence = "HIGH"
    action = "Auto-report with mask overlay"
elif cascade_has_lesion and safety_has_lesion and overlap_dice > 0.3:
    confidence = "MEDIUM"
    action = "Report with both masks shown"
elif cascade_has_lesion != safety_has_lesion:
    confidence = "LOW"
    action = "Flag for mandatory radiologist review"
else:
    confidence = "REVIEW"
    action = "Flag with disagreement details"
```

## Preprocessing Details

### Input Modalities

| Channel | Modality | What it shows | Role in stroke detection |
|---------|----------|--------------|--------------------------|
| 0 | TRACE (DWI) | Diffusion-weighted imaging | Acute lesion appears BRIGHT (restricted diffusion) |
| 1 | ADC | Apparent Diffusion Coefficient | Acute lesion appears DARK (low ADC confirms true restriction) |
| 2 | FLAIR | Fluid-attenuated inversion recovery | Shows chronic lesions and edema, provides anatomical context |

### Normalization Strategy

**TRACE and FLAIR: Z-score normalization**
```python
brain_voxels = data[brain_mask > 0]
p1, p99 = percentile(brain_voxels, [1, 99])
data = clip(data, p1, p99)
data = (data - brain_mean) / brain_std
```

**ADC: Auto-detect units**
```python
if brain_mean > 100:
    # Standard units (x10^-6 mm^2/s): clip and scale
    data = clip(data, 0, 3000) / 3000  # → [0, 1]
else:
    # Already scaled or different units: z-score
    data = z_score(data, brain_mask)
```

### Data Split Strategy

```
Total: 1715 subjects
├── Train: 1199 (70%)
├── Val:    258 (15%)
└── Test:   258 (15%)

Stratified by:
├── Lesion volume bin (tiny/small/medium/large/none)
├── Has chronic lesion (yes/no)
└── NIHSS quartile (from participants.tsv)
```

Negative controls (266 subjects with no confirmed stroke) are distributed proportionally across splits (~12% of each batch) to train the model to say "no lesion" when appropriate.

## Training Protocol

### Stage 1
1. Train SegResNet on downsampled full-brain volumes
2. Optimize for recall using Focal Loss
3. Early stopping on validation Dice (patience=50)
4. Save best checkpoint by recall AND by Dice

### Stage 2
1. Run Stage 1 inference on training set → generate ROI bounding boxes
2. Crop full-resolution patches around each ROI (+ negative patches at 30% ratio)
3. Train SegResNet, nnU-Net, Swin-UNETR independently on the same crops
4. 5-fold cross-validation for each model
5. Early stopping on validation Dice (patience=100)

### Safety Net
1. Train standalone nnU-Net on full-volume data (no cascade dependency)
2. Uses nnU-Net's native sliding window training and inference

## Inference Pipeline (Hospital Deployment)

```
1. DICOM received from scanner
2. Convert DICOM → NIfTI
3. Preprocessing (co-register, normalize, resample) — ~5s
4. Path A: Stage 1 detection — ~3s
5. Path A: ROI cropping — <1s
6. Path A: Stage 2 ensemble (15 models) — ~10s
7. Path B: nnU-Net sliding window — ~15s (runs parallel with Path A)
8. Comparator: merge + confidence scoring — <1s
9. Generate clinical report — <1s
                                    Total: ~20-30s
```

## Clinical Output

The system produces:
- **Lesion segmentation mask** (NIfTI format, can overlay on original scan)
- **Lesion volume** in milliliters
- **Lesion location** (anatomical region via atlas lookup)
- **Confidence level** (HIGH / MEDIUM / LOW / CLEAR / REVIEW)
- **Heatmap overlay** (probability map for visualization)
- **Disagreement report** (if paths disagree, both masks shown side-by-side)

## Limitations

1. **Skull stripping:** Currently uses Otsu threshold fallback (SynthStrip not available). May affect normalization quality at brain edges. Impact on final Dice estimated at 1-2%.

2. **Tiny lesions (<1ml):** Expected sensitivity 50-70%. Fundamental limitation — at 1mm resolution, a 0.5ml lesion is ~500 voxels in a 7M voxel volume.

3. **Cascade failure mode:** If Stage 1 misses a lesion, the cascade path won't segment it. Mitigated by: (a) very low detection threshold (0.2), (b) safety net path, (c) 100% detection recall on current data.

4. **Scanner variability:** SOOP data is from one center. Performance on external data (different scanners, protocols) may degrade. Recommend validation on external datasets before deployment.

5. **Chronic vs acute:** The model is trained to segment acute lesions only. Chronic lesions (visible on FLAIR but not restricted on DWI) should be excluded. The compound loss + ADC channel help, but edge cases exist.
