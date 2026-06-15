Sentinel Stroke System — SOOP Migration & Architecture Guide
Table of Contents
1.	SOOP Dataset: What Changed from ISLES 2022
2.	SOOP-Specific Preprocessing Pipeline
3.	Cascaded Detection → Segmentation for Tiny Lesions
4.	Lesion-Aware Patching Strategy
5.	Key Notes & Gotchas
6.	Updated Phase-by-Phase Checklist
________________________________________
1. SOOP Dataset: What Changed from ISLES 2022
1.1 Scale & Structure Differences
Aspect	ISLES 2022	SOOP (ds004889)
Subjects	~400	1715 (1449 confirmed stroke)
Modalities	DWI, ADC, FLAIR (per subject)	TRACE (≈DWI), ADC, FLAIR (per subject)
Lesion masks	Single binary mask (msk.nii.gz)	Three masks: acute, chronic, combined (in derivatives)
Mask space	Varies	All in TRACE space (space-TRACE)
Data format	Custom folders per subject	Full BIDS format
Heterogeneity	Multi-vendor, moderate variation	High variation — protocols and diffusion parameters differ between individuals
Public ground truth	Training set only (250)	All 1449 stroke-confirmed cases have masks
1.2 File Naming Conventions (SOOP)
Per subject (e.g., sub-1):
sub-1/
├── anat/
│   ├── sub-1_FLAIR.json          # acquisition parameters
│   └── sub-1_FLAIR.nii.gz        # T2-FLAIR anatomical
├── dwi/
│   ├── sub-1_rec-ADC_dwi.json    # ADC parameters
│   ├── sub-1_rec-ADC_dwi.nii.gz  # ADC map
│   ├── sub-1_rec-TRACE_dwi.json  # TRACE/DWI parameters
│   └── sub-1_rec-TRACE_dwi.nii.gz # TRACE image (≈ DWI)
Derivatives (lesion masks):
derivatives/lesion_masks/sub-1/dwi/
├── sub-1_space-TRACE_desc-lesionAcute_mask.nii.gz   # ← USE THIS for training
├── sub-1_space-TRACE_desc-lesionChronic_mask.nii.gz  # ← USE as hard negative
└── sub-1_space-TRACE_desc-lesion_mask.nii.gz         # Combined (acute + chronic)
1.3 Critical Mapping: ISLES → SOOP
Your ISLES pipeline expects	SOOP equivalent	Notes
dwi.nii.gz	sub-X_rec-TRACE_dwi.nii.gz	TRACE ≈ DWI. Direct substitute.
adc.nii.gz	sub-X_rec-ADC_dwi.nii.gz	Same modality, different naming.
flair.nii.gz	sub-X_FLAIR.nii.gz	Located in anat/, not dwi/.
msk.nii.gz	sub-X_space-TRACE_desc-lesionAcute_mask.nii.gz	Use acute mask only for your task.
1.4 Subjects Without Stroke
266 subjects (1715 − 1449) have suspected stroke but no confirmed lesion. These are valuable negative examples — use them for training to reduce false positives. Check participants.tsv for confirmation status.
________________________________________
2. SOOP-Specific Preprocessing Pipeline
2.1 Step 0: Dataset Audit (DO THIS FIRST)
Before any preprocessing, you must audit the entire dataset. SOOP has high inter-subject variability in acquisition protocols.
What to audit per subject:
•	Which modalities are present (some subjects may lack FLAIR)
•	Voxel spacing from JSON sidecars (will vary significantly)
•	Image dimensions
•	Whether acute lesion mask exists and is non-empty
•	ADC value ranges (check the JSON for b-value and reconstruction method)
What to record:
•	Subjects with missing modalities → either exclude or handle with missing-channel strategy
•	Voxel spacing distribution → determines your target resampling resolution
•	Lesion volume distribution → determines your size-based stratification bins
•	Cases where acute mask is empty but chronic mask is not → these had old strokes only
2.2 Step 1: Co-registration (TRACE → FLAIR Space or Vice Versa)
Critical issue: In SOOP, the TRACE/ADC images are in DWI space while FLAIR is in anatomical space. These are not necessarily aligned.
The official SOOP processing pipeline (from the StrokeOutcomeOptimizationProjectDemo repo) co-registers TRACE → FLAIR using rigid registration, then normalizes FLAIR to MNI space. For your segmentation task, you have two options:
Option A — Work in native TRACE space (recommended for segmentation):
•	Keep TRACE and ADC as-is (they share the same space)
•	Register FLAIR → TRACE space using rigid registration (mutual information metric)
•	Lesion masks are already in TRACE space — no transformation needed
•	Pro: No interpolation artifacts on your primary imaging modality (DWI/TRACE is what shows acute lesions most clearly)
•	Pro: Ground truth masks don't need resampling
Option B — Work in FLAIR space:
•	Register TRACE and ADC → FLAIR space
•	Lesion masks must also be transformed (nearest-neighbor interpolation)
•	Pro: FLAIR typically has higher in-plane resolution
•	Con: Introduces interpolation on the most critical modality
Registration tool: Use SimpleITK rigid registration with mutual information. Do NOT use affine or deformable — you want to preserve anatomy.
2.3 Step 2: Skull Stripping
Use SynthStrip (recommended — it's what the ISLES'24 winning solution used) or HD-BET.
•	Apply skull stripping on the FLAIR (best tissue contrast available)
•	Propagate the brain mask to all other co-registered modalities
•	Apply brain mask to lesion masks as well (sanity check — lesion should be within brain)
Why this matters more for SOOP than ISLES: SOOP is clinical data from a comprehensive stroke center, so image quality is more variable. Extra-cranial tissue and artifacts are more common.
2.4 Step 3: Intensity Normalization (Per-Modality)
TRACE (DWI) Normalization:
•	Compute brain-only statistics (use the skull-stripped mask)
•	Clip to [1st percentile, 99th percentile] within the brain mask
•	Z-score normalize: (x - mean) / std using brain-only voxels
•	Rationale: TRACE intensity is scanner-dependent. Z-score preserves the relative hyperintensity of lesions.
ADC Normalization:
•	Read the JSON sidecar first — SOOP ADC values may use different units/scales across subjects depending on the scanner and reconstruction
•	If values are in standard units (×10⁻⁶ mm²/s): clip to [0, 3000], scale to [0, 1]
•	If values are raw scanner units: compute brain-only z-score instead
•	Critical check: For a few subjects, manually verify that lesion regions show LOW ADC (restricted diffusion). If they show HIGH ADC, the values may be inverted or using a different convention.
FLAIR Normalization:
•	Clip to [1st percentile, 99th percentile] within brain mask
•	Z-score normalize
•	Same rationale as TRACE
2.5 Step 4: Spatial Resampling
Target resolution decision:
•	Analyze the median voxel spacing across all subjects from the JSON sidecars
•	SOOP clinical scans typically have anisotropic voxels (e.g., 1.0 × 1.0 × 5.0 mm for diffusion)
•	Recommended: Resample to 1.0 × 1.0 × 1.0 mm isotropic (same as your ISLES pipeline)
Interpolation:
•	Images: trilinear (or B-spline order 3)
•	Masks: nearest-neighbor (preserves binary labels)
Warning for SOOP: Some subjects may have very thick slices (6-7mm in z). Resampling these to 1mm isotropic creates interpolated data in between slices — the model needs to learn that these inter-slice voxels are less reliable. Consider:
•	Adding a slice-thickness channel as an auxiliary input (binary map indicating original vs. interpolated slices)
•	Or being conservative and resampling to the dataset's median z-spacing instead of forcing 1mm
2.6 Step 5: Cropping/Padding
•	Compute brain bounding box for each subject after skull stripping
•	Analyze the 95th percentile of bounding box dimensions
•	Pad/crop to a fixed size (e.g., 192 × 192 × 192 or 160 × 192 × 160)
•	Center the crop on the brain centroid
•	Zero-pad smaller images
2.7 Step 6: Quality Control
Automated QC checks:
•	Verify TRACE-ADC consistency: In the lesion mask region, TRACE should be hyperintense (above brain mean) and ADC should be hypointense (below brain mean). Flag subjects that violate this.
•	Verify brain mask covers the entire brain (no clipping)
•	Verify lesion mask is within brain mask
•	Check for empty lesion masks that shouldn't be empty
Visual QC:
•	Generate a montage image per subject (similar to the SOOP demo's bids_bitmaps.py): overlay lesion mask on TRACE, show corresponding ADC and FLAIR slices
•	Manually review flagged subjects
________________________________________
3. Cascaded Detection → Segmentation for Tiny Lesions
3.1 Why Cascade?
Your original pipeline treats all lesions equally. The problem: for lesions <5ml (especially <1ml), the foreground-to-background ratio in a full-brain volume is catastrophically low (often <0.01%). The model has almost no gradient signal from these lesions.
A cascade solves this by:
1.	Stage 1 (Detection): Find approximate lesion locations at low resolution — a much easier binary task
2.	Stage 2 (Segmentation): Zoom in and segment precisely at full resolution — much better class balance
3.2 Stage 1: Lesion Detection Network
Architecture: Lightweight 3D CNN (e.g., small SegResNet or ResNet-18 adapted for 3D)
Input: Downsampled full-brain volume (e.g., 96 × 96 × 96 at 2mm isotropic), all channels concatenated (TRACE + ADC + FLAIR = 3 channels)

Output options — pick one:
Option A — Patch-level classification (simpler):
•	Divide the brain into overlapping 3D patches (e.g., 32³ at 2mm resolution, stride 16)
•	Binary classification per patch: "contains lesion?" (yes/no)
•	At inference, patches classified as positive define the ROIs
•	Pro: Very fast, easy to train. Con: Coarse localization.
Option B — Coarse segmentation (recommended):
•	Full 3D segmentation at low resolution (2mm isotropic)
•	Output: probability heatmap at 2mm resolution
•	Threshold at a LOW value (e.g., 0.2) — you want high recall, low precision is OK
•	Connected component analysis → bounding boxes around each detected region
•	Pro: More precise ROIs, natural integration with Stage 2. Con: Slightly more complex.
Training strategy for Stage 1:
•	Optimize for RECALL, not Dice. Use a loss function that heavily penalizes false negatives: 
o	Binary cross-entropy with high positive class weight (e.g., 10:1)
o	Or Focal Loss with α=0.9 (focus on the minority class)
•	NOT Dice loss — Dice loss tends to ignore very small objects
•	Data augmentation: standard spatial + intensity augmentations
•	Target metric: >95% lesion detection rate at the case level
3.3 Stage 2: Precision Segmentation Network
Architecture: Your existing models (nnU-Net, Swin-UNETR, SegResNet) — but now operating on cropped ROIs
Input: Cropped ROI at full resolution (1mm isotropic)
•	ROI size: fixed (e.g., 64³ or 96³ depending on GPU memory)
•	Center the crop on each detected region from Stage 1
•	Add generous margins: expand the Stage 1 bounding box by 50-100% in each direction before cropping
•	If a lesion is larger than the crop size, use sliding window with overlap (same as your current approach)
Key design decisions:
•	Multiple ROIs per subject: If Stage 1 detects 3 separate regions, run Stage 2 on each independently, then stitch predictions back into full-brain space
•	Overlapping ROIs: If two detected regions overlap after margin expansion, merge them into one larger ROI
•	Stitching strategy: Use Gaussian weighting in overlap regions (center of each crop has higher weight than edges)
Training strategy for Stage 2:
•	Train on cropped patches centered on ground-truth lesion locations (with random jitter for robustness)
•	Also include negative patches (no lesion) at a ratio of ~30-40% to prevent the model from predicting "lesion everywhere"
•	Standard Dice + CE loss works well here because class balance is much better after cropping
•	Can use your existing ensemble approach (nnU-Net + Swin-UNETR + SegResNet) on the cropped ROIs
3.4 Cascade Training Protocol
Important: Train Stage 1 and Stage 2 independently, not end-to-end.
1.	Train Stage 1 on full-resolution data (downsampled at input)
2.	Run Stage 1 inference on the training set itself → collect detected ROIs
3.	Use those ROIs (plus ground-truth ROIs for any missed lesions) to generate Stage 2 training crops
4.	Train Stage 2 on the cropped data
At inference:
Input Image → Preprocessing → Stage 1 (low-res detection)
                                    ↓
                              Detected ROIs
                                    ↓
                   Crop full-res patches around each ROI
                                    ↓
                          Stage 2 (full-res segmentation per ROI)
                                    ↓
                    Stitch predictions → Post-processing → Output
3.5 Handling Edge Cases in the Cascade
•	Stage 1 misses a lesion (false negative): This is the cascade's critical failure mode. Mitigate by:
o	Using very low detection threshold (0.1-0.2)
o	Ensemble multiple detection models
o	Adding a "safety net": also run a lightweight full-brain segmentation and union the results
•	Stage 1 produces many false positive ROIs: Not a problem — Stage 2 will reject them (predicts no lesion in that crop). The computational cost is manageable since Stage 2 runs on small crops.
•	Lesion spans multiple ROIs: Merge overlapping/adjacent ROIs before passing to Stage 2. After stitching, run connected component analysis to ensure continuity.
________________________________________
4. Lesion-Aware Patching Strategy
4.1 The Problem with Uniform Random Patching
If you randomly sample 128³ patches from a 192³ brain volume, the probability that any given patch contains a 0.5ml lesion is very low. Most training iterations see only background — the model learns to predict "no lesion" and gets rewarded for it.
4.2 Foreground-Biased Sampling (Basic)
MONAI's RandCropByPosNegLabeld: This is your starting point.
•	pos = fraction of patches centered on lesion voxels
•	neg = fraction of patches centered on non-lesion brain voxels
•	Recommended ratio: pos=0.7, neg=0.3 (70% of patches contain lesion)
How it works internally:
1.	Randomly decide: positive sample (70%) or negative sample (30%)
2.	If positive: randomly pick a voxel from the lesion mask, center the patch there
3.	If negative: randomly pick a voxel from the brain mask (excluding lesion), center the patch there
4.3 Adaptive Multi-Scale Patching (Advanced — for tiny lesions)
Go beyond simple pos/neg sampling with a size-aware strategy:
Bin your lesions by volume:
•	Tiny: <1ml
•	Small: 1-5ml
•	Medium: 5-50ml
•	Large: >50ml
Adaptive patch size per bin:
Lesion Size	Patch Size	Rationale
Tiny (<1ml)	64³	Zoom in — lesion fills more of the patch
Small (1-5ml)	96³	Moderate zoom
Medium (5-50ml)	128³	Standard
Large (>50ml)	160³ or 192³	Need context for boundaries
During training:
1.	Randomly select a training subject
2.	Check its lesion volume → select the appropriate patch size
3.	Sample a foreground-centered patch at that size
4.	Resize all patches to a common size (e.g., 128³) before batching
Why this works: A 0.3ml lesion in a 64³ patch at 1mm isotropic occupies ~0.7% of the volume. In a 192³ patch, it occupies ~0.004%. That's a 175× improvement in class balance.
4.4 Hard Example Mining During Training
Maintain a difficulty score per training subject:
difficulty[subject_id] = 1.0 - dice_score_on_last_validation
During training, sample subjects with probability proportional to their difficulty:
•	Subjects where the model performs worst get sampled more often
•	Subjects where the model already performs well get sampled less
•	Re-compute difficulty scores every N epochs (e.g., every 50 epochs)
This naturally focuses training on the hardest cases (often the tiny lesions).
4.5 Combining Cascade + Lesion-Aware Patching
These two strategies are complementary:
•	Stage 1 (detection): Use foreground-biased sampling (70/30) with standard patch size on downsampled data
•	Stage 2 (segmentation): Use adaptive multi-scale patching on the cropped ROIs 
o	For tiny detected lesions: smaller crops, higher foreground ratio (80/20)
o	For large detected lesions: larger crops, moderate foreground ratio (60/40)
________________________________________
5. Key Notes & Gotchas
5.1 SOOP-Specific Gotchas
1.	Acute vs. Chronic masks: Always use desc-lesionAcute for training your acute stroke segmentation. The desc-lesion (combined) mask includes chronic lesions that do NOT show restricted diffusion on DWI — training on these will confuse your model.
2.	Chronic masks as hard negatives: Use desc-lesionChronic masks during training — add them as regions where the model should predict "no acute lesion" even though there IS signal abnormality on FLAIR. This teaches the model the DWI-ADC physics.
3.	Missing modalities: Not all 1715 subjects will have all three modalities. Before training, create a manifest of which modalities exist per subject. Options:
o	Exclude subjects with missing channels (safest)
o	Zero-fill missing channels (works if <10% are missing)
o	Train with channel dropout augmentation (randomly zero one channel during training) so the model learns to work without any single modality
4.	Variable acquisition parameters: Each subject's JSON sidecar may show different TE, TR, b-values, slice thickness, matrix size. This is a feature, not a bug — it makes your model more generalizable. But it means your preprocessing must be robust. Always read the JSON sidecar to know what you're dealing with.
5.	participants.tsv: Contains demographic data (age, NIHSS scores, stroke etiology). Use NIHSS for your stratified splitting — it correlates with lesion severity.
6.	No-stroke subjects (266 of 1715): These are negative controls. Include ~10-15% of them in each training batch as pure negative examples. This is extremely valuable for reducing false positives — ISLES 2022 didn't give you this.
5.2 DeepISLES Insights (Published August 2025)
The recently published DeepISLES paper is highly relevant to your work:
•	DeepISLES is an ensemble of SEALS (nnU-Net based), NVAUTO (MONAI Auto3DSeg), and SWAN (FACTORIZER) — similar philosophy to your three-model approach
•	They use majority voting for ensembling (simpler than your learned fusion)
•	They validated on N=1685 external cases (likely including SOOP data) and outperformed individual models by significant margins
•	The tool works in native image space with no external preprocessing required
•	Key takeaway: preprocessing robustness matters more than model architecture complexity
5.3 Loss Function Recommendations for Tiny Lesions
Your document mentions DiceCE loss. For tiny lesions, consider these alternatives:
•	Generalized Dice Loss (GDL): Weights each class inversely by its volume. Naturally handles extreme class imbalance. Use this instead of standard Dice.
•	Dice + Focal Loss: Focal loss down-weights easy negatives and focuses on hard examples (the boundary voxels and tiny lesions)
•	Boundary-aware loss (e.g., Hausdorff distance loss): Add as auxiliary loss to improve boundary delineation for small lesions
•	Compound loss: 0.4 × GDL + 0.4 × Focal + 0.2 × Boundary — covers volume overlap, hard example mining, and boundary accuracy
5.4 Data Splitting for SOOP
With 1449 stroke-confirmed subjects + 266 negative controls, you have much more data than ISLES 2022. Recommended split:
•	Train: 70% (~1015 stroke + ~186 negative = ~1201)
•	Validation: 15% (~217 stroke + ~40 negative = ~257)
•	Test: 15% (~217 stroke + ~40 negative = ~257)
Stratify by:
1.	Lesion volume bins (tiny/small/medium/large)
2.	Presence of chronic lesions (yes/no) — these are harder cases
3.	NIHSS score quartiles
4.	Stroke etiology (if available in participants.tsv)
5.5 Augmentation Adjustments for SOOP
Because SOOP already has high natural variation (multi-protocol, clinical data), you can use lighter augmentation than you would for ISLES 2022:
•	Reduce elastic deformation intensity (the real data already has registration imperfections)
•	Keep intensity augmentations (Gaussian noise, gamma correction) — these remain important
•	Add channel dropout (randomly zero one modality with 10-15% probability) to handle missing data
•	Add resolution simulation: randomly downsample-then-upsample one axis by 2× to simulate thick-slice acquisitions
5.6 Physics-Based Post-Processing (Revised for SOOP)
Your Phase 6 physics validation becomes even more important with SOOP because chronic lesions coexist with acute ones:
Enhanced physics check:
Is_acute_lesion = (TRACE > TRACE_threshold)     # hyperintense on TRACE/DWI
                AND (ADC < ADC_threshold)        # hypointense on ADC
                AND NOT (chronic_lesion_region)   # not in known chronic territory
If you have the chronic mask available at inference (you won't in real deployment, but you do during training/validation), use it to verify your model isn't confusing chronic for acute.
________________________________________
6. Updated Phase-by-Phase Checklist
Phase 0: Setup
•	[ ] Download SOOP dataset from OpenNeuro (ds004889 v1.1.2)
•	[ ] Download participants.tsv
•	[ ] Set up BIDS-aware data loading (use PyBIDS or custom BIDS parser)
•	[ ] Audit all subjects: modalities present, voxel spacings, lesion volumes
•	[ ] Generate dataset statistics report
•	[ ] Review the StrokeOutcomeOptimizationProjectDemo repo for reference preprocessing
Phase 1: Preprocessing
•	[ ] Implement BIDS-aware file discovery (not hardcoded paths)
•	[ ] Co-register FLAIR → TRACE space (rigid registration)
•	[ ] Skull strip using SynthStrip
•	[ ] Intensity normalize per-modality (check ADC units from JSON)
•	[ ] Resample to target isotropic resolution
•	[ ] Crop/pad to fixed dimensions
•	[ ] Run automated QC checks
•	[ ] Visual QC review of flagged cases
•	[ ] Split into train/val/test with stratification
•	[ ] Save split as JSON
Phase 2: Stage 1 — Detection Network
•	[ ] Prepare downsampled (2mm) full-brain volumes
•	[ ] Train detection model optimized for recall
•	[ ] Validate detection rate >95% on validation set
•	[ ] Generate ROI bounding boxes for all training subjects
Phase 3: Stage 2 — Segmentation Models
•	[ ] Generate cropped ROI training data from Stage 1 outputs
•	[ ] Implement adaptive multi-scale patching
•	[ ] Train nnU-Net on cropped ROIs (use nnU-Net's built-in pipeline)
•	[ ] Train Swin-UNETR on cropped ROIs (sensitivity-focused)
•	[ ] Train SegResNet on cropped ROIs (stability-focused)
•	[ ] Validate each model independently on cropped validation ROIs
Phase 4: Ensemble & Fusion
•	[ ] Implement cascade inference pipeline (Stage 1 → crop → Stage 2)
•	[ ] Test simple ensemble averaging on full pipeline
•	[ ] Test majority voting (DeepISLES approach)
•	[ ] Optionally train learned fusion network
•	[ ] Compare all ensemble strategies on validation set
Phase 5: Post-Processing
•	[ ] Implement physics-aware validation (DWI-ADC consistency)
•	[ ] Implement small component removal
•	[ ] Implement morphological closing
•	[ ] Tune thresholds on validation set
•	[ ] Verify chronic lesions are correctly excluded
Phase 6: Evaluation
•	[ ] Full pipeline evaluation on held-out test set
•	[ ] Stratified analysis by lesion size (tiny/small/medium/large)
•	[ ] Ablation studies: cascade vs. single-stage, ensemble vs. single model
•	[ ] Compare against DeepISLES as baseline (run their Docker on your test set)
•	[ ] Failure mode analysis with visualization
________________________________________
Appendix A: Key References
1.	SOOP Dataset Paper: Absher et al. (2024). "The stroke outcome optimization project." Scientific Data, 11(1), 839.
2.	SOOP Processing Demo: https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo
3.	DeepISLES (2025): de la Rosa et al. "DeepISLES: a clinically validated ischemic stroke segmentation model." Nature Communications 16, 7357.
4.	ISLES'24 Winner — Preprocessing Focus: Heras Rivera et al. (2025). "How We Won the ISLES'24 Challenge by Preprocessing." MIDL 2025.
5.	Small Lesion Detection: Liu et al. (2021). "Deep learning-based detection and segmentation of diffusion abnormalities." Communications Medicine.
Appendix B: Quick Sanity Checks Before Training
Run these checks before committing to a multi-day training run:
1.	Visualize 10 random preprocessed subjects — overlay lesion mask on TRACE. Does the mask align with the hyperintense region?
2.	Check ADC in lesion ROI: Mean ADC in lesion should be LOWER than mean ADC in contralateral brain. If it's higher, your ADC normalization or registration is wrong.
3.	Verify no data leakage: Ensure no subject ID appears in more than one split.
4.	Check patch sampling: Visualize 20 random training patches. Do ~70% contain visible lesion? Are tiny lesions actually visible in the patches?
5.	Run 10-epoch sanity check: Train for 10 epochs. Loss should decrease. If Dice stays at 0.0 for all 10 epochs, something is fundamentally wrong (usually data loading or label mismatch).

________________________________________
Appendix C: Bugs Found and Fixed During Training (2026-04-05)

### Bug 1: Wrong derivatives path (CRITICAL — wasted 4.5 hours)

**File:** `01_preprocess_soop.py` line 431
**Symptom:** All 1715 masks saved as zeros. Training showed Dice=0.000 for 50 epochs.
**Root cause:** `derivatives_dir = raw_dir / "derivatives"` resolved to `/home/ubuntu/data/SOOP_Dataset/raw/derivatives/` which doesn't exist. Masks are at `/home/ubuntu/data/SOOP_Dataset/derivatives/`.
**Fix:** Read derivatives path from config: `derivatives_dir = Path(config["paths"].get("derivatives", str(raw_dir.parent / "derivatives")))`
**Lesson:** Always verify a sample output after preprocessing before starting training. A 10-second check (`mask_sum > 0?`) would have caught this immediately.

### Bug 2: Focal Loss numerical instability (CRITICAL)

**Files:** `04_train_stage1_detection.py`, `07_train_stage2_segresnet.py`, `08_train_stage2_swin_unetr.py`
**Symptom:** Training loss explodes to negative trillions (e.g., -28035721801106111233785856.0)
**Root cause:** `p_t = torch.exp(-bce)` overflows when logits are large. BCE can be very large for confident wrong predictions, causing `exp(-large)` → 0, then `(1-0)^2 * large_bce` → huge number.
**Fix:** Clamp predictions to [-20, 20] and compute p_t via sigmoid: `p_t = sigmoid(pred) * target + (1 - sigmoid(pred)) * (1 - target)`
**Lesson:** Always clamp logits in custom loss functions. Test loss functions with extreme input values before training.

### Bug 3: Broken import in Swin-UNETR script

**File:** `08_train_stage2_swin_unetr.py` line 33
**Symptom:** Script crashes on startup with ModuleNotFoundError
**Root cause:** `from train_stage2_segresnet_imports import ...` — module doesn't exist. Fallback `except NameError` doesn't catch `ModuleNotFoundError`.
**Fix:** Changed to `except (ImportError, ModuleNotFoundError)`

### Bug 4: DiceMetric returns NaN

**Files:** `04_train_stage1_detection.py`, `07_train_stage2_segresnet.py`
**Symptom:** Validation Dice is NaN or 0.0 even when model is clearly learning
**Root cause:** `DiceMetric(include_background=False)` with single-channel output skips the only channel available, returning NaN.
**Fix:** Convert to 2-channel one-hot before passing to DiceMetric: `pred_oh = cat([1-pred, pred], dim=1)`

### Bug 5: Stage 2 crops only generated for train split

**File:** `06_prepare_stage2_crops.py`
**Symptom:** Stage 2 training crashes with `FileNotFoundError: manifest.json` for val split
**Root cause:** Script defaults to `--split train`. Must be run separately for train, val, and test.
**Fix:** Run with `--split val` and `--split test` explicitly. Documented in LAMBDA_SOOP_SETUP.md.

### General Lessons

1. **Always test one subject end-to-end before full pipeline run.** A single-subject test in preprocessing would have caught Bug 1 in seconds instead of 4.5 hours.
2. **Check mask sums after preprocessing.** `mask_sum=0` for all subjects is an immediate red flag.
3. **Test loss functions with extreme values.** Focal loss, GDL, and other custom losses can have numerical edge cases.
4. **Print loss values from epoch 1.** If loss is 0.0000 or NaN at epoch 1, something is fundamentally wrong — don't wait 50 epochs hoping it improves.
5. **Verify file discovery paths on the actual server.** Local paths and server paths may have different directory structures.

________________________________________
Appendix D: Clinical Deployment Architecture

See `ARCHITECTURE.md` for the full dual-path clinical deployment design including:
- Cascade pipeline (Path A) for fast, focused segmentation
- Full-volume nnU-Net safety net (Path B) to catch missed lesions
- Comparator module for confidence scoring
- Clinical output format with confidence levels (HIGH/MEDIUM/LOW/CLEAR/REVIEW)

