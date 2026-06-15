#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Sentinel Stroke - Lambda Labs Full Prediction Pipeline
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS DOES:
#   1. Sets up the environment on Lambda Labs
#   2. Organizes your test data into nnU-Net format
#   3. Runs ALL 15 base models (5 nnU-Net + 5 Swin + 5 SegResNet)
#   4. Runs Fusion network on top
#   5. Saves final predictions + visualizations
#
# HOW TO USE:
#   1. Launch a Lambda Labs GPU instance (A100 recommended)
#   2. Upload your project:  scp -r stroke_mvp/ ubuntu@<LAMBDA_IP>:/home/ubuntu/
#   3. SSH in:               ssh ubuntu@<LAMBDA_IP>
#   4. Run this script:      cd /home/ubuntu/stroke_mvp && bash lambda_setup_and_predict.sh
#
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "════════════════════════════════════════════"
echo "  Sentinel Stroke - Lambda Labs Setup"
echo "════════════════════════════════════════════"

# ─── Step 1: Install dependencies ───
echo "[1/6] Installing dependencies..."
pip install -q torch torchvision monai[all] nibabel SimpleITK \
    numpy scipy scikit-image scikit-learn pandas \
    PyYAML matplotlib seaborn tqdm rich nnunetv2

echo "[1/6] Done!"

# ─── Step 2: Setup directory structure ───
echo "[2/6] Setting up directories..."

PROJECT_DIR="/home/ubuntu/stroke_mvp"
NNUNET_RAW="/home/ubuntu/nnUNet/nnUNet_raw/Dataset001_ISLES2022"
NNUNET_PREPROCESSED="/home/ubuntu/nnUNet/nnUNet_preprocessed/Dataset001_ISLES2022"

# Create nnU-Net directory structure
mkdir -p "$NNUNET_RAW/imagesTr"
mkdir -p "$NNUNET_RAW/imagesTs"
mkdir -p "$NNUNET_RAW/labelsTr"
mkdir -p "$NNUNET_RAW/labelsTs"
mkdir -p "$PROJECT_DIR/test_results"
mkdir -p "$PROJECT_DIR/test_visualizations"

# Set nnU-Net environment variables
export nnUNet_raw="/home/ubuntu/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/nnUNet/nnUNet_results"

echo "[2/6] Done!"

# ─── Step 3: Check what's available ───
echo "[3/6] Checking available models and data..."

echo "  Trained models:"
for model_dir in segresnet swin_unetr nnunet; do
    count=$(find "$PROJECT_DIR/trained_models/$model_dir" -name "*.pth" 2>/dev/null | wc -l)
    echo "    $model_dir: $count checkpoints"
done

echo "  Fusion checkpoints:"
ls -la "$PROJECT_DIR/checkpoints/"*.pth 2>/dev/null || echo "    (none in checkpoints/)"
ls -la "$PROJECT_DIR/trained_models/fusion_network"*.pth 2>/dev/null || echo "    (none in trained_models/)"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE! Now put your test data in place."
echo "════════════════════════════════════════════════════════"
echo ""
echo "  See lambda_predict.py for next steps."
echo ""
