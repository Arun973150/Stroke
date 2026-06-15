#!/bin/bash
# =============================================================================
# Sentinel Stroke - Environment Setup Script
# =============================================================================
# Run this script on Lambda Labs to set up the complete environment
# Usage: bash scripts/setup_environment.sh

set -e  # Exit on error

echo "=============================================================="
echo "Sentinel Stroke - Environment Setup"
echo "=============================================================="

# -----------------------------------------------------------------------------
# 1. Create Conda Environment
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Creating Conda environment..."
conda create -n sentinel_stroke python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate sentinel_stroke

echo "✓ Conda environment 'sentinel_stroke' created and activated"

# -----------------------------------------------------------------------------
# 2. Install PyTorch with CUDA
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✓ PyTorch installed"

# -----------------------------------------------------------------------------
# 3. Install nnU-Net v2
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing nnU-Net v2..."
pip install nnunetv2

echo "✓ nnU-Net v2 installed"

# -----------------------------------------------------------------------------
# 4. Install remaining dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing other dependencies..."
pip install -r requirements.txt

echo "✓ All dependencies installed"

# -----------------------------------------------------------------------------
# 5. Create directory structure
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Creating directory structure..."

# Data directories
mkdir -p /home/ubuntu/data/ISLES-2022
mkdir -p /home/ubuntu/data/preprocessed

# nnU-Net directories
mkdir -p /home/ubuntu/nnUNet/nnUNet_raw
mkdir -p /home/ubuntu/nnUNet/nnUNet_preprocessed
mkdir -p /home/ubuntu/nnUNet/nnUNet_results

# Project directories
mkdir -p /home/ubuntu/sentinel_stroke/logs/tensorboard
mkdir -p /home/ubuntu/sentinel_stroke/checkpoints/nnunet
mkdir -p /home/ubuntu/sentinel_stroke/results

echo "✓ Directory structure created"

# -----------------------------------------------------------------------------
# 6. Set environment variables
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Setting environment variables..."

# Add to .bashrc
cat >> ~/.bashrc << 'EOF'

# Sentinel Stroke - nnU-Net Environment Variables
export nnUNet_raw="/home/ubuntu/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/nnUNet/nnUNet_results"

# Activate environment on login
conda activate sentinel_stroke
EOF

# Set for current session
export nnUNet_raw="/home/ubuntu/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/nnUNet/nnUNet_results"

echo "✓ Environment variables set"

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Verifying installation..."
echo "=============================================================="

# Check Python
python --version

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check nnU-Net
python -c "import nnunetv2; print(f'nnU-Net v2: OK')"

# Check MONAI
python -c "import monai; print(f'MONAI: {monai.__version__}')"

echo ""
echo "=============================================================="
echo "✓ SETUP COMPLETE!"
echo "=============================================================="
echo ""
echo "Environment variables:"
echo "  nnUNet_raw:          $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"  
echo "  nnUNet_results:      $nnUNet_results"
echo ""
echo "Next steps:"
echo "  1. Upload ISLES-2022 data to: /home/ubuntu/data/ISLES-2022/"
echo "  2. Run preprocessing: python scripts/01_preprocess_isles.py"
echo "  3. Setup nnU-Net dataset: python scripts/02_setup_nnunet_dataset.py"
echo "  4. Plan and preprocess: python scripts/03_plan_and_preprocess.py"
echo "  5. Start training: python scripts/04_train_nnunet.py --fold 0"
echo ""
