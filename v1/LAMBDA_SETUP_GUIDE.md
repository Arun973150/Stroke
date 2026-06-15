# Lambda Labs Setup Guide - Sentinel Stroke
# ==========================================

## Quick Reference

| Item | Value |
|------|-------|
| **Recommended GPU** | A100 (40GB) - $1.10/hr |
| **Project folder** | `~/sentinel_stroke/` |
| **Data folder** | `~/data/ISLES-2022/` |
| **Total cases** | 250 cases |
| **Estimated training time** | 8-10 hours (1 fold) |
| **Estimated cost** | ~$9-11 (1 fold) |

---

## STEP 1: Launch Lambda Labs Instance

1. Go to [cloud.lambdalabs.com](https://cloud.lambdalabs.com)
2. Click **Launch Instance**
3. Select: **1x NVIDIA A100 (40GB)**
4. Select: **Ubuntu 22.04 + PyTorch 2.0**
5. Add your SSH key (or create one)
6. Click **Launch**
7. Wait for instance to start (~1-2 min)
8. Copy the IP address

---

## STEP 2: Upload Data & Code (Windows PowerShell)

Open PowerShell on your Windows machine and run:

```powershell
# Replace <LAMBDA_IP> with your instance IP

# Step A: Create directories on Lambda
ssh ubuntu@<LAMBDA_IP> "mkdir -p ~/data ~/sentinel_stroke"

# Step B: Compress your project (faster upload)
cd "C:\Users\ADMIN\OneDrive\Desktop\stroke_mvp"
tar -cvzf stroke_project.tar.gz --exclude="ISLES-2022" *

# Step C: Upload project (small, fast)
scp stroke_project.tar.gz ubuntu@<LAMBDA_IP>:~/sentinel_stroke/

# Step D: Compress ISLES data
cd "C:\Users\ADMIN\OneDrive\Desktop\stroke_mvp\ISLES-2022"
tar -cvzf ISLES-2022.tar.gz ISLES-2022

# Step E: Upload data (~10-15 min depending on connection)
scp ISLES-2022.tar.gz ubuntu@<LAMBDA_IP>:~/data/
```

---

## STEP 3: SSH Into Lambda Labs

```powershell
ssh ubuntu@<LAMBDA_IP>
```

---

## STEP 4: Extract Files (On Lambda)

```bash
# Extract project
cd ~/sentinel_stroke
tar -xvzf stroke_project.tar.gz
rm stroke_project.tar.gz

# Extract data
cd ~/data
tar -xvzf ISLES-2022.tar.gz
rm ISLES-2022.tar.gz

# Verify data
ls ~/data/ISLES-2022/
# Should show: sub-strokecase0001/ sub-strokecase0002/ ... derivatives/
```

---

## STEP 5: Run Environment Setup (5 min)

```bash
cd ~/sentinel_stroke
bash scripts/setup_environment.sh
```

This will:
- Create conda environment
- Install PyTorch + CUDA
- Install nnU-Net v2
- Set up environment variables

**After setup, run:**
```bash
source ~/.bashrc
conda activate sentinel_stroke
```

---

## STEP 6: Verify GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should show:
```
GPU: NVIDIA A100-SXM4-40GB
CUDA available: True
```

---

## STEP 7: Start Training (Use Screen)

```bash
# Start a screen session (keeps running if SSH disconnects)
screen -S training

# Run preprocessing (~45 min)
cd ~/sentinel_stroke
python scripts/01_preprocess_isles.py

# Setup nnU-Net dataset
python scripts/02_setup_nnunet_dataset.py

# nnU-Net planning (~30 min)
python scripts/03_plan_and_preprocess.py

# Train fold 0 (~8-10 hours)
python scripts/04_train_nnunet.py --fold 0

# To detach from screen: Press Ctrl+A, then D
# To reattach later: screen -r training
```

---

## STEP 8: Monitor Training (Optional - New Terminal)

```bash
# SSH into Lambda from another terminal
ssh ubuntu@<LAMBDA_IP>

# Monitor GPU usage
watch -n 1 nvidia-smi

# OR use our monitoring script
cd ~/sentinel_stroke
python scripts/monitor_training.py --fold 0
```

---

## STEP 9: Download Results (After Training)

From Windows PowerShell:

```powershell
# Download trained model
scp -r ubuntu@<LAMBDA_IP>:~/nnUNet/nnUNet_results/ ./results_backup/

# Download checkpoints
scp -r ubuntu@<LAMBDA_IP>:~/sentinel_stroke/checkpoints/ ./checkpoints_backup/
```

---

## Troubleshooting

### SSH Permission Denied
```powershell
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add the public key (~/.ssh/id_ed25519.pub) to Lambda Labs
```

### Training Interrupted
```bash
# Resume from checkpoint
python scripts/04_train_nnunet.py --fold 0 --continue
```

### Out of Memory
```bash
# Check current GPU memory
nvidia-smi
# nnU-Net auto-adjusts batch size, but you can reduce manually in plans
```

---

## Cost Summary

| Phase | Time | Cost |
|-------|------|------|
| Setup + Upload | 30 min | $0.55 |
| Preprocess | 45 min | $0.83 |
| nnU-Net Planning | 30 min | $0.55 |
| Train 1 fold | 10 hours | $11.00 |
| **Total (1 fold)** | **~12 hours** | **~$13** |

**💡 Tip**: Train 1 fold first to validate, then train remaining folds.
