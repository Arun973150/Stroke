import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Edit these two paths:
dwi_path  = ROOT / "ISLES-2022/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz"
pred_path = ROOT / "stroke_result_case0001.nii.gz"  # your saved output

dwi = nib.load(dwi_path).get_fdata()
pred = nib.load(pred_path).get_fdata()

z = dwi.shape[2] // 2  # middle axial slice
img = dwi[:, :, z]
mask = pred[:, :, z] > 0.5

plt.figure(figsize=(6,6))
plt.imshow(img.T, cmap="gray", origin="lower")
plt.imshow(np.ma.masked_where(~mask.T, mask.T), cmap="autumn", alpha=0.5, origin="lower")
plt.axis("off"); plt.tight_layout()

out = ROOT / "results" / "visualizations" / "case0001_overlay.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200)
print(f"Saved {out}")