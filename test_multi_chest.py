"""
test_multi_chest.py
===================
Tests the trained autoencoder on multiple real chest DICOM slices.
Saves: test_output/multi_chest_result.png
"""
import sys, os, math
sys.path.insert(0, os.path.abspath("backend"))

import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recovery_engine import get_autoencoder, _device, _torch_available

OUT_DIR  = "test_output"
OUT_FILE = os.path.join(OUT_DIR, "multi_chest_result.png")
os.makedirs(OUT_DIR, exist_ok=True)

# Pick 4 different chest CT slices
DICOM_FOLDER = os.path.join("dicom_data", "1_3_6_1_4_1_14519_5_2_1_99_1071_33780533")
SLICES = ["slice_0020.dcm", "slice_0040.dcm", "slice_0060.dcm", "slice_0080.dcm"]

def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 20 * math.log10(1.0) - 10 * math.log10(mse)

def load_dicom(path):
    ds  = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    norm = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
    return cv2.resize(norm, (224, 224))

def corrupt(img):
    c = img.copy()
    c += np.random.randn(*c.shape).astype(np.float32) * 0.12
    h, w = c.shape
    c[h//4:h//4+h//4, w//4:w//4+w//4] = 0.0         # block erase
    for row in np.random.choice(h, 6, replace=False):  # scan-line dropout
        c[row, :] = 0.0
    return np.clip(c, 0, 1)

# Load model
print(f"Device: {_device}")
model = get_autoencoder()
print("Model loaded OK\n")

# Setup plot: 4 rows x 3 cols (Original | Corrupted | Recovered)
fig, axes = plt.subplots(len(SLICES), 3, figsize=(14, 4.5 * len(SLICES)))
fig.patch.set_facecolor("#0a0a18")

col_labels = ["Original (Clean)", "Corrupted", "Autoencoder Recovered"]
col_colors = ["#00e5ff", "#ff4444", "#00ff88"]

for row, fname in enumerate(SLICES):
    path    = os.path.join(DICOM_FOLDER, fname)
    resized = load_dicom(path)
    corrupt_img = corrupt(resized)

    inp_t   = torch.from_numpy(corrupt_img).unsqueeze(0).unsqueeze(0).float().to(_device)
    clean_t = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(_device)
    with torch.no_grad():
        out_t = model(inp_t)
    recovered = out_t.squeeze().cpu().numpy()

    psnr_c = psnr(inp_t, clean_t)
    psnr_r = psnr(out_t, clean_t)
    gain   = psnr_r - psnr_c

    print(f"{fname}  |  Corrupted: {psnr_c:.1f} dB  |  Recovered: {psnr_r:.1f} dB  |  Gain: +{gain:.1f} dB")

    images = [resized, corrupt_img, recovered]
    titles = [
        f"{fname}\n(Original)",
        f"Corrupted\n{psnr_c:.1f} dB",
        f"Recovered  +{gain:.1f} dB\n{psnr_r:.1f} dB"
    ]

    for col, (img, title, color) in enumerate(zip(images, titles, col_colors)):
        ax = axes[row][col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

# Column headers
for col, (label, color) in enumerate(zip(col_labels, col_colors)):
    axes[0][col].set_title(
        f"{label}\n{axes[0][col].get_title()}",
        color=color, fontsize=11, fontweight="bold", pad=6
    )

fig.suptitle(
    "PhantomaShield — Autoencoder Chest CT Recovery Test (4 slices)",
    color="white", fontsize=15, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nSaved: {OUT_FILE}")
