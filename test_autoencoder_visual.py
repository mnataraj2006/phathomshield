"""
test_autoencoder_visual.py
==========================
Tests the trained autoencoder on a real DICOM file from dicom_data/.
Shows: Original | Corrupted | Autoencoder Recovered
Saves output to: test_output/autoencoder_result.png
"""
import sys, os, math
sys.path.insert(0, os.path.abspath("backend"))

import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to file
import matplotlib.pyplot as plt

from recovery_engine import get_autoencoder, _device, _torch_available

# ── Config ────────────────────────────────────────────────────────────────────
DICOM_PATH = os.path.join(
    "dicom_data",
    "1_3_6_1_4_1_14519_5_2_1_99_1071_33780533",
    "slice_0010.dcm"
)
OUT_DIR  = "test_output"
OUT_FILE = os.path.join(OUT_DIR, "autoencoder_result.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── PSNR helper ───────────────────────────────────────────────────────────────
def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 20 * math.log10(1.0) - 10 * math.log10(mse)

# ── Load DICOM ────────────────────────────────────────────────────────────────
print(f"Loading DICOM: {DICOM_PATH}")
ds  = pydicom.dcmread(DICOM_PATH, force=True)
arr = ds.pixel_array.astype(np.float32)
if arr.ndim == 3:
    arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
print(f"  Shape: {arr.shape}  dtype: {arr.dtype}  min={arr.min():.0f}  max={arr.max():.0f}")

# ── Normalize [0, 1] ──────────────────────────────────────────────────────────
p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
norm    = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
resized = cv2.resize(norm, (224, 224))

# ── Apply synthetic corruption ────────────────────────────────────────────────
def corrupt(img: np.ndarray) -> np.ndarray:
    c = img.copy()
    c += np.random.randn(*c.shape).astype(np.float32) * 0.12   # Gaussian noise
    h, w = c.shape
    rh, rw = h // 4, w // 4                                     # block erase
    c[h//4 : h//4+rh, w//4 : w//4+rw] = 0.0
    for row in np.random.choice(h, 5, replace=False):            # scan-line dropout
        c[row, :] = 0.0
    return np.clip(c, 0, 1)

corrupted = corrupt(resized)

# ── Run autoencoder ───────────────────────────────────────────────────────────
print(f"\nPyTorch available : {_torch_available}")
print(f"Device            : {_device}")

model = get_autoencoder()
if model is None:
    print("ERROR: Model not loaded!")
    sys.exit(1)

print("Model loaded OK")

inp_t   = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).float().to(_device)
clean_t = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(_device)

with torch.no_grad():
    out_t = model(inp_t)

recovered = out_t.squeeze().cpu().numpy()

# ── Metrics ───────────────────────────────────────────────────────────────────
psnr_corrupted = psnr(inp_t, clean_t)
psnr_recovered = psnr(out_t, clean_t)
improvement    = psnr_recovered - psnr_corrupted

print(f"\n{'='*50}")
print(f"  PSNR (corrupted) : {psnr_corrupted:.2f} dB")
print(f"  PSNR (recovered) : {psnr_recovered:.2f} dB")
print(f"  Improvement      : +{improvement:.2f} dB")
print(f"{'='*50}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0f0f1a")

titles = [
    "Original (Clean)",
    f"Corrupted\nPSNR: {psnr_corrupted:.1f} dB",
    f"Autoencoder Recovered\nPSNR: {psnr_recovered:.1f} dB  (+{improvement:.1f} dB)"
]
images  = [resized, corrupted, recovered]
colors  = ["#00e5ff", "#ff4444", "#00ff88"]

for ax, img, title, color in zip(axes, images, titles, colors):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

fig.suptitle(
    "PhantomaShield — Autoencoder Recovery Test\n"
    f"DICOM: {os.path.basename(DICOM_PATH)}",
    color="white", fontsize=15, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nResult saved to : {OUT_FILE}")
