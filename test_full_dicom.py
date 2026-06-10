"""
test_full_dicom.py
==================
Comprehensive test of the updated BRaTs2021-trained autoencoder.
Tests across all available local DICOM series with varied corruption types.
Saves:  test_output/full_test_result.png
        test_output/full_test_summary.txt
"""
import sys, os, math
sys.path.insert(0, os.path.abspath("backend"))

import numpy as np
import cv2, pydicom, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recovery_engine import get_autoencoder, _device

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Device : {_device}")
model = get_autoencoder()
print("Model  : loaded OK\n")

# ── Collect DICOMs from all series ────────────────────────────────────────────
DICOM_ROOT = "dicom_data"
test_files = []
for series in os.listdir(DICOM_ROOT):
    series_path = os.path.join(DICOM_ROOT, series)
    if not os.path.isdir(series_path):
        continue
    slices = sorted([f for f in os.listdir(series_path) if f.endswith(".dcm")])
    # Pick 2 slices per series — one from start, one from middle
    picks = [slices[10], slices[len(slices)//2]] if len(slices) > 20 else slices[:2]
    for s in picks:
        test_files.append((series[:30], os.path.join(series_path, s)))

print(f"Testing {len(test_files)} DICOM slices from {len(os.listdir(DICOM_ROOT))} series\n")

# ── Corruption types (varied per image) ───────────────────────────────────────
def corrupt_gaussian(img):
    c = img.copy() + np.random.randn(*img.shape).astype(np.float32) * 0.15
    return np.clip(c, 0, 1)

def corrupt_block(img):
    c = img.copy()
    h, w = c.shape
    c[h//4:h//2, w//4:w//2] = 0.0
    return np.clip(c, 0, 1)

def corrupt_scanline(img):
    c = img.copy()
    h = c.shape[0]
    for row in np.random.choice(h, 12, replace=False):
        c[row, :] = 0.0
    return np.clip(c, 0, 1)

def corrupt_combined(img):
    c = img.copy()
    c += np.random.randn(*img.shape).astype(np.float32) * 0.12
    h, w = c.shape
    c[h//4:h//4+h//5, w//4:w//4+w//5] = 0.0
    for row in np.random.choice(h, 6, replace=False):
        c[row, :] = 0.0
    return np.clip(c, 0, 1)

CORRUPTIONS = [corrupt_gaussian, corrupt_block, corrupt_scanline, corrupt_combined]
CORR_NAMES  = ["Gaussian Noise", "Block Erase", "Scan-line Drop", "Combined"]

# ── PSNR ──────────────────────────────────────────────────────────────────────
def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse < 1e-10 else 20 * math.log10(1.0) - 10 * math.log10(mse)

# ── Load + normalize DICOM ────────────────────────────────────────────────────
def load_norm(path):
    ds  = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    norm = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
    return cv2.resize(norm, (224, 224))

# ── Run test ──────────────────────────────────────────────────────────────────
results = []
for i, (series_name, fpath) in enumerate(test_files):
    corr_fn   = CORRUPTIONS[i % len(CORRUPTIONS)]
    corr_name = CORR_NAMES[i % len(CORRUPTIONS)]
    clean     = load_norm(fpath)
    corrupted = corr_fn(clean)

    inp_t   = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).float().to(_device)
    clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float().to(_device)
    with torch.no_grad():
        out_t = model(inp_t)

    psnr_c = psnr(inp_t, clean_t)
    psnr_r = psnr(out_t, clean_t)
    gain   = psnr_r - psnr_c
    fname  = os.path.basename(fpath)

    results.append({
        "series": series_name, "file": fname,
        "corr": corr_name,
        "psnr_c": psnr_c, "psnr_r": psnr_r, "gain": gain,
        "clean": clean, "corrupted": corrupted,
        "recovered": out_t.squeeze().cpu().numpy()
    })
    print(f"[{i+1}/{len(test_files)}] {fname} | {corr_name:16s} | "
          f"Corrupted: {psnr_c:.1f} dB | Recovered: {psnr_r:.1f} dB | Gain: +{gain:.1f} dB")

# ── Summary stats ─────────────────────────────────────────────────────────────
avg_gain = np.mean([r["gain"] for r in results])
avg_psnr_r = np.mean([r["psnr_r"] for r in results])
print(f"\n{'='*60}")
print(f"  Average PSNR recovered : {avg_psnr_r:.2f} dB")
print(f"  Average PSNR gain      : +{avg_gain:.2f} dB")
print(f"  All tests passed       : {all(r['gain'] > 3 for r in results)}")
print(f"{'='*60}")

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
fig.patch.set_facecolor("#08080f")

for row, r in enumerate(results):
    titles = [
        f"{r['file']}\n(Clean)",
        f"{r['corr']}\n{r['psnr_c']:.1f} dB",
        f"Recovered  +{r['gain']:.1f} dB\n{r['psnr_r']:.1f} dB"
    ]
    images = [r["clean"], r["corrupted"], r["recovered"]]
    colors = ["#00e5ff", "#ff4444", "#00ff88"]

    for col, (img, title, color) in enumerate(zip(images, titles, colors)):
        ax = axes[row][col] if n > 1 else axes[col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=5)
        ax.axis("off")

fig.suptitle(
    f"PhantomaShield — BRaTs2021 Updated Autoencoder Test\n"
    f"Avg Recovered PSNR: {avg_psnr_r:.1f} dB  |  Avg Gain: +{avg_gain:.1f} dB",
    color="white", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()

OUT = "test_output/full_test_result.png"
plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nResult image : {OUT}")

# ── Text summary ──────────────────────────────────────────────────────────────
with open("test_output/full_test_summary.txt", "w") as f:
    f.write("PhantomaShield Autoencoder Test Summary (BRaTs2021 model)\n")
    f.write("="*60 + "\n")
    for r in results:
        f.write(f"{r['file']:20s} | {r['corr']:16s} | "
                f"corrupted={r['psnr_c']:.1f}dB | "
                f"recovered={r['psnr_r']:.1f}dB | gain=+{r['gain']:.1f}dB\n")
    f.write("="*60 + "\n")
    f.write(f"Average recovered PSNR : {avg_psnr_r:.2f} dB\n")
    f.write(f"Average gain           : +{avg_gain:.2f} dB\n")
print("Summary txt  : test_output/full_test_summary.txt")
