"""
generate_corrupted_dicoms.py
============================
Kaggle / Colab ready script.

Pass ONE clean DICOM file → get FOUR corrupted DICOM files back.

Each output file is a valid .dcm with the corrupted pixel data embedded.
Visual comparison plot is also saved.

Usage (Kaggle/Colab cell):
    INPUT_DCM  = "/kaggle/input/.../slice_0010.dcm"   ← your DICOM
    OUTPUT_DIR = "/kaggle/working/corrupted_dicoms"
"""

import os
import copy
import numpy as np
import cv2
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
#  CONFIG — change these two lines
# ══════════════════════════════════════════════════════════════
INPUT_DCM  = r"D:\Downloads 2\IM000012.dcm"
OUTPUT_DIR = r"D:\Downloads 2\corrupted_dicoms"
# ══════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load DICOM ────────────────────────────────────────────────────────────────
print(f"Loading: {INPUT_DCM}")
ds  = pydicom.dcmread(INPUT_DCM, force=True)
arr = ds.pixel_array.astype(np.float32)
if arr.ndim == 3:
    arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]

h, w       = arr.shape
orig_dtype = ds.pixel_array.dtype
pmin, pmax = float(arr.min()), float(arr.max())
print(f"  Shape : {arr.shape}  dtype: {orig_dtype}  range: [{pmin:.0f}, {pmax:.0f}]")


# ── Normalize helper ──────────────────────────────────────────────────────────
def to_norm(a):
    """float32 array → [0, 1]"""
    p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)

def from_norm(n, pmin, pmax, dtype):
    """[0, 1] → original pixel range + dtype"""
    return np.clip(n * (pmax - pmin) + pmin, pmin, pmax).astype(dtype)


# ── 4 Corruption functions ────────────────────────────────────────────────────
def corrupt_gaussian(arr):
    """Simulates scanner acquisition / low-dose CT noise"""
    norm = to_norm(arr)
    norm += np.random.randn(*norm.shape).astype(np.float32) * 0.15
    return from_norm(np.clip(norm, 0, 1), pmin, pmax, orig_dtype)

def corrupt_block_erase(arr):
    """Simulates a tampered / deleted rectangular region"""
    out  = arr.copy()
    rh   = h // 6
    rw   = w // 6
    sy   = h // 5
    sx   = w // 5
    out[sy : sy+rh, sx : sx+rw] = int(pmin)   # fill with minimum (black)
    return out.astype(orig_dtype)

def corrupt_scanline(arr):
    """Simulates MRI/CT readout failure — horizontal line dropout"""
    out = arr.copy()
    n_lines = max(10, h // 20)
    rows = np.random.choice(h, n_lines, replace=False)
    for r in rows:
        out[r, :] = int(pmin)
    return out.astype(orig_dtype)

def corrupt_combined(arr):
    """All three corruptions combined"""
    # Gaussian noise
    norm = to_norm(arr)
    norm += np.random.randn(*norm.shape).astype(np.float32) * 0.10
    norm  = np.clip(norm, 0, 1)
    out   = from_norm(norm, pmin, pmax, orig_dtype)
    # Block erase
    rh, rw = h // 4, w // 4
    out[h//4 : h//4+rh, w//4 : w//4+rw] = int(pmin)
    # Scan-line dropout
    for r in np.random.choice(h, 8, replace=False):
        out[r, :] = int(pmin)
    return out.astype(orig_dtype)


CORRUPTIONS = [
    ("gaussian_noise",  corrupt_gaussian,   "Gaussian Noise\n(scanner noise simulation)"),
    ("block_erase",     corrupt_block_erase,"Block Erase\n(tampered region)"),
    ("scanline_drop",   corrupt_scanline,   "Scan-line Dropout\n(readout failure)"),
    ("combined",        corrupt_combined,   "Combined\n(all three)"),
]


# ── Save corrupted DICOM helper ───────────────────────────────────────────────
def save_corrupted_dicom(original_ds, corrupted_arr, out_path):
    """
    Clone the original DICOM, replace pixel data with corrupted array,
    write as uncompressed ExplicitVRLittleEndian .dcm file.
    """
    new_ds = copy.deepcopy(original_ds)

    # ── Force uncompressed transfer syntax ────────────────────────────────────
    if not hasattr(new_ds, "file_meta") or new_ds.file_meta is None:
        new_ds.file_meta = pydicom.dataset.FileMetaDataset()
    new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    new_ds.is_implicit_VR   = False
    new_ds.is_little_endian = True

    # ── Set correct PixelRepresentation based on dtype ────────────────────────
    # int8/int16/int32 → signed (1), uint8/uint16 → unsigned (0)
    is_signed = np.issubdtype(corrupted_arr.dtype, np.signedinteger)
    new_ds.PixelRepresentation = 1 if is_signed else 0

    # ── Replace pixel data ────────────────────────────────────────────────────
    new_ds.PixelData     = corrupted_arr.tobytes()
    new_ds.Rows          = corrupted_arr.shape[0]
    new_ds.Columns       = corrupted_arr.shape[1]
    new_ds.BitsAllocated = corrupted_arr.dtype.itemsize * 8
    new_ds.BitsStored    = new_ds.BitsAllocated
    new_ds.HighBit       = new_ds.BitsAllocated - 1

    # ── Remove window tags so viewer uses robust percentile normalization ───
    # Original window center/width is calibrated for clean pixel values.
    # After corruption those tags no longer match, causing all-white display.
    for tag in [(0x0028, 0x1050), (0x0028, 0x1051),   # WindowCenter, WindowWidth
                (0x0028, 0x1052), (0x0028, 0x1053)]:   # RescaleIntercept, RescaleSlope
        if tag in new_ds:
            del new_ds[tag]

    pydicom.dcmwrite(out_path, new_ds)


# ── Generate corrupted DICOMs ─────────────────────────────────────────────────
print(f"\nGenerating 4 corrupted DICOM files in: {OUTPUT_DIR}\n")

norm_orig    = to_norm(arr)
saved_paths  = []
corrupted_arrays = []

base_name = os.path.splitext(os.path.basename(INPUT_DCM))[0]

for tag, fn, label in CORRUPTIONS:
    corrupted   = fn(arr)
    out_fname   = f"{base_name}_corrupted_{tag}.dcm"
    out_path    = os.path.join(OUTPUT_DIR, out_fname)

    save_corrupted_dicom(ds, corrupted, out_path)
    size_kb = os.path.getsize(out_path) / 1024

    saved_paths.append(out_path)
    corrupted_arrays.append((to_norm(corrupted.astype(np.float32)), label))

    print(f"  [{tag:20s}]  saved -> {out_fname}  ({size_kb:.0f} KB)")


# ── Verification: re-read saved files ────────────────────────────────────────
print("\nVerification — re-reading saved files:")
for path in saved_paths:
    try:
        verify_ds  = pydicom.dcmread(path)
        verify_arr = verify_ds.pixel_array
        print(f"  OK  {os.path.basename(path):45s}  shape={verify_arr.shape}")
    except Exception as e:
        print(f"  ERR {os.path.basename(path)}: {e}")


# ── Visual comparison ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor("#0a0a18")

# Original
axes[0].imshow(norm_orig, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original (Clean)", color="#00e5ff", fontsize=11, fontweight="bold")
axes[0].axis("off")

colors = ["#ff9900", "#ff4444", "#cc44ff", "#ff88aa"]
for i, (img, label) in enumerate(corrupted_arrays):
    ax = axes[i + 1]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(label, color=colors[i], fontsize=10, fontweight="bold")
    ax.axis("off")

fig.suptitle(
    f"PhantomaShield — 4 Corrupted DICOM outputs from: {os.path.basename(INPUT_DCM)}",
    color="white", fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, f"{base_name}_corruption_preview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
plt.close()

print(f"\nPreview image : {plot_path}")
print(f"\nDone! {len(saved_paths)} corrupted DICOM files saved to: {OUTPUT_DIR}")
print("\nFiles created:")
for p in saved_paths + [plot_path]:
    print(f"  {os.path.basename(p)}")
