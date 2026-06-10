"""
build_dataset.py — PhantomaShield Dataset Builder v2
=====================================================
Generates a richer 3-class labeled dataset from real DICOM files.

Improvements over v1:
  • 6 tampering attack types (was 4)
  • More realistic AI-gen simulation (spectral texture, GAN checkerboard, oversmooth)
  • Per-class progress bars
  • Handles TCIA extensionless files and missing Transfer Syntax
  • Saves metadata sidecar (.meta.npy) with attack type for analysis

Classes:
  datasets/original/      ← real CT DICOM files (as .npy)
  datasets/tampered/      ← synthetically tampered copies
  datasets/ai_generated/  ← AI-pattern simulated copies

Usage:
  python build_dataset.py --source "C:/path/to/dicom_data" --out datasets --max 2000
"""

import os
import argparse
import random
import numpy as np
import cv2
import pydicom

random.seed(42)
np.random.seed(42)


# ─── DICOM Loader ─────────────────────────────────────────────────────────────
def load_image_array(fpath: str):
    """
    Load a pixel array from DICOM or standard image formats (PNG/JPG).
    Applies 3 fallback strategies for DICOM:
      1. Native pydicom
      2. Override Transfer Syntax
      3. Raw pixel bytes reconstruction
    Returns (ds, arr) or (None, None) on failure.
    """
    # Check for standard image formats
    ext = os.path.splitext(fpath)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        try:
            arr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return None, arr
            return None, None
        except Exception:
            return None, None

    try:
        ds = pydicom.dcmread(fpath, force=True)
        # Patch missing Transfer Syntax (common in TCIA exports)
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Strategy 1: native
        try:
            arr = ds.pixel_array
        except Exception:
            # Strategy 2: override TS
            try:
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                arr = ds.pixel_array
            except Exception:
                # Strategy 3: raw bytes
                arr = _raw_pixel_bytes(ds)
                if arr is None:
                    return None, None

        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
        return ds, arr
    except Exception:
        return None, None


def _raw_pixel_bytes(ds) -> np.ndarray | None:
    try:
        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))
        bits = int(getattr(ds, "BitsAllocated", 16))
        if rows <= 0 or cols <= 0:
            return None
        dtype = {8: np.uint8, 16: np.uint16, 32: np.int32}.get(bits, np.uint16)
        raw   = bytes(ds.PixelData)
        exp   = rows * cols * (bits // 8)
        raw   = (raw + b"\x00" * exp)[:exp]
        return np.frombuffer(raw, dtype=dtype).reshape(rows, cols)
    except Exception:
        return None


# ─── Normalisation helper ─────────────────────────────────────────────────────
def _norm(arr: np.ndarray) -> np.ndarray:
    f = arr.astype(np.float32)
    pmin, pmax = f.min(), f.max()
    return (f - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(f)


def _denorm(normed: np.ndarray, pmin: float, pmax: float,
            dtype) -> np.ndarray:
    return (normed * (pmax - pmin) + pmin).astype(dtype)


# ─── Tampering attacks ────────────────────────────────────────────────────────
def _rand_region(h, w, min_frac=0.1, max_frac=0.45):
    rh = random.randint(int(h * min_frac), int(h * max_frac))
    rw = random.randint(int(w * min_frac), int(w * max_frac))
    sy = random.randint(0, h - rh - 1)
    sx = random.randint(0, w - rw - 1)
    return sy, sx, rh, rw


def tamper_copy_paste(arr):
    """Duplicate a region from elsewhere (region-duplication forgery)."""
    h, w = arr.shape
    out  = arr.copy()
    sy1, sx1, rh, rw = _rand_region(h, w)
    sy2, sx2, _,  _  = _rand_region(h, w)
    rh = min(rh, h - sy2)
    rw = min(rw, w - sx2)
    patch = arr[sy1:sy1+rh, sx1:sx1+rw].copy()
    alpha = random.uniform(0.65, 0.98)
    out[sy2:sy2+rh, sx2:sx2+rw] = (
        alpha * patch + (1 - alpha) * out[sy2:sy2+rh, sx2:sx2+rw]
    ).astype(arr.dtype)
    return out


def tamper_brightness_patch(arr):
    """Local brightness/contrast manipulation."""
    h, w = arr.shape
    out  = arr.copy().astype(np.float32)
    sy, sx, rh, rw = _rand_region(h, w, 0.12, 0.5)
    # Randomly choose: brighten, darken, or invert
    op = random.choice(["brighten", "darken", "gamma"])
    patch = out[sy:sy+rh, sx:sx+rw]
    if op == "brighten":
        patch *= random.uniform(1.5, 3.0)
    elif op == "darken":
        patch *= random.uniform(0.1, 0.5)
    else:  # gamma
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            n = (patch - pmin) / (pmax - pmin)
            gamma = random.uniform(0.3, 2.5)
            patch = np.power(n, gamma) * (pmax - pmin) + pmin
    out[sy:sy+rh, sx:sx+rw] = patch
    return np.clip(out, arr.min(), arr.max()).astype(arr.dtype)


def tamper_noise_injection(arr):
    """Inject structured or salt-and-pepper noise in a sub-region."""
    h, w = arr.shape
    out  = arr.copy().astype(np.float32)
    sy, sx, rh, rw = _rand_region(h, w, 0.1, 0.45)
    std  = float(out.std()) * random.uniform(1.5, 5.0)
    mode = random.choice(["gaussian", "impulse"])
    if mode == "gaussian":
        out[sy:sy+rh, sx:sx+rw] += np.random.normal(0, std, (rh, rw))
    else:  # salt & pepper
        mask = np.random.rand(rh, rw) < 0.1
        out[sy:sy+rh, sx:sx+rw][mask] = arr.max() if random.random() > 0.5 else arr.min()
    return np.clip(out, arr.min(), arr.max()).astype(arr.dtype)


def tamper_erase_region(arr):
    """Zero/fill a rectangular block (deletion attack)."""
    h, w = arr.shape
    out  = arr.copy()
    sy, sx, rh, rw = _rand_region(h, w, 0.08, 0.35)
    fill = random.choice([arr.min(), arr.max(),
                           int(arr.mean()), 0])
    out[sy:sy+rh, sx:sx+rw] = fill
    return out


def tamper_splicing(arr):
    """
    Splice a smoothed version of a region — simulates smooth boundary
    around a transplanted patch (compression artifact).
    """
    h, w = arr.shape
    out  = arr.copy().astype(np.float32)
    sy, sx, rh, rw = _rand_region(h, w, 0.15, 0.5)
    patch = out[sy:sy+rh, sx:sx+rw].copy()
    # Smooth the patch edges
    blurred = cv2.GaussianBlur(patch, (15, 15), 3)
    blend   = random.uniform(0.3, 0.7)
    out[sy:sy+rh, sx:sx+rw] = blend * blurred + (1 - blend) * patch
    return np.clip(out, arr.min(), arr.max()).astype(arr.dtype)


def tamper_frequency_manipulation(arr):
    """
    Alter DCT frequency components in a block (JPEG compression forgery trace).
    """
    h, w = arr.shape
    out  = arr.copy().astype(np.float32)
    pmin, pmax = out.min(), out.max()
    if pmax <= pmin:
        return out.astype(arr.dtype)

    norm = (out - pmin) / (pmax - pmin)
    sy, sx, rh, rw = _rand_region(h, w, 0.2, 0.5)
    patch = norm[sy:sy+rh, sx:sx+rw]
    patch_u8 = (patch * 255).astype(np.uint8)
    # Compress with very low JPEG quality → blocky artifacts
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, random.randint(5, 30)]
    _, enc = cv2.imencode(".jpg", patch_u8, encode_params)
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if dec.shape == patch.shape:
        norm[sy:sy+rh, sx:sx+rw] = dec
    out = norm * (pmax - pmin) + pmin
    return np.clip(out, arr.min(), arr.max()).astype(arr.dtype)


TAMPER_OPS = [
    tamper_copy_paste,
    tamper_brightness_patch,
    tamper_noise_injection,
    tamper_erase_region,
    tamper_splicing,
    tamper_frequency_manipulation,
]


def apply_tampering(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Apply 1–3 random tamper ops. Returns (tampered_arr, attack_name)."""
    ops = random.sample(TAMPER_OPS, k=random.randint(1, 3))
    out = arr.copy()
    for op in ops:
        try:
            out = op(out)
        except Exception:
            pass
    return out, "+".join(o.__name__ for o in ops)


# ─── AI-generation simulation ─────────────────────────────────────────────────
def simulate_ai_generated(arr: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
    """
    Simulate GAN/Diffusion model artifacts:
      1. Over-smoothing (kills natural scanner noise)
      2. Periodic GAN checkerboard in frequency domain
      3. Bit depth quantisation (lossy reconstruction)
      4. Spectral anomaly (atypical high-frequency distribution)
    Now applies to localized patches to train spatial awareness.
    """
    h, w = arr.shape
    f    = arr.astype(np.float32)
    pmin, pmax = f.min(), f.max()
    n = (f - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(f)

    steps = []
    mask = np.zeros_like(arr, dtype=np.uint8)

    # 3. LIMIT NUMBER OF MODIFICATIONS
    num_ops = random.randint(1, 2)
    available_ops = ["blur", "gan_grid", "quantize", "spectral_suppress"]
    chosen_ops = random.sample(available_ops, num_ops)

    # 4. ENSURE MIX OF LOCAL + GLOBAL CASES
    is_local = random.random() < 0.8
    if is_local:
        sy, sx, rh, rw = _rand_region(h, w, 0.15, 0.5)
        patch = n[sy:sy+rh, sx:sx+rw].copy()
        mask[sy:sy+rh, sx:sx+rw] = 1
    else:
        patch = n.copy()
        mask[...] = 1

    ph, pw = patch.shape

    if "blur" in chosen_ops:
        sigma = random.uniform(1.0, 3.5)
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
        steps.append(f"blur({sigma:.1f})")

    if "gan_grid" in chosen_ops:
        freq = random.uniform(0.06, 0.22)
        amp  = random.uniform(0.008, 0.035)
        xs   = np.linspace(0, freq * pw * 2 * np.pi, pw)
        ys   = np.linspace(0, freq * ph * 2 * np.pi, ph)
        XX, YY = np.meshgrid(xs, ys)
        patch = np.clip(patch + (np.sin(XX) * np.cos(YY) * amp).astype(np.float32), 0, 1)
        steps.append("gan_grid")

    if "quantize" in chosen_ops:
        levels = random.randint(32, 96)
        patch = (np.round(patch * levels) / levels).astype(np.float32)
        steps.append(f"quantize({levels})")

    if "spectral_suppress" in chosen_ops:
        fft_data = np.fft.fft2(patch)
        fft_shift = np.fft.fftshift(fft_data)
        cy, cx = ph // 2, pw // 2
        radius = int(min(ph, pw) * random.uniform(0.3, 0.5))
        Y, X = np.ogrid[:ph, :pw]
        fft_mask = ((Y - cy)**2 + (X - cx)**2) > radius**2
        fft_shift[fft_mask] *= random.uniform(0.0, 0.3)
        patch = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift))).astype(np.float32)
        patch = np.clip(patch, 0, 1)
        steps.append("spectral_suppress")

    if is_local:
        n[sy:sy+rh, sx:sx+rw] = patch
    else:
        n = patch

    result = (n * (pmax - pmin) + pmin).astype(arr.dtype)
    return result, "+".join(steps), mask


# ─── Per-file processing ──────────────────────────────────────────────────────
def process_one(img_path: str, out_original: str, out_tampered: str,
                out_ai: str, index: int) -> int:
    ds, arr = load_image_array(img_path)
    if arr is None:
        return 0
    if arr.shape[0] < 32 or arr.shape[1] < 32:
        return 0

    fname = f"sample_{index:05d}.npy"

    # 1) Original
    np.save(os.path.join(out_original, fname), arr)

    # 2) Tampered (1–3 attacks)
    try:
        tampered, attack_name = apply_tampering(arr)
        np.save(os.path.join(out_tampered, fname), tampered)
        np.save(os.path.join(out_tampered, fname.replace(".npy", ".meta.npy")),
                np.array([attack_name]))
        t_mask = (np.abs(tampered.astype(np.float32) - arr.astype(np.float32)) > 1e-5).astype(np.uint8)
        np.save(os.path.join(out_tampered, fname.replace(".npy", ".mask.npy")), t_mask)
    except Exception:
        np.save(os.path.join(out_tampered, fname), arr)  # fallback: save clean

    # 3) AI-generated
    try:
        ai_arr, ai_desc, ai_mask = simulate_ai_generated(arr)
        np.save(os.path.join(out_ai, fname), ai_arr)
        np.save(os.path.join(out_ai, fname.replace(".npy", ".meta.npy")),
                np.array([ai_desc]))
        np.save(os.path.join(out_ai, fname.replace(".npy", ".mask.npy")), ai_mask)
    except Exception:
        np.save(os.path.join(out_ai, fname), arr)

    return 1


# ─── Main builder ─────────────────────────────────────────────────────────────
def build_dataset(source_dir: str, out_dir: str, max_files: int = 2000):
    # Collect all image files (DICOM, PNG, JPG)
    img_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith((".dcm", ".dicom", ".png", ".jpg", ".jpeg")) or "." not in f:
                img_files.append(fp)

    if not img_files:
        print(f"No image files found in {source_dir}")
        return

    random.shuffle(img_files)
    img_files = img_files[:max_files]
    print(f"Found {len(img_files)} image files -> building 3-class dataset...\n")

    out_original = os.path.join(out_dir, "original")
    out_tampered = os.path.join(out_dir, "tampered")
    out_ai       = os.path.join(out_dir, "ai_generated")
    for d in [out_original, out_tampered, out_ai]:
        os.makedirs(d, exist_ok=True)

    count, skipped = 0, 0
    for i, path in enumerate(img_files):
        n = process_one(path, out_original, out_tampered, out_ai, i)
        if n:
            count += 1
        else:
            skipped += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(img_files):
            pct = (i + 1) / len(img_files) * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"  [{bar}] {pct:5.1f}%  {i+1}/{len(img_files)}"
                  f"  valid: {count}  skipped: {skipped}")

    print(f"\nDataset built successfully:")
    print(f"  original/:     {len([f for f in os.listdir(out_original) if f.endswith('.npy') and '.meta' not in f])} samples")
    print(f"  tampered/:     {len([f for f in os.listdir(out_tampered) if f.endswith('.npy') and '.meta' not in f])} samples")
    print(f"  ai_generated/: {len([f for f in os.listdir(out_ai)       if f.endswith('.npy') and '.meta' not in f])} samples")
    print(f"\nNext step:")
    print(f"  python train_detector.py --data {out_dir} --epochs 50 --batch 8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhantomaShield Dataset Builder v2")
    parser.add_argument("--source", required=True, help="Folder with real DICOM files")
    parser.add_argument("--out",    default="datasets")
    parser.add_argument("--max",    type=int, default=2000,
                        help="Max DICOM source files to use")
    args = parser.parse_args()
    build_dataset(args.source, args.out, args.max)
