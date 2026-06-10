"""
corruption_detector.py
Analyzes DICOM pixel data for corruption.
Detects: missing regions, broken pixel blocks, incomplete scan data, noise corruption.
"""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

CORRUPTION_TYPES = {
    "MISSING_REGIONS": "Missing Regions",
    "BROKEN_BLOCKS": "Broken Pixel Blocks",
    "INCOMPLETE_SCAN": "Incomplete Scan Data",
    "NOISE_CORRUPTION": "Noise/Artifact Corruption",
    "NONE": "No Corruption Detected",
}


def detect_corruption(pixel_array: np.ndarray | None, ds=None) -> dict:
    """
    Analyze DICOM data for corruption.
    Returns:
      {
        type, severity, affected_percentage,
        recoverable, description,
        mask (numpy array, True where corrupted)
      }
    """
    if pixel_array is None:
        return {
            "type": CORRUPTION_TYPES["INCOMPLETE_SCAN"],
            "severity": "CRITICAL",
            "affected_percentage": 100.0,
            "recoverable": False,
            "description": "Pixel data could not be decoded. The DICOM file may be severely corrupted or uses an unsupported transfer syntax.",
            "mask": None,
        }

    arr = pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]

    h, w = arr.shape
    total_pixels = h * w

    # Normalize
    pmin, pmax = arr.min(), arr.max()
    if pmax > pmin:
        arr_norm = (arr - pmin) / (pmax - pmin)
    else:
        arr_norm = np.zeros_like(arr)

    corruption_mask = np.zeros((h, w), dtype=bool)

    # 1. Missing regions: blocks of exact zero or max
    zero_mask = arr_norm < 0.005
    zero_pct = zero_mask.sum() / total_pixels * 100

    max_mask = arr_norm > 0.995
    max_pct = max_mask.sum() / total_pixels * 100

    # 2. Broken block detection: look for uniform square blocks
    block_size = max(8, min(16, h // 16, w // 16))
    block_anomaly_count = 0
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = arr_norm[i:i+block_size, j:j+block_size]
            if block.std() < 0.001:  # perfectly uniform = likely broken
                block_anomaly_count += 1
                corruption_mask[i:i+block_size, j:j+block_size] = True

    block_pct = corruption_mask.sum() / total_pixels * 100

    # 3. Noise detection using high-frequency energy
    # NOTE: OpenCV 4.13 AVX2 does NOT support float32 → CV_64F Laplacian.
    # Use CV_32F output (same type as input), then cast to float64 for math.
    if h >= 4 and w >= 4:
        laplacian = cv2.Laplacian(arr_norm.astype(np.float32), cv2.CV_32F)
        noise_score = float(np.abs(laplacian.astype(np.float64)).mean())
    else:
        noise_score = 0.0

    # Determine corruption type
    if zero_pct > 15 or (max_pct > 5 and max_pct < 80):
        ctype = "MISSING_REGIONS"
        affected = max(zero_pct, max_pct)
        corruption_mask |= (zero_mask | max_mask)
    elif block_pct > 5:
        ctype = "BROKEN_BLOCKS"
        affected = block_pct
    elif noise_score > 0.3:
        ctype = "NOISE_CORRUPTION"
        affected = min(90, noise_score * 100)
        corruption_mask = np.ones((h, w), dtype=bool)  # noise is global — full image mask
    elif zero_pct > 40:
        ctype = "INCOMPLETE_SCAN"
        affected = zero_pct
        corruption_mask = zero_mask
    elif zero_pct < 1 and max_pct < 1 and block_pct < 1 and noise_score < 0.1:
        ctype = "NONE"
        affected = 0.0
    else:
        ctype = "MISSING_REGIONS"
        affected = max(zero_pct, block_pct)

    affected = round(float(affected), 1)

    # Severity classification
    if affected >= 40 or ctype == "NONE":
        severity = "HIGH" if ctype != "NONE" else "NONE"
    elif affected >= 15:
        severity = "MODERATE"
    else:
        severity = "LOW"

    recoverable = affected < 70 and ctype != "INCOMPLETE_SCAN" or (
        ctype == "INCOMPLETE_SCAN" and affected < 50
    )

    desc_map = {
        "MISSING_REGIONS": f"Detected missing or zeroed pixel regions covering ~{affected:.1f}% of the image. These regions will be reconstructed using autoencoder inpainting.",
        "BROKEN_BLOCKS": f"Found {block_anomaly_count} uniform pixel blocks (~{affected:.1f}% of image) indicating block-level corruption. Inpainting will restore missing block content.",
        "NOISE_CORRUPTION": f"Image shows elevated noise/artifact patterns (Laplacian score: {noise_score:.3f}). Denoising and reconstruction will be applied.",
        "INCOMPLETE_SCAN": f"Scan data appears {affected:.0f}% complete. Missing data will be estimated from learned normal scan patterns.",
        "NONE": "No significant corruption detected. The image appears intact.",
    }

    return {
        "type": CORRUPTION_TYPES.get(ctype, ctype),
        "severity": severity,
        "affected_percentage": affected,
        "recoverable": recoverable,
        "description": desc_map.get(ctype, "Corruption analysis complete."),
        "mask": corruption_mask,
    }


def generate_corrupted_dicoms(
    pixel_array: np.ndarray,
    n_patches: int = 3,
    min_size_frac: float = 0.06,
    max_size_frac: float = 0.12,
    fill_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block-erase corruption: erases exactly `n_patches` (default 3) small
    rectangular regions at random positions in a copy of the DICOM pixel array.

    Args:
        pixel_array   : 2-D (H, W) numpy array of original pixel values.
        n_patches     : Number of small blocks to erase (default 3).
        min_size_frac : Min patch side as fraction of image size (default 6%).
        max_size_frac : Max patch side as fraction of image size (default 12%).
        fill_value    : Value written into erased blocks.
                        None → uses image minimum (black / background).

    Returns:
        corrupted : np.ndarray — same shape & dtype as input, blocks erased.
        mask      : np.ndarray bool (H, W) — True where pixels were erased.

    Usage:
        corrupted, mask = generate_corrupted_dicoms(pixel_array)
        corrupted, mask = generate_corrupted_dicoms(pixel_array, n_patches=3,
                                                    min_size_frac=0.05,
                                                    max_size_frac=0.10)
    """
    arr = pixel_array.copy()
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]

    h, w = arr.shape
    mask = np.zeros((h, w), dtype=bool)
    fill = float(arr.min()) if fill_value is None else float(fill_value)

    # Compute patch size bounds in pixels
    min_ph = max(4, int(h * min_size_frac))
    max_ph = max(min_ph + 1, int(h * max_size_frac))
    min_pw = max(4, int(w * min_size_frac))
    max_pw = max(min_pw + 1, int(w * max_size_frac))

    rng = np.random.default_rng()

    for _ in range(n_patches):
        ph = int(rng.integers(min_ph, max_ph))      # random patch height
        pw = int(rng.integers(min_pw, max_pw))      # random patch width
        sy = int(rng.integers(0, max(1, h - ph)))   # random top-left y
        sx = int(rng.integers(0, max(1, w - pw)))   # random top-left x

        arr[sy:sy + ph, sx:sx + pw]  = fill         # erase block → fill value
        mask[sy:sy + ph, sx:sx + pw] = True         # mark as corrupted

    logger.debug(
        "generate_corrupted_dicoms: %d patches erased | mask_coverage=%.1f%%",
        n_patches, mask.sum() / (h * w) * 100,
    )
    return arr.astype(pixel_array.dtype), mask
