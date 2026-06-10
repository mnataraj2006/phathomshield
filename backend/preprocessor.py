"""
preprocessor.py
Preprocesses DICOM pixel arrays for model input and display.

Correct DICOM display pipeline (in order):
  1. Apply RescaleSlope / RescaleIntercept  (CT HU conversion, some MRI)
  2. Apply WindowCenter / WindowWidth        (clinical contrast mapping)
  3. Invert if MONOCHROME1                   (X-ray film convention)

WindowCenter/WindowWidth are now read as numeric floats directly from the
dicom_loader (bypassing str/MultiValue conversion bugs that caused images
to be displayed very dark when the fallback percentile normalizer was used
instead of proper DICOM windowing).
"""
import numpy as np
import cv2
from PIL import Image
import io
import base64


# ─── Rescale: apply RescaleSlope / RescaleIntercept ──────────────────────────
def _apply_rescale(arr: np.ndarray,
                   slope: float | None,
                   intercept: float | None) -> np.ndarray:
    """
    Convert stored pixel values to manufacturer-intended signal values.
    For CT: slope=1, intercept=-1024  → HU = pixel * slope + intercept
    For MRI: usually slope=1, intercept=0 (identity — no change needed).
    Safe to call with (None, None) — returns arr unchanged.
    """
    arr = arr.astype(np.float32)
    if slope is not None and float(slope) != 1.0:
        arr = arr * float(slope)
    if intercept is not None and float(intercept) != 0.0:
        arr = arr + float(intercept)
    return arr


# ─── Windowing ────────────────────────────────────────────────────────────────
def normalize_pixel_array(pixel_array: np.ndarray,
                           window_center: float | None = None,
                           window_width:  float | None = None) -> np.ndarray:
    """
    Normalize DICOM pixel values to [0, 255] uint8 for display.

    Priority:
      1. DICOM clinical windowing (WindowCenter + WindowWidth) — matches what
         professional DICOM viewers like MicroDicom display.
      2. Robust percentile clipping (1st–99th pct of NON-ZERO pixels) —
         prevents background zeros from crushing the brain-tissue range.
    """
    arr = pixel_array.astype(np.float32)

    # Flatten multi-frame to first frame
    if arr.ndim == 3 and arr.shape[0] > 3:
        arr = arr[0]

    if window_center is not None and window_width is not None and float(window_width) > 0:
        # ── DICOM clinical windowing ───────────────────────────────────────
        wc  = float(window_center)
        ww  = float(window_width)
        low  = wc - ww / 2.0
        high = wc + ww / 2.0
        arr  = np.clip(arr, low, high)
        arr  = (arr - low) / (high - low) * 255.0
    else:
        # ── Robust percentile fallback ─────────────────────────────────────
        # Use NON-ZERO pixels only to avoid scanner background zeros
        # crushing the useful tissue contrast range.
        nonzero = arr[arr > 0]
        if len(nonzero) > 10:
            pmin = float(np.percentile(nonzero, 1.0))
            pmax = float(np.percentile(nonzero, 99.5))
        else:
            pmin = float(arr.min())
            pmax = float(arr.max())
        if pmax <= pmin:
            pmin, pmax = float(arr.min()), float(arr.max())
        arr = np.clip(arr, pmin, pmax)
        if pmax > pmin:
            arr = (arr - pmin) / (pmax - pmin) * 255.0
        else:
            arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


# ─── String-based window parsing (legacy fallback for tags-only callers) ──────
def _parse_window(tags: dict | None) -> tuple[float | None, float | None]:
    """
    Extract WindowCenter / WindowWidth from the string tags dict.
    Handles both DICOM backslash multi-value ('40\\400') and Python list
    repr ('[40, 400]') that newer pydicom versions produce.
    """
    if not tags:
        return None, None
    try:
        wc_raw = tags.get("(0028,1050)")
        ww_raw = tags.get("(0028,1051)")
        if not wc_raw or not ww_raw:
            return None, None

        def _first_num(s):
            s = str(s).strip()
            # Python list repr: "[1119.0, 400.0]" → strip brackets, take first
            s = s.strip("[]'\" ")
            # DICOM backslash multi-value: "1119\\400" → take first
            s = s.split("\\")[0].split(",")[0].strip().strip("'\" ")
            return float(s)

        return _first_num(wc_raw), _first_num(ww_raw)
    except (TypeError, ValueError):
        return None, None


def _is_monochrome1(tags: dict | None) -> bool:
    """Return True if PhotometricInterpretation is MONOCHROME1.
    MONOCHROME1: high pixel value = DARK (inverted from normal display).
    Common in scanned X-rays and some CR/DR modalities."""
    if not tags:
        return False
    pi = tags.get("(0028,0004)", "") or ""
    return str(pi).strip().upper() == "MONOCHROME1"


# ─── Main display conversion ──────────────────────────────────────────────────
def to_rgb(pixel_array: np.ndarray,
           dicom_data: dict | None = None,
           tags: dict | None = None) -> np.ndarray:
    """
    Convert DICOM pixel array to an RGB (H, W, 3) uint8 image, applying
    the full DICOM display pipeline:
        RescaleSlope/Intercept → Window/Level → MONOCHROME1 inversion

    Parameters
    ----------
    pixel_array : np.ndarray     Raw pixel values from the DICOM file.
    dicom_data  : dict | None    Full dict returned by load_dicom() — preferred.
                                 Contains numeric window_center, window_width,
                                 rescale_slope, rescale_intercept.
    tags        : dict | None    String-format tags dict — legacy fallback when
                                 dicom_data is not available.
    """
    wc = ww = slope = intercept = None
    is_mono1 = False

    if dicom_data is not None:
        # ── Preferred: use pre-parsed numeric values from loader ──────────
        wc        = dicom_data.get("window_center")
        ww        = dicom_data.get("window_width")
        slope     = dicom_data.get("rescale_slope")
        intercept = dicom_data.get("rescale_intercept")
        pi_str    = dicom_data.get("photometric_interpretation", "")
        is_mono1  = str(pi_str).strip().upper() == "MONOCHROME1"

        # Also pull from string tags if numeric values are missing
        if (wc is None or ww is None) and dicom_data.get("tags"):
            wc_f, ww_f = _parse_window(dicom_data["tags"])
            if wc is None: wc = wc_f
            if ww is None: ww = ww_f
        if not is_mono1 and dicom_data.get("tags"):
            is_mono1 = _is_monochrome1(dicom_data["tags"])

    elif tags is not None:
        # ── Legacy fallback: string tags only ─────────────────────────────
        wc, ww   = _parse_window(tags)
        is_mono1 = _is_monochrome1(tags)

    # Step 1: Apply RescaleSlope / RescaleIntercept
    arr = _apply_rescale(pixel_array, slope, intercept)

    # Step 2: DICOM windowing → [0, 255] uint8
    norm = normalize_pixel_array(arr, window_center=wc, window_width=ww)

    # Step 3: Invert MONOCHROME1 (high pixel = dark in X-ray film convention)
    if is_mono1:
        norm = 255 - norm

    # Step 4: Convert to RGB
    if norm.ndim == 2:
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    elif norm.ndim == 3 and norm.shape[-1] == 3:
        rgb = norm
    elif norm.ndim == 3 and norm.shape[-1] == 1:
        rgb = cv2.cvtColor(norm[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    return rgb


# ─── Model preprocessing ─────────────────────────────────────────────────────
def preprocess_for_model(pixel_array: np.ndarray,
                          dicom_data: dict | None = None,
                          size: int = 224) -> np.ndarray:
    """
    Full preprocessing pipeline for model inference:
    1. DICOM-aware normalize (windowing + rescale)
    2. Convert to RGB
    3. Resize to (size × size)
    4. Mild Gaussian denoise
    Returns float32 array (H, W, 3) in [0, 1].
    """
    rgb = to_rgb(pixel_array, dicom_data=dicom_data)
    resized  = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    return denoised.astype(np.float32) / 255.0


# ─── Base64 export ────────────────────────────────────────────────────────────
def array_to_base64_png(arr: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string for API response."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.shape[-1] == 3:
        img = Image.fromarray(arr, mode='RGB')
    else:
        img = Image.fromarray(arr[:, :, 0], mode='L')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')
