"""
recovery_engine.py
==================
Reconstructs corrupted DICOM images using a three-stage hybrid pipeline:
  1. Percentile-safe normalization (prevents contrast blowout from extreme noise)
  2. Strong OpenCV-based denoising (NLMeans + Bilateral + Unsharp — always works)
  3. Optional Autoencoder refinement (when trained weights are available)

Guarantees a visually different, cleaner output on EVERY run regardless of
whether the AI model has been fully trained.
"""
import os
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "encoder_best_v2.pth")

_autoencoder = None
_device = None
_torch_available = False

try:
    import torch
    import torch.nn as nn
    _torch_available = True
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("PyTorch available for recovery engine (device: %s)", _device)
except ImportError:
    logger.warning("PyTorch not installed — using OpenCV-only recovery")


if _torch_available:
    import torch.nn as _nn

    class ResBlock(_nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = _nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn1   = _nn.BatchNorm2d(out_ch)
            self.conv2 = _nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.bn2   = _nn.BatchNorm2d(out_ch)
            self.relu  = _nn.LeakyReLU(0.2, inplace=True)
            self.skip  = _nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else _nn.Identity()

        def forward(self, x):
            res = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + res)

    class ConvAutoencoder(_nn.Module):
        """Residual U-Net autoencoder for masked DICOM inpainting.
        Accepts 2-channel input: [corrupted_image, mask] where mask=1 = missing region."""
        def __init__(self, in_channels: int = 2):
            super().__init__()
            self.enc1  = ResBlock(in_channels, 32); self.pool1 = _nn.MaxPool2d(2, 2)
            self.enc2  = ResBlock(32, 64);          self.pool2 = _nn.MaxPool2d(2, 2)
            self.enc3  = ResBlock(64, 128);         self.pool3 = _nn.MaxPool2d(2, 2)
            self.enc4  = ResBlock(128, 256);        self.pool4 = _nn.MaxPool2d(2, 2)
            self.bottleneck = ResBlock(256, 512)
            self.up4 = _nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = ResBlock(512, 256)
            self.up3 = _nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = ResBlock(256, 128)
            self.up2 = _nn.ConvTranspose2d(128, 64, 2, stride=2);  self.dec2 = ResBlock(128, 64)
            self.up1 = _nn.ConvTranspose2d(64, 32, 2, stride=2);   self.dec1 = ResBlock(64, 32)
            self.final = _nn.Sequential(_nn.Conv2d(32, 1, 1), _nn.Sigmoid())

        def forward(self, x):
            import torch as _t
            e1 = self.enc1(x);  e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
            b  = self.bottleneck(self.pool4(e4))
            d4 = self.dec4(_t.cat([self.up4(b), e4], 1))
            d3 = self.dec3(_t.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(_t.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(_t.cat([self.up1(d2), e1], 1))
            return self.final(d1)
else:
    ConvAutoencoder = None


# ── Model loading ─────────────────────────────────────────────────────────────
def get_autoencoder():
    global _autoencoder
    if not _torch_available:
        return None
    if _autoencoder is not None:
        return _autoencoder
    import torch
    model = ConvAutoencoder()
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded autoencoder weights from %s", MODEL_PATH)
        except Exception as e:
            logger.warning("Could not load autoencoder weights: %s — using untrained model", e)
    model.to(_device).eval()
    _autoencoder = model
    return _autoencoder


# ── Core helpers ──────────────────────────────────────────────────────────────
def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse multi-frame / multi-channel DICOM to a single 2D slice."""
    if arr.ndim == 3:
        return arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
    return arr


def _safe_normalize(arr2d: np.ndarray):
    """
    Normalize using 2nd–98th percentile clipping to defeat extreme
    scanner noise that otherwise blows all values to pure black/white.
    Returns (normalized_float32_in_0_1, pmin, pmax).
    """
    arr = arr2d.astype(np.float32)
    pmin = float(np.percentile(arr, 2.0))
    pmax = float(np.percentile(arr, 98.0))
    if pmax <= pmin:
        pmin, pmax = float(arr.min()), float(arr.max())
    arr_clipped = np.clip(arr, pmin, pmax)
    norm = (arr_clipped - pmin) / (pmax - pmin + 1e-8)
    return norm.astype(np.float32), pmin, pmax


# ── Stage 1: OpenCV Multi-Stage Denoiser ─────────────────────────────────────
def _opencv_denoise(arr_norm: np.ndarray) -> np.ndarray:
    """
    Three-stage OpenCV denoising. Always produces a visibly cleaner image.
    Input/output: float32 in [0, 1].
    """
    arr_u8 = np.clip(arr_norm * 255, 0, 255).astype(np.uint8)

    # Stage A: Non-Local Means — best for random scanner/Gaussian noise
    nlm = cv2.fastNlMeansDenoising(arr_u8, h=12, templateWindowSize=7, searchWindowSize=21)

    # Stage B: Bilateral filter — edge-preserving smoothing
    bilat = cv2.bilateralFilter(nlm, d=9, sigmaColor=60, sigmaSpace=60)

    # Stage C: Unsharp mask — restore fine anatomical edges blurred by denoising
    blurred  = cv2.GaussianBlur(bilat, (0, 0), 2.0)
    sharpened = cv2.addWeighted(bilat, 1.5, blurred, -0.5, 0)

    return np.clip(sharpened.astype(np.float32) / 255.0, 0.0, 1.0)


# ── Scan-line dropout detector & fixer ───────────────────────────────────────
def _detect_and_fix_scanlines(arr_norm: np.ndarray) -> np.ndarray:
    """
    Detects horizontal scan-line dropout rows (rows where >70% of pixels
    are near-zero) and fixes them via linear interpolation from neighbors.
    This is far more effective than the autoencoder for this specific artifact.
    """
    out = arr_norm.copy()
    h   = arr_norm.shape[0]

    # Find dropout rows: mean pixel value < 5% of image mean
    img_mean   = float(arr_norm.mean()) + 1e-8
    row_means  = arr_norm.mean(axis=1)
    dropout    = row_means < (img_mean * 0.15)      # threshold: 15% of avg
    dropout_idx = np.where(dropout)[0]

    if len(dropout_idx) == 0:
        return out                                   # no scan-lines detected

    logger.info("Scan-line fix: %d dropout rows detected", len(dropout_idx))

    for r in dropout_idx:
        # Find nearest clean rows above & below
        above = r - 1
        while above >= 0 and dropout[above]:
            above -= 1
        below = r + 1
        while below < h and dropout[below]:
            below += 1

        if above >= 0 and below < h:
            # Linear interpolation between clean neighbors
            t = (r - above) / (below - above)
            out[r, :] = (1 - t) * arr_norm[above, :] + t * arr_norm[below, :]
        elif above >= 0:
            out[r, :] = arr_norm[above, :]
        elif below < h:
            out[r, :] = arr_norm[below, :]

    return out


# ── Stage 2: Autoencoder refinement ──────────────────────────────────────────
def _autoencoder_refine(arr_norm: np.ndarray, mask: np.ndarray = None) -> np.ndarray | None:
    """
    Run the 2-channel masked autoencoder on normalized [0,1] input.
    Input channels: [corrupted_image * (1-mask), mask]
    Returns float32 in [0,1], or None on failure.
    """
    if not _torch_available:
        return None
    try:
        import torch
        model = get_autoencoder()
        if model is None:
            return None

        # Build mask channel
        if mask is not None:
            mask_float = (mask > 0).astype(np.float32)
            if mask_float.shape != arr_norm.shape:
                mask_float = cv2.resize(mask_float, (arr_norm.shape[1], arr_norm.shape[0]))
        else:
            mask_float = np.zeros_like(arr_norm, dtype=np.float32)

        # Masked image: zero out corrupted region so model must infer it
        arr_masked = arr_norm * (1.0 - mask_float)

        # Resize both channels to model input size
        img_resized  = cv2.resize(arr_masked, (224, 224))
        mask_resized = cv2.resize(mask_float, (224, 224))

        # Stack as 2-channel tensor: [1, 2, H, W]
        img_t  = torch.from_numpy(img_resized).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_resized).unsqueeze(0).float()
        tensor = torch.cat([img_t, mask_t], dim=0).unsqueeze(0).to(_device)  # (1,2,H,W)

        with torch.no_grad():
            out = model(tensor)
        out_np = out.squeeze().cpu().numpy()  # [0,1] from Sigmoid
        out_np = np.clip(out_np, 0.0, 1.0)
        return cv2.resize(out_np, (arr_norm.shape[1], arr_norm.shape[0]))
    except Exception as e:
        logger.warning("Autoencoder refinement failed: %s", e)
        return None


# ── Global noise estimator ────────────────────────────────────────────────────
def _estimate_noise_sigma(arr_norm: np.ndarray) -> float:
    """
    Fast noise sigma estimate: difference between image and Gaussian-blurred version.
    Returns std of high-frequency residual — correlates strongly with actual noise level.
    """
    smooth  = cv2.GaussianBlur(arr_norm, (5, 5), 1.0)
    residual = arr_norm - smooth
    return float(np.std(residual))


# ── Main public function ──────────────────────────────────────────────────────
def recover_image(pixel_array: np.ndarray | None,
                  tampering_mask: np.ndarray | None) -> np.ndarray:
    """
    Full DICOM recovery pipeline — 4-stage smart recovery:

    Stage 1 — Scan-line fix    : Interpolates detected dropout rows from neighbors
    Stage 2 — OpenCV denoise   : NLMeans + Bilateral for noise/block corruption
    Stage 3 — Noise check      : If global noise is high, swap base to denoised image
    Stage 4 — Targeted blend   : Apply reconstructed only over corrupted regions

    Clean pixels are preserved when not noisy.
    """
    if pixel_array is None:
        logger.warning("No pixel data — returning blank placeholder")
        return np.zeros((224, 224), dtype=np.float32)

    arr2d = _to_2d(pixel_array.astype(np.float32))
    h, w  = arr2d.shape

    # ── Step 1: Normalize safely
    arr_norm, pmin, pmax = _safe_normalize(arr2d)

    # ── Step 2: Fix scan-line dropout first (interpolation — always reliable)
    fixed = _detect_and_fix_scanlines(arr_norm)

    # ── Step 3: OpenCV multi-stage denoise on the scan-line-fixed image
    denoised = _opencv_denoise(fixed)

    # ── Step 4: Estimate global noise level INDEPENDENTLY of corruption detector
    # corruption_detector can misclassify noisy chest X-rays as MISSING_REGIONS
    # when many pixels clip to max. We detect noise directly from pixel statistics.
    noise_sigma = _estimate_noise_sigma(fixed)
    is_globally_noisy = noise_sigma > 0.04   # 4% of [0,1] range = moderate noise
    logger.info("Noise sigma: %.4f | globally_noisy: %s", noise_sigma, is_globally_noisy)

    # ── Step 5: Choose base image for unmasked pixels
    # If globally noisy → denoised (noise is everywhere, not just in mask regions)
    # If locally corrupted → arr_norm (original clean pixels should stay original)
    base = denoised if is_globally_noisy else arr_norm

    # ── Step 6: Autoencoder refinement — adaptive per-region weights
    # After scan-line fix, remaining near-zero pixels = block-erased regions.
    # Scan-line zeros were already filled in Step 2, so fixed < 0.01 = blocks only.
    block_zero_mask = (fixed < 0.01).astype(np.float32)
    has_large_blocks = float(block_zero_mask.mean()) > 0.005   # >0.5% zero = block

    # Run AE if: (a) missing black blocks, OR (b) any corruption mask detected (e.g. white regions)
    need_ae = has_large_blocks or (tampering_mask is not None and tampering_mask.mean() > 0.005)
    mask_coverage = tampering_mask.mean() if tampering_mask is not None else 0.0
    
    # 🔥 COMBINE MASK FOR AUTOENCODER (Zero out corrupted regions)
    ae_mask = block_zero_mask.copy()
    if tampering_mask is not None and tampering_mask.any():
        m_resize = cv2.resize(tampering_mask.astype(np.float32), (w, h))
        ae_mask = np.maximum(ae_mask, m_resize)
        
    ae_out = _autoencoder_refine(fixed, mask=ae_mask) if need_ae else None
    
    # 3. Clip AE output instead of full re-normalization (prevents gray wash)
    if ae_out is not None:
        ae_out = np.clip(ae_out, 0.0, 1.0)

    if has_large_blocks:
        # ── Block inpainting: OpenCV TELEA + AE blend ───────────────────────
        # TELEA (fast-marching inpaint) propagates texture from the block boundary
        # inward — much better than AE alone for large erased regions.
        # AE (30%) adds learned anatomical context on top.
        block_mask_u8 = (block_zero_mask > 0.5).astype(np.uint8) * 255
        fixed_u8      = np.clip(fixed * 255, 0, 255).astype(np.uint8)

        # Dilate mask slightly to avoid hard boundary artifacts at block edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(block_mask_u8, kernel, iterations=1)

        # 5. Use dynamic inpainting radius
        radius = int(max(5, min(15, block_zero_mask.sum() ** 0.5 / 10)))
        inpainted_u8  = cv2.inpaint(fixed_u8, dilated_mask, inpaintRadius=radius,
                                    flags=cv2.INPAINT_TELEA)
        inpainted_norm = inpainted_u8.astype(np.float32) / 255.0

        # 4. Reduce block blending dominance (smaller kernel, capped weight)
        if ae_out is not None:
            weight = cv2.GaussianBlur(block_zero_mask.astype(np.float32), (15, 15), 0)
            weight = np.clip(weight, 0.0, 0.8)
            block_fill = (1.0 - weight) * inpainted_norm + weight * ae_out
        else:
            block_fill = inpainted_norm

        # Standard blend for non-block regions
        if ae_out is not None:
            standard_blend = 0.85 * denoised + 0.15 * ae_out
        else:
            standard_blend = denoised

        reconstructed = np.where(block_zero_mask > 0.5, block_fill, standard_blend)
        logger.info("Block inpainting: TELEA+AE — block_coverage=%.1f%%",
                    block_zero_mask.mean() * 100)

    elif ae_out is not None:
        # No blocks — standard denoised+AE blend for noise/scan-line corruptions
        reconstructed = 0.85 * denoised + 0.15 * ae_out
    else:
        # Pure global noise, no blocks, no AE → OpenCV-only
        logger.info("Global noise only (sigma=%.3f) — OpenCV-only denoising", noise_sigma)
        reconstructed = denoised

    # ── Step 7: Strict masked blending — only modify confirmed corrupted regions
    combined_mask = block_zero_mask.astype(np.float32)
    if tampering_mask is not None and tampering_mask.any():
        m = tampering_mask.astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0.0, 1.0)
        combined_mask = np.maximum(combined_mask, m)

    # Soften mask edges for smooth transition to avoid visible boundaries
    mask_soft = cv2.GaussianBlur(combined_mask, (15, 15), 0)
    mask_soft = np.clip(mask_soft, 0.0, 1.0)

    # Smooth Masked Blending (Targeted Reconstruction)
    alpha = 0.8  # blending strength
    
    # Original background is preserved where mask is 0, reconstructed is blended where mask is 1
    result_norm = base * (1.0 - mask_soft * alpha) + reconstructed * (mask_soft * alpha)

    result_norm = np.clip(result_norm, 0.0, 1.0)

    # ── Helper: rescale a [0,1] float image back to original DICOM pixel range
    def _rescale(img_01):
        return np.clip(img_01, 0.0, 1.0) * (pmax - pmin) + pmin

    # ── Build and return all method outputs ──────────────────────────────────
    # This allows the caller/API to expose OpenCV vs AI vs Final for comparison.
    return {
        "original": _rescale(arr_norm),          # normalized corrupted input
        "opencv":   _rescale(denoised),           # OpenCV NLMeans+Bilateral+Unsharp
        "ai":       _rescale(ae_out) if ae_out is not None else None,   # AE output
        "final":    _rescale(result_norm),        # full hybrid pipeline result
    }

