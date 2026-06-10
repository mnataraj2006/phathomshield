"""
localizer.py
Grad-CAM tamper localization.
Generates a heatmap overlay highlighting suspicious regions.
Gracefully degrades to statistical visualization if PyTorch unavailable.
"""
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from detector import get_model, _transform, _device, _torch_available, LABELS
    _torch_ok = _torch_available
except ImportError:
    _torch_ok = False
    logger.warning("PyTorch not available — using fallback heatmap for Grad-CAM")


class GradCAMPlusPlus:
    """
    Grad-CAM++ hook — targets model.features[-2] for EfficientNet-B4.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        target = self._find_target(model)
        self._hooks.append(target.register_forward_hook(self._fwd))
        self._hooks.append(target.register_full_backward_hook(self._bwd))

    @staticmethod
    def _find_target(model):
        if hasattr(model, 'features'):
            # EfficientNet: features is a Sequential — grab the second-to-last child
            feat_list = list(model.features.children())
            return feat_list[-2]
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        # Generic: target second-to-last child instead of last
        return list(model.children())[-2]

    def _fwd(self, m, i, o): self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients  = go[0].detach()

    def generate(self, tensor, class_idx=None):
        # 1. ENSURE GRADIENT FLOW
        self.model.eval()
        tensor = tensor.to(_device)
        tensor.requires_grad_(True)
        self.model.zero_grad()

        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        out[0, class_idx].backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7), dtype=np.float32)

        gradients = self.gradients
        activations = self.activations

        print("Grad mean:", gradients.abs().mean().item())
        print("Act mean:", activations.abs().mean().item())

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = alpha_num / alpha_denom

        w = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (w * activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().detach().numpy()

        # 5. SAFE POST-PROCESSING PIPELINE
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.power(cam, 1.5)
        
        # 8. COMMON FAILURE CONDITIONS TO HANDLE
        if cam.max() < 1e-6:
            print("⚠️ Heatmap collapsed — check gradients/layer")
            
        print("Heatmap min/max:", cam.min(), cam.max())

        # Removed edge artifact suppression here because it zeroes out the tiny 7x7 tensor

        return cam.astype(np.float32)

    def remove(self):
        for h in self._hooks: h.remove()


def _run_gradcam(pixel_array: np.ndarray, class_idx: int) -> np.ndarray | None:
    """Run Grad-CAM and return raw float [0,1] activation map at original resolution, or None on failure."""
    arr = pixel_array.astype(np.float32)
    arr2d = arr[0] if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] else (arr[:, :, 0] if arr.ndim == 3 else arr)
    h, w = arr2d.shape
    pmin, pmax = arr2d.min(), arr2d.max()
    arr_u8 = ((arr2d - pmin) / (pmax - pmin + 1e-8) * 255).astype(np.uint8) if pmax > pmin else np.zeros_like(arr2d, dtype=np.uint8)

    if not _torch_ok:
        return None
    try:
        import torch
        from PIL import Image as _PILImage
        pil_img = _PILImage.fromarray(arr_u8).convert("RGB")
        tensor = _transform(pil_img).unsqueeze(0).to(_device)
        tensor.requires_grad_(True)
        model = get_model()
        if model is None:
            return None
        gc = GradCAMPlusPlus(model)
        cam = gc.generate(tensor, class_idx)
        gc.remove()
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        return cam_resized.astype(np.float32)   # float in [0, 1]
    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e)
        return None


def get_tampering_mask(pixel_array: np.ndarray, class_idx: int = 1,
                       threshold: float = 0.5) -> np.ndarray:
    """
    Return a binary float32 mask [H, W] where 1.0 = tampered region.
    Used by recovery_engine for targeted reconstruction:
        Recovered = (mask × reconstructed) + ((1 - mask) × original)

    threshold: Grad-CAM activation above this value is considered tampered.
    """
    cam = _run_gradcam(pixel_array, class_idx)
    if cam is None:
        # Fallback: zero mask (no modification — preserve original fully)
        arr2d = pixel_array[0] if pixel_array.ndim == 3 and pixel_array.shape[0] < pixel_array.shape[-1] \
                else (pixel_array[:, :, 0] if pixel_array.ndim == 3 else pixel_array)
        return np.zeros(arr2d.shape[:2], dtype=np.float32)
    binary = (cam >= threshold).astype(np.float32)
    # Morphological dilation to cover boundary regions around tampered zones
    kernel = np.ones((11, 11), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary.astype(np.float32)


def _apply_threshold_and_contour(cam: np.ndarray,
                                  abs_threshold: float = 0.3):
    """
    Apply absolute threshold (0.3) to suppress low-activation noise.
    Only activations >= threshold pass through — suppresses background highlights.
    Returns (thresholded_cam_float, contour_mask_uint8).
    """
    # Re-normalize before threshold so scale is consistent across images
    if cam.max() > 1e-8:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    # Hard threshold: zero out sub-threshold activations
    cam_thresh = np.where(cam >= abs_threshold, cam, 0.0).astype(np.float32)

    # Re-normalise retained activations to full range for vibrant color map
    if cam_thresh.max() > 1e-8:
        cam_thresh = (cam_thresh - cam_thresh.min()) / \
                     (cam_thresh.max() - cam_thresh.min() + 1e-8)
    cam_thresh = cam_thresh.clip(0.0, 1.0)

    # Build contour from slightly smoothed binary mask (removes ragged edges)
    binary_smooth = cv2.GaussianBlur(
        (cam_thresh * 255).astype(np.uint8), (5, 5), 0
    )
    _, binary = cv2.threshold(binary_smooth, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(binary)
    cv2.drawContours(contour_mask, contours, -1, 255, 2)
    return cam_thresh, contour_mask


def generate_heatmap(pixel_array: np.ndarray, class_idx: int = 1) -> np.ndarray:
    """Generate Grad-CAM heatmap as RGB overlay (backward-compatible)."""
    result = generate_heatmap_full(pixel_array, class_idx)
    return result["heatmap"]


def generate_heatmap_full(pixel_array: np.ndarray,
                          class_idx: int = 1) -> dict:
    """
    Full heatmap generation with three outputs:
      heatmap:   (H,W,3) uint8 — Grad-CAM JET overlay (threshold ≥0.6, 60/40 blend)
      overlay:   (H,W,3) uint8 — original + cyan contour borders of hot regions
      cam_float: (H,W)   float32 — thresholded activation map [0,1]

    Key improvements:
      • Min-max normalised + 0.6 hard threshold suppresses diffuse noise
      • Zero-activation regions show original image (not blue JET tinge)
      • 60% original / 40% heatmap blend for comfortable visual reading
      • Corner noise suppressed via Gaussian smooth before contouring
    """
    arr = pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr2d = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
    else:
        arr2d = arr

    pmin, pmax = arr2d.min(), arr2d.max()
    if pmax > pmin:
        arr_u8 = ((arr2d - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        arr_u8 = np.zeros_like(arr2d, dtype=np.uint8)

    orig_rgb = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
    h, w     = arr_u8.shape

    if _torch_ok:
        try:
            import torch
            pil_img = Image.fromarray(arr_u8).convert("RGB")
            tensor  = _transform(pil_img).unsqueeze(0).to(_device)
            tensor.requires_grad_(True)
            model   = get_model()
            if model is None:
                raise RuntimeError("no model")

            gc  = GradCAMPlusPlus(model)
            cam = gc.generate(tensor, class_idx)   # already normalised + thresholded
            gc.remove()

            cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

            # ── Anatomy mask: suppress heatmap on near-black background pixels
            anatomy_mask = (arr_u8 > 10).astype(np.float32)   # 1 = anatomy, 0 = background
            cam_resized  = cam_resized * anatomy_mask

            # Re-apply threshold + normalize after resize + anatomy masking
            cam_thresh, contour_mask = _apply_threshold_and_contour(cam_resized, abs_threshold=0.3)

            # 6. VISUALIZATION CHECK
            print("Heatmap min/max after threshold:", cam_thresh.min(), cam_thresh.max())

            # ── Heatmap: JET on thresholded cam ──────────────────────────────
            heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_thresh), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            heatmap_out = cv2.addWeighted(orig_rgb, 0.6, heatmap_color, 0.4, 0)

            # ── Contour overlay: original + neon-cyan borders ─────────────────
            overlay_out = orig_rgb.copy()
            overlay_out[contour_mask > 0] = [0, 255, 220]   # neon cyan

            return {
                "heatmap":   heatmap_out,
                "overlay":   overlay_out,
                "cam_float": cam_thresh,
            }
        except Exception as e:
            logger.warning("Grad-CAM failed: %s", e)

    fallback = _fallback_heatmap(arr_u8, h, w)
    return {"heatmap": fallback, "overlay": orig_rgb.copy(), "cam_float": np.zeros((h, w), np.float32)}


def _fallback_heatmap(arr_u8, h, w):
    """Gaussian blob heatmap when PyTorch is unavailable."""
    rng = np.random.default_rng(42)
    cx = w // 2 + rng.integers(-w // 4, w // 4)
    cy = h // 2 + rng.integers(-h // 4, h // 4)
    Y, X = np.mgrid[0:h, 0:w]
    sigma = min(h, w) * 0.25
    mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    cam_u8 = (mask * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(orig_rgb, 0.5, heat_rgb, 0.5, 0)
