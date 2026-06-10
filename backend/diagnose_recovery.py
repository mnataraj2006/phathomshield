"""
diagnose_recovery.py
Runs a deep check on the recovery pipeline to find why output looks same as input.
"""
import sys, os
import numpy as np
import torch

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "autoencoder_dicom.pth")
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("PHANTOMASHIELD — Recovery Diagnostics")
print("=" * 60)

# ── 1. Check model file
print(f"\n[1] Model file: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"    EXISTS  |  Size: {size_mb:.1f} MB")
    mtime = os.path.getmtime(MODEL_PATH)
    import datetime
    print(f"    Modified: {datetime.datetime.fromtimestamp(mtime)}")
else:
    print("    ERROR: Model file NOT FOUND!")
    sys.exit(1)

# ── 2. Load model and check state_dict
print("\n[2] Attempting to load model weights...")
device = torch.device("cpu")
try:
    state = torch.load(MODEL_PATH, map_location=device)
    keys = list(state.keys())
    print(f"    Loaded OK  |  {len(keys)} weight tensors")
    print(f"    First 5 keys: {keys[:5]}")
    # Check if it's ResUNet (enc1) or old ConvAutoencoder (encoder)
    if any("enc1" in k for k in keys):
        print("    Architecture: ResUNet (CORRECT)")
    elif any("encoder" in k for k in keys):
        print("    Architecture: OLD ConvAutoencoder — MISMATCH with current code!")
    else:
        print(f"    Architecture: UNKNOWN keys — {keys[:3]}")
except Exception as e:
    print(f"    FAILED to load: {e}")
    sys.exit(1)

# ── 3. Build model and try load_state_dict
print("\n[3] Building model and loading weights...")
from recovery_engine import ConvAutoencoder, _torch_available

if not _torch_available:
    print("    PyTorch not available!")
    sys.exit(1)

model = ConvAutoencoder()
try:
    model.load_state_dict(state)
    print("    load_state_dict: SUCCESS — architecture matches")
except Exception as e:
    print(f"    load_state_dict: FAILED — {e}")
    print("    This means architecture mismatch — model uses wrong weights!")
    sys.exit(1)

model.eval()

# ── 4. Test forward pass variance
print("\n[4] Running forward pass with dummy noisy input...")
noisy = torch.rand(1, 1, 224, 224)  # simulate pure noise

with torch.no_grad():
    out = model(noisy)

out_np = out.squeeze().numpy()
in_np = noisy.squeeze().numpy()

print(f"    Input  — mean: {in_np.mean():.4f}, std: {in_np.std():.4f}")
print(f"    Output — mean: {out_np.mean():.4f}, std: {out_np.std():.4f}")

diff = np.abs(out_np - in_np)
print(f"    Diff   — mean: {diff.mean():.4f}, max: {diff.max():.4f}")

if diff.mean() < 0.005:
    print("    WARNING: Output is nearly IDENTICAL to input — model doing nothing!")
elif diff.mean() < 0.03:
    print("    NOTE: Subtle denoising only (trained only 3 epochs, minimal change)")
else:
    print("    Good: Model is actively transforming the input")

# ── 5. Test with a 'clean-ish' input
print("\n[5] Running forward pass with smooth (low-noise) input...")
smooth = torch.zeros(1, 1, 224, 224)
smooth[0, 0, 50:170, 50:170] = 0.7  # simulate a bright region
smooth += torch.randn_like(smooth) * 0.05  # tiny noise
smooth = smooth.clamp(0, 1)

with torch.no_grad():
    out2 = model(smooth)

out2_np = out2.squeeze().numpy()
diff2 = np.abs(out2_np - smooth.squeeze().numpy())
print(f"    Input  — mean: {smooth.mean():.4f}, std: {smooth.std():.4f}")
print(f"    Output — mean: {out2_np.mean():.4f}, std: {out2_np.std():.4f}")
print(f"    Diff   — mean: {diff2.mean():.4f}")

# ── 6. Test _normalize with extreme values
print("\n[6] Testing _normalize with extreme/noisy DICOM-like values...")
from recovery_engine import _normalize
fake_dicom = np.random.randn(512, 512).astype(np.float32) * 1000  # extreme spread
norm, pmin, pmax = _normalize(fake_dicom)
print(f"    Raw    — min: {fake_dicom.min():.1f}, max: {fake_dicom.max():.1f}")
print(f"    pmin (2nd pct): {pmin:.1f}  |  pmax (98th pct): {pmax:.1f}")
print(f"    Normalized — min: {norm.min():.4f}, max: {norm.max():.4f}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
