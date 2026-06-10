"""
kaggle_train_autoencoder.py
===========================
Kaggle-optimised masked inpainting autoencoder for PhantomaShield.

Paste the contents of this file into a Kaggle notebook code cell.
Or upload this file as a dataset and run: !python kaggle_train_autoencoder.py

Kaggle paths used:
  Input data  : /kaggle/input/<your-dataset-name>/original/
  Best model  : /kaggle/working/encoder_inpaint_best.pth
  Last model  : /kaggle/working/encoder_inpaint_last.pth
"""

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pydicom

# ─── DEFAULTS (overridden by CLI args) ───────────────────────────────────────────────
DATA_DIR        = "/kaggle/input/datasets/simongraves/brain-mri-dataset/ST000001/SE000001"
MODEL_BEST_PATH = "/kaggle/working/encoder_best_v2.pth"
MODEL_LAST_PATH = "/kaggle/working/encoder_last_v2.pth"
RESUME_PATH     = "/kaggle/input/encoder-checkpoint/encoder_last_v2.pth"

# Fixed architectural constant — must be module-level because it is used as a
# default argument in CleanDICOMDataset.__init__, which Python evaluates at
# class-definition time (before argparse runs).
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CleanDICOMDataset(Dataset):
    """
    Loads CLEAN medical images and generates (inp=[corrupted,mask], mask, clean).
    Model learns masked inpainting: given (corrupted, mask) → reconstruct clean.
    """
    def __init__(self, root: str, img_size: int = IMG_SIZE,
                 augment: bool = True, limit: int | None = None):
        self.img_size = img_size
        self.augment  = augment
        self.files    = []
        for root_dir, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(root_dir, f)
                if f.lower().endswith((".dcm", ".npy")) or "." not in f:
                    self.files.append(fp)
        self.files = sorted(self.files)            # deterministic order
        if limit is not None:
            self.files = self.files[:limit]        # cap dataset size
        print(f"Dataset: {len(self.files)} images loaded"
              + (f" (limited from full set)" if limit is not None else ""))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr   = self._load(self.files[idx])
        clean = self._preprocess(arr)              # (1,H,W) float32 [0,1]
        if self.augment:
            clean = self._geometric_aug(clean)

        mask = self._random_mask(clean.shape[1], clean.shape[2])  # (1,H,W) soft [0,1]

        # ── Advanced corruption inside masked region (not just zero)
        import random
        corrupted = clean.clone()
        m = mask  # (1,H,W) in [0,1]

        # a) Zero-out masked pixels (base)
        corrupted = corrupted * (1.0 - m)

        # b) Add Gaussian noise in masked region (30% of samples)
        if self.augment and random.random() < 0.30:
            noise = torch.randn_like(corrupted) * random.uniform(0.05, 0.15)
            corrupted = corrupted + noise * m

        # c) Mild Gaussian blur bleed at mask boundary (20% of samples)
        if self.augment and random.random() < 0.20:
            import cv2
            c_np = corrupted.squeeze(0).numpy()
            c_np = cv2.GaussianBlur(c_np, (7, 7), 0)
            corrupted = torch.from_numpy(c_np).unsqueeze(0).float()

        corrupted = corrupted.clamp(0.0, 1.0)
        inp = torch.cat([corrupted, m], dim=0)     # (2,H,W)
        return inp, m, clean

    def _load(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        try:
            ds  = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, :, 0]
            return arr
        except Exception:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def _preprocess(self, arr):
        import cv2
        p2, p98 = np.percentile(arr, 2.0), np.percentile(arr, 98.0)
        arr = np.clip(arr, p2, p98)
        if p98 > p2:
            arr = (arr - p2) / (p98 - p2)
        else:
            arr = np.zeros_like(arr)
        arr = cv2.resize(arr.astype(np.float32), (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        return torch.from_numpy(arr).unsqueeze(0).float()

    def _geometric_aug(self, t):
        import random
        if random.random() > 0.5: t = torch.flip(t, dims=[2])
        if random.random() > 0.5: t = torch.flip(t, dims=[1])
        k = random.randint(0, 3)
        if k > 0: t = torch.rot90(t, k, dims=[1, 2])
        return t

    def _random_mask(self, h, w):
        """
        Advanced multi-strategy masking — simulates real-world corruption:
          1. Irregular blob masks (ellipses at random angles)
          2. Random brush-stroke lines
          3. Cutout blocks (small/medium/large — varied size)
          4. Combinations of the above
        Mask = 1.0 → corrupted region the model must reconstruct.
        """
        import cv2, random
        mask = np.zeros((h, w), dtype=np.float32)
        strategy = random.choices(
            ['blob', 'brush', 'block', 'mixed'],
            weights=[0.25, 0.25, 0.30, 0.20]
        )[0]

        if strategy in ('blob', 'mixed'):
            # 1–4 irregular ellipses
            for _ in range(random.randint(1, 4)):
                cx  = random.randint(w // 6, 5 * w // 6)
                cy  = random.randint(h // 6, 5 * h // 6)
                rx  = random.randint(w // 12, w // 4)
                ry  = random.randint(h // 12, h // 4)
                ang = random.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)

        if strategy in ('brush', 'mixed'):
            # 1–3 random brush strokes (thick polyline)
            for _ in range(random.randint(1, 3)):
                n_pts = random.randint(4, 10)
                pts   = np.array([
                    [random.randint(0, w), random.randint(0, h)]
                    for _ in range(n_pts)
                ], dtype=np.int32)
                thick = random.randint(h // 16, h // 6)
                cv2.polylines(mask, [pts], False, 1.0, thick)

        if strategy in ('block', 'mixed'):
            # 1–4 rectangular cutouts with varied size
            size_choice = random.choice(['small', 'medium', 'large'])
            count = {'small': 4, 'medium': 2, 'large': 1}[size_choice]
            for _ in range(count):
                if size_choice == 'small':
                    rh = random.randint(h // 16, h // 8)
                    rw = random.randint(w // 16, w // 8)
                elif size_choice == 'medium':
                    rh = random.randint(h // 8, h // 4)
                    rw = random.randint(w // 8, w // 4)
                else:
                    rh = random.randint(h // 4, h // 2)
                    rw = random.randint(w // 4, w // 2)
                sy = random.randint(0, max(1, h - rh))
                sx = random.randint(0, max(1, w - rw))
                mask[sy:sy+rh, sx:sx+rw] = 1.0

        # Soft edges — avoids hard checkerboard boundary
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.clip(mask, 0.0, 1.0)
        return torch.from_numpy(mask).unsqueeze(0).float()


# ─── Model ────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class ConvAutoencoder(nn.Module):
    """Residual U-Net for masked inpainting — 2-channel input: [corrupted, mask]."""
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.enc1 = ResBlock(in_channels, 32); self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ResBlock(32,  64);         self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ResBlock(64,  128);        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = ResBlock(128, 256);        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = ResBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = ResBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64,  2, stride=2); self.dec2 = ResBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64,  32,  2, stride=2); self.dec1 = ResBlock(64,  32)
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x);             e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# ─── Loss ─────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    Masked inpainting loss — gradient flows ONLY through corrupted pixels.

    Formula:
        MSE_masked  = mean( (pred - target)^2 * mask )   / (mean(mask) + eps)
        SSIM_masked = 1 - ssim computed on masked pixels only
        Loss        = 0.5 * MSE_masked + 0.5 * SSIM_masked

    This forces the model to learn WHAT to fill, not how to copy valid pixels.
    """
    def __init__(self):
        super().__init__()

    def _ssim(self, img1, img2, win: int = 11):
        """Pure-PyTorch SSIM — no external library required."""
        ch  = img1.size(1)
        # Gaussian kernel for local statistics
        coords = torch.arange(win, dtype=img1.dtype, device=img1.device) - win // 2
        g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
        kernel = kernel.expand(ch, 1, win, win)
        pad = win // 2

        mu1  = F.conv2d(img1,      kernel, padding=pad, groups=ch)
        mu2  = F.conv2d(img2,      kernel, padding=pad, groups=ch)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        s1   = F.conv2d(img1 * img1, kernel, padding=pad, groups=ch) - mu1_sq
        s2   = F.conv2d(img2 * img2, kernel, padding=pad, groups=ch) - mu2_sq
        s12  = F.conv2d(img1 * img2, kernel, padding=pad, groups=ch) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        num  = (2 * mu1 * mu2 + C1) * (2 * s12  + C2)
        den  = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        return (num / den.clamp(min=1e-8))   # (B,C,H,W) map in [-1,1]

    def forward(self, pred, target, mask=None):
        """
        pred, target, mask : (B, 1, H, W) float tensors in [0, 1].
        mask = 1 → corrupted region.  0 → valid pixel (no gradient here).
        """
        eps = 1e-6
        if mask is None:
            mask = torch.ones_like(pred)

        # ── Pure masked MSE  (only corrupted pixels contribute)
        sq_err  = (pred - target) ** 2          # (B,1,H,W)
        mse     = (sq_err * mask).sum() / (mask.sum() + eps)

        # ── Masked SSIM  (compare only inside masked region)
        ssim_map = self._ssim(pred, target)     # (B,1,H,W)
        # ssim_map in [-1,1]; (1 - ssim) is the dissimilarity loss
        ssim_loss = ((1.0 - ssim_map) * mask).sum() / (mask.sum() + eps)

        return 0.30 * mse + 0.40 * ssim_loss


# ─── Perceptual Loss (VGG16) ─────────────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss for grayscale medical images.

    Extracts feature maps from relu2_2 (layer idx 9) and relu3_3 (layer idx 16).
    Grayscale (1-ch) input is tiled to 3-ch and ImageNet-normalized before VGG.
    VGG weights are FROZEN — no gradient flows into VGG parameters.

    Loss = mean L1 distance between pred and target feature maps.
    This penalizes structure/texture mismatch, not just pixel error,
    which is the primary cause of over-smoothed / gray output.
    """
    # ImageNet statistics for grayscale → 3-ch normalization
    _MEAN = torch.tensor([0.485, 0.456, 0.406])
    _STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # relu2_2 = layers 0–9,  relu3_3 = layers 0–16
        self.slice1 = nn.Sequential(*list(vgg.children())[:10]).eval()   # up to relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[10:17]).eval() # up to relu3_3

        # Freeze all VGG parameters
        for p in self.parameters():
            p.requires_grad = False

    def _to_vgg_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B,1,H,W) grayscale [0,1] to (B,3,H,W) ImageNet-normalized."""
        x3 = x.repeat(1, 3, 1, 1)                          # tile to 3 channels
        mean = self._MEAN.to(x.device).view(1, 3, 1, 1)
        std  = self._STD .to(x.device).view(1, 3, 1, 1)
        return (x3 - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, 1, H, W) float [0, 1]
        Returns scalar perceptual L1 loss.
        """
        p = self._to_vgg_input(pred)
        t = self._to_vgg_input(target)

        # Feature maps at relu2_2
        f1_p = self.slice1(p)
        f1_t = self.slice1(t)
        loss  = F.l1_loss(f1_p, f1_t)

        # Feature maps at relu3_3 (passed through slice2)
        f2_p = self.slice2(f1_p)
        f2_t = self.slice2(f1_t)
        loss += F.l1_loss(f2_p, f2_t)

        return loss * 0.5   # average of both layers


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse == 0 else 20 * math.log10(1.0) - 10 * math.log10(mse)


# ─── LR Schedule ──────────────────────────────────────────────────────────────
def get_lr(epoch, warmup, total, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training ─────────────────────────────────────────────────────────────────
def train(args):
    dataset = CleanDICOMDataset(args.data, img_size=IMG_SIZE,
                                augment=True, limit=args.limit)
    if len(dataset) == 0:
        raise RuntimeError(f"No files found at: {args.data}")

    val_size   = max(2, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=num_workers)

    model = ConvAutoencoder(in_channels=2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── Resume from checkpoint ──────────────────────────────────────────
    if args.resume:
        _inner = model.module if isinstance(model, nn.DataParallel) else model
        ckpt = args.resume_path
        if os.path.exists(ckpt):
            try:
                state = torch.load(ckpt, map_location=device)
                _inner.load_state_dict(state, strict=True)
                print(f"\n✅ Resumed from: {ckpt}")
            except RuntimeError as e:
                print(f"\n⚠️  Resume FAILED (architecture mismatch): {e}")
                print(   "    Starting from scratch with random weights.")
        else:
            print(f"\n⚠️  --resume set but file not found: {ckpt}")
            print(   "    Starting from scratch.")

    criterion   = CombinedLoss().to(device)
    percept     = PerceptualLoss().to(device)
    optimizer   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model_best = args.out_best
    model_last = args.out_last
    os.makedirs(os.path.dirname(model_best), exist_ok=True)

    best_val, no_improve = float("inf"), 0
    total_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = get_lr(epoch - 1, args.warmup, args.epochs, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, (inp, mask, clean) in enumerate(train_loader):
            inp, mask, clean = inp.to(device), mask.to(device), clean.to(device)
            out  = model(inp)
            # Combined loss: 0.3 MSE + 0.4 SSIM + 0.3 Perceptual
            base_loss  = criterion(out, clean, mask)          # masked MSE + SSIM
            perc_loss  = percept(out, clean)                  # VGG feature match
            loss       = (base_loss + 0.3 * perc_loss) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * args.accum * inp.size(0)

        # ── Validate
        model.eval()
        val_loss, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for inp, mask, clean in val_loader:
                inp, mask, clean = inp.to(device), mask.to(device), clean.to(device)
                out = model(inp)
                val_loss += criterion(out, clean, mask).item() * inp.size(0)
                val_psnr += compute_psnr(out, clean) * inp.size(0)

        train_loss /= train_size
        val_loss   /= val_size
        val_psnr   /= val_size
        elapsed     = time.time() - t0
        eta_min     = ((time.time() - total_t0) / epoch) * (args.epochs - epoch) / 60

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"PSNR: {val_psnr:.2f}dB | LR: {lr:.6f} | "
              f"{elapsed:.0f}s | ETA: {eta_min:.0f}min")

        to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(to_save.state_dict(), model_last)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(to_save.state_dict(), model_best)
            print(f"  ✅ [BEST] val={val_loss:.5f} → saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  ⏹ Early stop after {args.patience} epochs without improvement.")
                break

    total_min = (time.time() - total_t0) / 60
    print(f"\nDone in {total_min:.1f} min | Best val: {best_val:.5f}")
    print(f"Best → {model_best}")
    print(f"Last → {model_last}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhantomaShield Masked Inpainting Autoencoder")
    parser.add_argument("--data",        default=DATA_DIR,        help="Path to clean training images (.npy or .dcm)")
    parser.add_argument("--epochs",      type=int,   default=80,  help="Max training epochs")
    parser.add_argument("--batch",       type=int,   default=16,  help="Batch size")
    parser.add_argument("--accum",       type=int,   default=2,   help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=5e-4,help="Peak learning rate")
    parser.add_argument("--warmup",      type=int,   default=5,   help="LR warmup epochs")
    parser.add_argument("--patience",    type=int,   default=20,  help="Early stopping patience")
    parser.add_argument("--resume",      action="store_true",     help="Resume from --resume-path checkpoint")
    parser.add_argument("--resume-path", default=RESUME_PATH,     help="Path to checkpoint .pth to resume from")
    parser.add_argument("--out-best",    default=MODEL_BEST_PATH, help="Output path for best model")
    parser.add_argument("--out-last",    default=MODEL_LAST_PATH, help="Output path for last model")
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Max images to load (default: all). E.g. --limit 10000")
    args = parser.parse_args()

    print(f"\nConfig:")
    print(f"  data        : {args.data}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch       : {args.batch}  (effective {args.batch * args.accum} with accum)")
    print(f"  lr          : {args.lr}")
    print(f"  resume      : {args.resume}")
    if args.resume:
        print(f"  resume-path : {args.resume_path}")
    print(f"  limit       : {args.limit if args.limit else 'all images'}")
    print(f"  out-best    : {args.out_best}")
    print(f"  out-last    : {args.out_last}\n")

    train(args)
