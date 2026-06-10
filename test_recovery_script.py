import sys
import os
import torch
import numpy as np
import cv2

# Add backend directory to sys.path
sys.path.append(os.path.abspath('backend'))

from recovery_engine import get_autoencoder, recover_image, _torch_available

print("PyTorch available:", _torch_available)

if _torch_available:
    model = get_autoencoder()
    print("Model loaded successfully:", model is not None)

    # Test with dummy data
    dummy_pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
    
    # Dummy corruption mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:300, 200:300] = 1

    print("Attempting to recover image...")
    recovered = recover_image(dummy_pixel_array, mask)

    if recovered is not None:
        print("Recovered image shape:", recovered.shape)
        print("Recovery engine is functioning properly.")
    else:
        print("Recovery image returned None!")
else:
    print("PyTorch not installed, using OpenCV fallback")
