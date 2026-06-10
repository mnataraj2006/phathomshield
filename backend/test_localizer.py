import sys
import numpy as np
import cv2
import torch
from localizer import generate_heatmap_full
import logging

logging.basicConfig(level=logging.INFO)

print("\n--- NORMAL IMAGE TEST ---")
normal_image = np.zeros((224, 224), dtype=np.float32)
# Adding some tiny randomness to avoid totally constant inputs
normal_image += np.random.randn(*normal_image.shape) * 1.0 
result_normal = generate_heatmap_full(normal_image, class_idx=1)

print("\n--- TAMPERED IMAGE TEST ---")
tampered_image = np.zeros((224, 224), dtype=np.float32)
tampered_image[100:150, 100:150] = 200.0  # Big box of high signal
result_tampered = generate_heatmap_full(tampered_image, class_idx=1)

print("\nDONE.")
