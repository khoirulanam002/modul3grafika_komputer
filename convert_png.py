import os
import numpy as np
from PIL import Image
from ppm_io import read_ppm

out_dir = "out"
for fname in sorted(os.listdir(out_dir)):
    if not fname.endswith(".ppm"):
        continue
    path = os.path.join(out_dir, fname)
    arr = read_ppm(path)
    png_path = path.replace(".ppm", ".png")
    if arr.ndim == 3:
        # RGB color image
        img = Image.fromarray(arr, mode="RGB")
    else:
        # Grayscale depth map - normalize to 0-255
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(arr_norm, mode="L")
    img.save(png_path)
    print(f"converted: {fname} {arr.shape} -> {os.path.basename(png_path)}")
