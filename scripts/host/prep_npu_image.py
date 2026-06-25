#!/usr/bin/env python3
"""Prepare image for SmolVLM vision encoder NBG (int16 DFP, fl=15)."""
import sys
import struct
import numpy as np
from PIL import Image
from pathlib import Path

def prep_image(image_path, output_path, fl=15):
    """Resize 512x512, normalize to [-1,1], convert to int16 DFP."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr.reshape(1, 3, 512, 512)  # NCHW
    
    # Convert to int16 DFP with given fl
    scale = 2.0 ** fl
    int16_arr = np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)
    
    with open(output_path, 'wb') as f:
        int16_arr.tofile(f)
    
    print(f"Image: {image_path} -> {output_path}")
    print(f"  Shape: {int16_arr.shape}, dtype: {int16_arr.dtype}")
    print(f"  Range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Int range: [{int16_arr.min()}, {int16_arr.max()}]")
    print(f"  Size: {Path(output_path).stat().st_size} bytes")
    return output_path

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image.jpg> <output.dat> [fl=15]")
        sys.exit(1)
    fl = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    prep_image(sys.argv[1], sys.argv[2], fl)
