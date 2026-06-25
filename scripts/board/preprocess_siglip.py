#!/usr/bin/env python3
"""Preprocess image for SmolVLM SigLIP NBG using OpenCV."""
import sys, os
import numpy as np
import cv2

def preprocess_image(image_path, output_dat, size=512, fl=15):
    # Load and resize
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load {image_path}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 [0, 255]
    pixels = img.astype(np.float32)
    
    # Normalize: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1 -> [-1, 1]
    pixels = pixels / 127.5 - 1.0
    
    # HWC to CHW, add extra dim for ACUITY format [1, 1, 3, 512, 512]
    pixels = np.transpose(pixels, (2, 0, 1))  # [3, 512, 512]
    pixels = pixels[np.newaxis, np.newaxis, :, :, :]  # [1, 1, 3, 512, 512]
    
    # Quantize to int16 DFP
    scale = 2.0 ** fl  # 32768 for fl=15
    quantized = np.round(pixels * scale).clip(-32768, 32767).astype(np.int16)
    
    # Write as raw int16 binary (NCHW layout, matching ACUITY format)
    quantized.tofile(output_dat)
    
    print(f"Image: {image_path}")
    print(f"Shape: {img.shape} -> normalized [{pixels.min():.3f}, {pixels.max():.3f}]")
    print(f"Quantized int16: [{quantized.min()}, {quantized.max()}], fl={fl}")
    print(f"Saved: {output_dat} ({os.path.getsize(output_dat)} bytes)")
    return quantized

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image.jpg> [output.dat] [fl=15]")
        sys.exit(1)
    image_path = sys.argv[1]
    output_dat = sys.argv[2] if len(sys.argv) > 2 else 'input_test.dat'
    fl = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    preprocess_image(image_path, output_dat, fl=fl)
