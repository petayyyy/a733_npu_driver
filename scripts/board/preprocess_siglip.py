#!/usr/bin/env python3
"""Preprocess image for SmolVLM SigLIP NBG and run on NPU."""
import sys, os, struct
import numpy as np
from pathlib import Path

def preprocess_image(image_path, output_dat, size=512, fl=15):
    """Preprocess image to NBG input format."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL not available. Install with: pip install Pillow")
        sys.exit(1)
    
    # Load and resize
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    
    # Convert to numpy float32 [0, 1]
    pixels = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize: (pixel - 0.5) / 0.5 = pixel * 2 - 1 → [-1, 1]
    pixels = pixels * 2.0 - 1.0
    
    # Rearrange from HWC to CHW
    pixels = np.transpose(pixels, (2, 0, 1))  # [3, 512, 512]
    
    # Add batch dim to match ACUITY format [1, 1, 3, 512, 512]
    # Wait - ACUITY format adds EXTRA dim. The actual NBG input shape is [1,1,3,512,512]
    # Let me check the ACUITY preprocess format by looking at vnn_pre_process.c
    # Actually, vpm_run expects the data in a specific binary format
    # The input_0.dat from ACUITY is the quantized int16 data in NHWC or NCHW layout
    
    # The nbg_meta.json says format is "nchw" with shape [1, 1, 3, 512, 512]
    # So we need: [1, 1, 3, 512, 512] with quant_format=1 (dynamic_fixed_point)
    # Add extra dimension
    pixels = pixels[np.newaxis, np.newaxis, :, :, :]  # [1, 1, 3, 512, 512]
    
    # Quantize to int16 DFP
    scale = 2.0 ** fl  # 32768
    quantized = np.round(pixels * scale).clip(-32768, 32767).astype(np.int16)
    
    # Write as binary .dat file (vpm_run format)
    # The exact format depends on what ACUITY's preprocess generates
    # Let me check by reading the existing input_0.dat
    quantized.tofile(output_dat)
    
    print(f"Input shape: {quantized.shape}")
    print(f"Input range: [{quantized.min()}, {quantized.max()}]")
    print(f"Saved: {output_dat} ({os.path.getsize(output_dat)} bytes)")
    return quantized

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image.jpg> [output.dat]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dat = sys.argv[2] if len(sys.argv) > 2 else 'input_test.dat'
    
    preprocess_image(image_path, output_dat)
