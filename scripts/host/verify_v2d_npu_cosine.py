#!/usr/bin/env python3
"""V2d: Compare NPU int16 output vs PyTorch FP32 golden."""
import numpy as np
from pathlib import Path

golden = np.load("work/generated/smolvlm_256m_v2d/golden_output.npy")

# Read NPU output (dequantized int16 DFP fl=8)
npu_vals = []
with open("work/generated/smolvlm_256m_v2d/npu_output_dog.txt", "r") as f:
    for line in f:
        line = line.strip()
        if ':' in line:
            line = line.split(':', 1)[1]
        for x in line.replace(',', ' ').split():
            try:
                npu_vals.append(float(x))
            except:
                pass

scale = 2.0 ** 8  # fl=8
npu = np.array(npu_vals, dtype=np.float32) / scale
npu = npu.reshape(1, 64, 576)

print(f"Golden shape: {golden.shape}")
print(f"NPU shape:    {npu.shape}")

diff = np.abs(npu - golden).max()
mean_diff = np.abs(npu - golden).mean()
cos = np.dot(npu.flatten(), golden.flatten()) / (
    np.linalg.norm(npu.flatten()) * np.linalg.norm(golden.flatten())
)

print(f"Max abs diff:  {diff:.8f}")
print(f"Mean abs diff: {mean_diff:.8f}")
print(f"Cosine:        {cos:.8f}")

# Per-token cosine
per_token_cos = []
for i in range(64):
    h = npu[0, i, :]
    g = golden[0, i, :]
    c = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
    per_token_cos.append(c)

print(f"Per-token cos: min={min(per_token_cos):.4f}, mean={np.mean(per_token_cos):.4f}, max={max(per_token_cos):.4f}")

# Quality gate
if cos > 0.95:
    print(f"\nBOARD QUALITY GATE: PASSED (cosine {cos:.4f} > 0.95)")
else:
    print(f"\nBOARD QUALITY GATE: FAILED (cosine {cos:.4f} <= 0.95)")
