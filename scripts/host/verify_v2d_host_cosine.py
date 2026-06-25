#!/usr/bin/env python3
"""V2d: Compare ACUITY host int16 output vs PyTorch FP32 golden."""
import numpy as np
from pathlib import Path

pkg = Path("work/model-packages/smolvlm_256m_vision_v2d/int16")
golden = np.load("work/generated/smolvlm_256m_v2d/golden_output.npy")

# Read ACUITY host output
host_vals = []
with open(pkg / "host_output_0.txt", "r") as f:
    for line in f:
        try:
            host_vals.append(float(line.strip()))
        except ValueError:
            pass
host = np.array(host_vals, dtype=np.float32)

expected_shape = (1, 64, 576)
host = host.reshape(expected_shape)

print(f"Golden shape: {golden.shape}")
print(f"Host shape:   {host.shape}")

diff = np.abs(host - golden).max()
mean_diff = np.abs(host - golden).mean()
cos = np.dot(host.flatten(), golden.flatten()) / (
    np.linalg.norm(host.flatten()) * np.linalg.norm(golden.flatten())
)

print(f"Max abs diff:  {diff:.8f}")
print(f"Mean abs diff: {mean_diff:.8f}")
print(f"Cosine:        {cos:.8f}")

# Per-token cosine (64 tokens, 576 dims each)
per_token_cos = []
for i in range(64):
    h = host[0, i, :]
    g = golden[0, i, :]
    c = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
    per_token_cos.append(c)

print(f"Per-token cosines: min={min(per_token_cos):.4f}, mean={np.mean(per_token_cos):.4f}, max={max(per_token_cos):.4f}")

# Quality gate
if cos > 0.95:
    print(f"\nHOST QUALITY GATE: PASSED (cosine {cos:.4f} > 0.95)")
else:
    print(f"\nHOST QUALITY GATE: FAILED (cosine {cos:.4f} <= 0.95)")
