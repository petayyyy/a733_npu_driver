#!/usr/bin/env python3
"""Create calibration dataset for SmolVLM vision encoder (LF line endings)."""
import numpy as np
from pathlib import Path

work_dir = Path("work/generated/smolvlm_256m_vision_encoder")
calib_dir = work_dir / "calibration"
calib_dir.mkdir(parents=True, exist_ok=True)

num_samples = 8
for i in range(num_samples):
    sample = np.random.randn(1, 3, 512, 512).astype(np.float32)
    np.save(calib_dir / f"sample_{i}.npy", sample)
    print(f"Sample {i}: shape {sample.shape}")

# Write dataset.txt with LF line endings
dataset_path = work_dir / "dataset_fixed.txt"
with open(dataset_path, "w", newline="\n") as f:
    for i in range(num_samples):
        f.write(f"calibration/sample_{i}.npy\n")

print(f"\nDataset: {dataset_path}")
print(f"Line endings: LF")
