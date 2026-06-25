#!/usr/bin/env python3
"""V2d: Verify ONNX vs PyTorch and create calibration dataset."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import numpy as np
from pathlib import Path
import onnxruntime as ort
from PIL import Image

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
work_dir = Path("work/generated/smolvlm_256m_v2d")
work_dir.mkdir(parents=True, exist_ok=True)

# Load ONNX
onnx_path = Path("work/generated/smolvlm_256m_v2b/smolvlm_vision_v2b_final.onnx")
sess = ort.InferenceSession(str(onnx_path))

# Load PyTorch model for comparison
print("=== Loading PyTorch model ===")
from transformers import AutoModel
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

vm = model.vision_model
connector = model.connector

# Replace Conv2d with MatMul version
conv_w = vm.embeddings.patch_embedding.weight.data.clone()
conv_b = vm.embeddings.patch_embedding.bias.data.clone()

class PatchEmbedMatMul(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.out_channels = weight.shape[0]
        self.register_buffer("weight_flat", weight.reshape(768, 768))
        self.register_buffer("bias", bias)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 32, 16, 32, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, 1024, 768)
        x = torch.matmul(x, self.weight_flat.T) + self.bias
        x = x.transpose(1, 2).reshape(B, self.out_channels, 32, 32)
        return x

vm.embeddings.patch_embedding = PatchEmbedMatMul(conv_w, conv_b)

# Test images for calibration
test_images_dir = Path("work/generated/smolvlm_256m_v2d/test_images")

# Generate calibration data from test images + random uniform
# SigLIP preprocessing: resize 512x512, normalize mean=0.5 std=0.5
calib_data = []
calib_dir = work_dir / "calibration"
calib_dir.mkdir(parents=True, exist_ok=True)

# Use real test images for calibration (SigLIP normalization: mean=0.5, std=0.5 → range [-1,1])
print("\n=== Creating calibration dataset ===")
for i, img_path in enumerate(sorted(test_images_dir.glob("*"))):
    if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        img = Image.open(img_path).convert('RGB')
        # SigLIP preprocessing
        img_resized = img.resize((512, 512), Image.BICUBIC)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # normalize to [-1, 1]
        img_tensor = img_np.transpose(2, 0, 1)  # HWC → CHW
        img_tensor = np.expand_dims(img_tensor, 0)  # BCHW
        calib_data.append(img_tensor)
        np.save(calib_dir / f"calib_{i:03d}.npy", img_tensor)
        print(f"  calib_{i:03d}.npy: range [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

# Add uniform calibration samples
rng = np.random.RandomState(42)
for i in range(len(calib_data), 10):
    img_tensor = rng.uniform(-1.0, 1.0, (1, 3, 512, 512)).astype(np.float32)
    np.save(calib_dir / f"calib_{i:03d}.npy", img_tensor)
    calib_data.append(img_tensor)
    print(f"  calib_{i:03d}.npy: uniform [-1,1]")

# Create dataset.txt
dataset_path = work_dir / "dataset.txt"
lines = []
for i in range(len(calib_data)):
    lines.append(f"calibration/calib_{i:03d}.npy\n")
with open(dataset_path, 'w', newline='\n') as f:
    f.writelines(lines)
print(f"\nDataset: {dataset_path}")

# Verify ONNX vs PyTorch
print("\n=== ONNX vs PyTorch Verification ===")
class ExportWrapper(torch.nn.Module):
    def __init__(self, vision_model, connector):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
    def forward(self, pixel_values):
        x = self.vision_model(pixel_values).last_hidden_state
        x = self.connector(x)
        return x

wrapper = ExportWrapper(vm, connector)
wrapper.eval()

dummy = torch.tensor(calib_data[0])
with torch.no_grad():
    torch_out = wrapper(dummy).numpy()

onnx_out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
diff = np.abs(onnx_out - torch_out).max()
cos = np.dot(onnx_out.flatten(), torch_out.flatten()) / (np.linalg.norm(onnx_out.flatten()) * np.linalg.norm(torch_out.flatten()))
print(f"Max diff: {diff:.8f}")
print(f"Cosine: {cos:.8f}")

np.save(work_dir / "golden_output.npy", torch_out)
print(f"Golden output: {work_dir / 'golden_output.npy'}")

# Also copy ONNX to work dir for ACUITY
import shutil
shutil.copy2(onnx_path, work_dir / "smolvlm_vision_v2d.onnx")
print(f"\nONNX copied to: {work_dir / 'smolvlm_vision_v2d.onnx'}")
print("\n=== DONE - Ready for ACUITY conversion ===")
