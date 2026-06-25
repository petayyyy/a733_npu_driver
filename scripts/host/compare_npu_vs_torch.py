#!/usr/bin/env python3
"""Compare NPU int16 output vs PyTorch FP32 for same dog.jpg input."""
import os; os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

# Load NPU input (int16, fl=15)
npu_input_int16 = np.fromfile('work/generated/smolvlm_256m_v2c/dog_real_input.dat', dtype=np.int16)
npu_input_f32 = npu_input_int16.astype(np.float32) / (2.0**15)
npu_input_f32 = npu_input_f32.reshape(1, 1, 3, 512, 512)
print(f'NPU input shape: {npu_input_f32.shape}, range: [{npu_input_f32.min():.4f}, {npu_input_f32.max():.4f}]')

# Load NPU output (float32 dequantized by vpm_run)
npu_output = np.loadtxt('work/generated/smolvlm_256m_v2c/npu_dog_output.txt', dtype=np.float32)
npu_output = npu_output.reshape(1, 64, 576)
print(f'NPU output shape: {npu_output.shape}')

# PyTorch reference
model = AutoModel.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct', trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

class PatchEmbedMatMul(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.out_channels = weight.shape[0]
        self.register_buffer('weight_flat', weight.reshape(768, 768))
        self.register_buffer('bias', bias)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 32, 16, 32, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, 1024, 768)
        x = torch.matmul(x, self.weight_flat.T) + self.bias
        x = x.transpose(1, 2).reshape(B, self.out_channels, 32, 32)
        return x

conv_w = model.vision_model.embeddings.patch_embedding.weight.data.clone()
conv_b = model.vision_model.embeddings.patch_embedding.bias.data.clone()
model.vision_model.embeddings.patch_embedding = PatchEmbedMatMul(conv_w, conv_b)

# Remove ACUITY extra batch dim: [1,1,3,512,512] -> [1,3,512,512]
torch_input = torch.from_numpy(npu_input_f32[:, 0, :, :, :])

with torch.no_grad():
    vm_out = model.vision_model(torch_input)
    conn_out = model.connector(vm_out.last_hidden_state)
    torch_output = conn_out.numpy()

print(f'PyTorch output shape: {torch_output.shape}')

# Compare
diff = np.abs(npu_output - torch_output)
max_diff = diff.max()
mean_diff = diff.mean()
cos = np.dot(npu_output.flatten(), torch_output.flatten()) / (
    np.linalg.norm(npu_output.flatten()) * np.linalg.norm(torch_output.flatten())
)

print(f'\n=== NPU int16 vs PyTorch FP32 (SAME dog.jpg) ===')
print(f'Max diff: {max_diff:.4f}')
print(f'Mean diff: {mean_diff:.4f}')
print(f'Cosine:   {cos:.8f}')
gate = "PASSED" if cos > 0.95 else "FAILED"
print(f'GATE (>0.95): {gate}')

# Per-token cosine
token_cos = []
for i in range(64):
    c = np.dot(npu_output[0,i], torch_output[0,i]) / (
        np.linalg.norm(npu_output[0,i]) * np.linalg.norm(torch_output[0,i])
    )
    token_cos.append(c)
print(f'Per-token cosine: min={min(token_cos):.4f}, mean={np.mean(token_cos):.4f}, max={max(token_cos):.4f}')

# Save results
np.save('work/generated/smolvlm_256m_v2c/npu_vs_torch_diff.npy', diff)
print('\nResults saved.')
