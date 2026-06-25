#!/usr/bin/env python3
"""V2b Attempt 1 v2: Replace only the Conv2d, keep Idefics3 wrapper intact."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import onnx
from onnx import helper, numpy_helper

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
work_dir = Path("work/generated/smolvlm_256m_v2b")
work_dir.mkdir(parents=True, exist_ok=True)

print("=== Loading model ===")
from transformers import AutoModel
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

vm = model.vision_model
connector = model.connector

# Get original Conv weight/bias
conv_w = vm.embeddings.patch_embedding.weight.data.clone()
conv_b = vm.embeddings.patch_embedding.bias.data.clone()

# Step 1: Define Conv2d replacement
class PatchEmbedMatMul(nn.Module):
    """Drop-in replacement for patch-embedding Conv2d using Reshape+MatMul."""
    def __init__(self, weight, bias):
        super().__init__()
        # weight: [768, 3, 16, 16]
        self.out_channels = weight.shape[0]
        self.kernel_size = (weight.shape[2], weight.shape[3])
        self.stride = (16, 16)
        self.padding = (0, 0)
        self.register_buffer("weight_flat", weight.reshape(768, 768))
        self.register_buffer("bias", bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Patchify: extract 16x16 non-overlapping patches
        # H=512, W=512, kernel=16 -> 32x32 patches
        x = x.reshape(B, C, 32, 16, 32, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, 32, 32, 3, 16, 16]
        x = x.reshape(B, 1024, 768)        # [B, 1024, 768]
        # MatMul
        x = torch.matmul(x, self.weight_flat.T) + self.bias  # [B, 1024, 768]
        # Reshape back to Conv2d output format: [B, 768, 32, 32]
        x = x.transpose(1, 2).reshape(B, self.out_channels, 32, 32)
        return x

# Step 2: Verify replacement matches Conv2d
print("\n=== Verifying MatMul replacement ===")
dummy = torch.randn(1, 3, 512, 512)

replacement = PatchEmbedMatMul(conv_w, conv_b)
with torch.no_grad():
    conv_out = F.conv2d(dummy, conv_w, conv_b, stride=16)
    repl_out = replacement(dummy)
    
    diff = (conv_out - repl_out).abs().max().item()
    print(f"Max diff: {diff:.10f}")
    assert diff < 1e-4, f"Mismatch: {diff}"
    print("VERIFIED: MatMul replacement matches Conv2d exactly!")

# Step 3: Replace the Conv2d in the model
print("\n=== Replacing Conv2d in model ===")
vm.embeddings.patch_embedding = replacement

# Step 4: Verify full model output matches
print("\n=== Verifying full model output ===")
with torch.no_grad():
    original = vm.embeddings.patch_embedding  # keep reference to replacement
    
    out_vm = vm(dummy)
    out_conn = connector(out_vm.last_hidden_state)
    
    # Restore original to compare
    vm.embeddings.patch_embedding = nn.Conv2d(3, 768, 16, stride=16, bias=True)
    vm.embeddings.patch_embedding.weight.data = conv_w
    vm.embeddings.patch_embedding.bias.data = conv_b
    
    orig_vm = vm(dummy)
    orig_conn = connector(orig_vm.last_hidden_state)
    
    diff = (out_conn - orig_conn).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(out_conn.flatten(), orig_conn.flatten(), dim=0).item()
    print(f"Full model max diff: {diff:.8f}")
    print(f"Full model cosine: {cosine:.8f}")
    
    # Restore replacement for export
    vm.embeddings.patch_embedding = replacement

assert diff < 1e-4, f"Full model mismatch: diff={diff}"
print("VERIFIED: Full model with MatMul patch embed matches original!")

# Step 5: Export to ONNX
print("\n=== Exporting to ONNX ===")

class ExportWrapper(nn.Module):
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

onnx_path = work_dir / "smolvlm_vision_v2b_raw.onnx"
torch.onnx.export(
    wrapper,
    dummy,
    str(onnx_path),
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    opset_version=17,
    do_constant_folding=True,
)
print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size/(1024**2):.1f} MB)")

# Step 6: Remove NonZero from ONNX
print("\n=== Removing NonZero ===")
m = onnx.load(str(onnx_path))

ops = set()
nonzero_found = False
for i, n in enumerate(m.graph.node):
    ops.add(n.op_type)
    if n.op_type == 'NonZero':
        nonzero_found = True
        print(f"Found NonZero: {n.name}")
        const_values = np.arange(1024, dtype=np.int64).reshape(1024, 1)
        const_tensor = numpy_helper.from_array(const_values, name=f'{n.name}_const')
        const_node = helper.make_node('Constant', inputs=[], outputs=n.output, value=const_tensor, name=f'{n.name}_const')
        m.graph.node.remove(n)
        m.graph.node.insert(i, const_node)

print(f"Ops: {sorted(ops)}")
print(f"NonZero found: {nonzero_found}")
print(f"Conv ops: {sum(1 for n in m.graph.node if n.op_type == 'Conv')}")

fixed_path = work_dir / "smolvlm_vision_v2b_fixed.onnx"
onnx.save(m, str(fixed_path))
onnx.checker.check_model(m)
print(f"Fixed ONNX: {fixed_path}")
print("ONNX check: PASSED")

# Step 7: Verify with ONNX Runtime
print("\n=== ONNX Runtime verification ===")
import onnxruntime as ort
sess = ort.InferenceSession(str(fixed_path))
onnx_out = sess.run(None, {"pixel_values": dummy.numpy()})

# Compare with PyTorch output
with torch.no_grad():
    torch_out = wrapper(dummy).numpy()

out_diff = np.abs(onnx_out[0] - torch_out).max()
out_cos = np.dot(onnx_out[0].flatten(), torch_out.flatten()) / (
    np.linalg.norm(onnx_out[0].flatten()) * np.linalg.norm(torch_out.flatten())
)
print(f"ONNX vs Torch max diff: {out_diff:.8f}")
print(f"ONNX vs Torch cosine: {out_cos:.8f}")

np.save(work_dir / "golden_output.npy", torch_out)
print(f"\nGolden output: {work_dir / 'golden_output.npy'}")

print("\n=== DONE - Ready for ACUITY ===")
print(f"ONNX: {fixed_path}")
