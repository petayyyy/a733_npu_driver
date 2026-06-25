#!/usr/bin/env python3
"""Export SmolVLM vision encoder + connector to ONNX."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import numpy as np
from pathlib import Path
import json

from transformers import AutoModel, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
work_dir = Path("work/generated/smolvlm_256m_vision_encoder")
work_dir.mkdir(parents=True, exist_ok=True)

print("=== Loading SmolVLM model ===")
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

vision_model = model.vision_model
connector = model.connector

print(f"Vision model: {type(vision_model).__name__}")
print(f"Connector: {type(connector).__name__}")

# The vision_model takes pixel_values as input
# pixel_values shape: [batch, channels, height, width] = [1, 3, 512, 512]
# Output: [batch, num_patches, hidden_size] = [1, 1025, 768]

# The connector projects to LLM hidden size: [1, 1025, 576]

# Step 1: Trace the vision encoder
print("\n=== Tracing vision encoder ===")
batch_size = 1
channels = 3
height = 512
width = 512

dummy_input = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

# Test forward pass
with torch.no_grad():
    vision_output = vision_model(dummy_input)
    # vision_output is BaseModelOutput with .last_hidden_state
    vision_embeds = vision_output.last_hidden_state
    print(f"Vision output shape: {vision_embeds.shape}")  # Should be (1, 1025, 768)
    print(f"Vision output dtype: {vision_embeds.dtype}")
    
    connector_output = connector(vision_embeds)
    print(f"Connector output shape: {connector_output.shape}")  # Should be (1, 1025, 576)
    print(f"Connector output dtype: {connector_output.dtype}")

# Export vision encoder
vision_onnx = work_dir / "smolvlm_vision_encoder.onnx"
print(f"\n=== Exporting vision encoder to ONNX ===")

class VisionWrapper(torch.nn.Module):
    def __init__(self, vision_model, connector):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
    
    def forward(self, pixel_values):
        x = self.vision_model(pixel_values).last_hidden_state
        x = self.connector(x)
        return x

wrapper = VisionWrapper(vision_model, connector)
wrapper.eval()

torch.onnx.export(
    wrapper,
    dummy_input,
    str(vision_onnx),
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_embeds": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"ONNX exported: {vision_onnx}")
print(f"Size: {vision_onnx.stat().st_size / (1024**2):.1f} MB")

# Verify ONNX
import onnx
onnx_model = onnx.load(str(vision_onnx))
onnx.checker.check_model(onnx_model)
print("ONNX check: PASSED")

# Save the output for comparison
with torch.no_grad():
    golden = wrapper(dummy_input).numpy()
np.save(work_dir / "vision_encoder_golden.npy", golden)
print(f"Golden output saved: shape {golden.shape}, dtype {golden.dtype}")

# Also save metadata
meta = {
    "model": model_id,
    "input_shape": [batch_size, channels, height, width],
    "input_dtype": "float32",
    "output_shape": list(golden.shape),
    "output_dtype": str(golden.dtype),
    "vision_config": {
        "hidden_size": 768,
        "num_layers": 12,
        "patch_size": 16,
        "image_size": 512,
    },
    "connector_output_dim": 576,
}
with open(work_dir / "vision_encoder_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n=== Done ===")
print(f"ONNX: {vision_onnx}")
print(f"Meta: {work_dir / 'vision_encoder_meta.json'}")
print(f"Golden output: {work_dir / 'vision_encoder_golden.npy'}")
