#!/usr/bin/env python3
"""Remove NonZero op from SmolVLM vision encoder ONNX and replace with Constant."""
import onnx
from onnx import helper, numpy_helper
import numpy as np
from pathlib import Path

work_dir = Path("work/generated/smolvlm_256m_vision_encoder")
input_path = work_dir / "smolvlm_vision_encoder_opset15.onnx"
output_path = work_dir / "smolvlm_vision_encoder_nononzero.onnx"

m = onnx.load(str(input_path))

# Find the NonZero node and understand its context
nonzero_node = None
nonzero_idx = None
for i, n in enumerate(m.graph.node):
    if n.op_type == 'NonZero':
        nonzero_node = n
        nonzero_idx = i
        break

if nonzero_node is None:
    print("No NonZero found!")
    exit()

print(f"NonZero node: {nonzero_node.name}")
print(f"  inputs: {nonzero_node.input}")
print(f"  outputs: {nonzero_node.output}")

# The NonZero takes a flattened patch_attention_mask 
# For a fixed 512x512 image with no padding, all 1024 patches are valid
# The output is position IDs: [0..1023] reshaped to [1024, 1]
# This is used by Gather to index into position embeddings

# Find what consumes the NonZero output
consumers = []
for n in m.graph.node:
    for inp in n.input:
        if inp == nonzero_node.output[0]:
            consumers.append(n)
            print(f"\nConsumer: {n.op_type} '{n.name}'")
            print(f"  inputs: {n.input}")
            print(f"  outputs: {n.output}")

# The consumer is likely a Gather that uses position IDs to index position embeddings
# For Gather, indices must be int32 or int64
# Create a constant tensor with values [0, 1, 2, ..., 1023] shape [1024, 1]

const_values = np.arange(1024, dtype=np.int64).reshape(1024, 1)
const_tensor = numpy_helper.from_array(const_values, name=f'{nonzero_node.name}_const')

# Create replacement Constant node
const_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=nonzero_node.output,
    value=const_tensor,
    name=f'{nonzero_node.name}_const'
)

# Replace NonZero with Constant
m.graph.node.remove(nonzero_node)
m.graph.node.insert(nonzero_idx, const_node)

# Also need to add the constant tensor to initializers
# Actually, the Constant node with a value attribute doesn't need initializer

# Check the graph
print(f"\nChecking graph...")
try:
    onnx.checker.check_model(m)
    print("ONNX check: PASSED")
except Exception as e:
    print(f"ONNX check warning: {e}")

# Save
onnx.save(m, str(output_path))
print(f"\nSaved: {output_path}")
print(f"Size: {output_path.stat().st_size / (1024**2):.1f} MB")

# Verify: check no NonZero ops remain
m2 = onnx.load(str(output_path))
for n in m2.graph.node:
    if n.op_type == 'NonZero':
        print("ERROR: NonZero still present!")
        break
else:
    print("Verified: No NonZero ops in output")

# Verify output matches
import onnxruntime as ort
import torch

# Load original model for reference
from transformers import AutoModel
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)

# Original output
with torch.no_grad():
    v_out = model.vision_model(dummy_input)
    c_out = model.connector(v_out.last_hidden_state)
    torch_output = c_out.numpy()

# Modified ONNX output
sess = ort.InferenceSession(str(output_path))
onnx_output = sess.run(None, {"pixel_values": dummy_input.numpy()})

diff = np.abs(onnx_output[0] - torch_output).max()
print(f"\nModified ONNX vs original torch max diff: {diff:.8f}")
if diff < 1e-4:
    print("SUCCESS: Output matches!")
else:
    print(f"WARNING: Output mismatch (diff={diff})")
    # The NonZero replacement might be wrong
    # Let me check more carefully what the NonZero does
