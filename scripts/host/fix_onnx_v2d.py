#!/usr/bin/env python3
"""Fix ONNX type error after NonZero removal: add Cast int64 for Gather indices."""
import onnx
from onnx import helper, TensorProto
import numpy as np
from pathlib import Path

work_dir = Path("work/generated/smolvlm_256m_v2b")
input_path = work_dir / "smolvlm_vision_v2b_fixed.onnx"
output_path = work_dir / "smolvlm_vision_v2b_final.onnx"

m = onnx.load(str(input_path))

# Fix Gather indices type: position_embedding/Gather needs int64 indices
for n in m.graph.node:
    if n.name == '/vision_model/embeddings/position_embedding/Gather':
        print(f"Found Gather: {n.name}")
        print(f"  inputs: {n.input}")
        indices_input = n.input[1]
        cast_name = indices_input.replace('/', '_') + '_cast_to_int64'
        cast_output = indices_input.replace('/', '_') + '_int64'
        cast_node = helper.make_node(
            'Cast', inputs=[indices_input], outputs=[cast_output],
            to=TensorProto.INT64, name=cast_name
        )
        new_inputs = [n.input[0], cast_output]
        n.ClearField('input')
        n.input.extend(new_inputs)
        for j, existing in enumerate(m.graph.node):
            if existing == n:
                m.graph.node.insert(j, cast_node)
                break
        print(f"  Added Cast, new inputs: {list(n.input)}")
        break

# Shape inference
try:
    m = onnx.shape_inference.infer_shapes(m)
    print("Shape inference: PASSED")
except Exception as e:
    print(f"Shape inference warning: {e}")

onnx.save(m, str(output_path))
print(f"Saved: {output_path}")

try:
    onnx.checker.check_model(m)
    print("ONNX check: PASSED")
except Exception as e:
    print(f"ONNX check: {e}")

try:
    import onnxruntime as ort
    import torch
    dummy = torch.randn(1, 3, 512, 512, dtype=torch.float32).numpy()
    sess = ort.InferenceSession(str(output_path))
    out = sess.run(None, {"pixel_values": dummy})
    print(f"ONNX Runtime: OK, output shape {out[0].shape}")
except Exception as e:
    print(f"ONNX Runtime failed: {e}")
