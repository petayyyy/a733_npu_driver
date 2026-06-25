#!/usr/bin/env python3
"""Fix ONNX: only replace NonZero, skip Cast fix (let ACUITY handle types)."""
import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from pathlib import Path

work_dir = Path("work/generated/smolvlm_256m_vision_encoder")
input_path = work_dir / "smolvlm_vision_encoder_opset15.onnx"
output_path = work_dir / "smolvlm_vision_encoder_nononzero_only.onnx"

m = onnx.load(str(input_path))

# Replace NonZero with Constant (ONLY this change)
nonzero_idx = None
nonzero_node = None
for i, n in enumerate(m.graph.node):
    if n.op_type == 'NonZero':
        nonzero_node = n
        nonzero_idx = i
        break

if nonzero_node:
    print(f"Replacing NonZero: {nonzero_node.name}")
    const_values = np.arange(1024, dtype=np.int64).reshape(1024, 1)
    const_tensor = numpy_helper.from_array(const_values, name=f'{nonzero_node.name}_const')
    const_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=nonzero_node.output,
        value=const_tensor,
        name=f'{nonzero_node.name}_const'
    )
    m.graph.node.remove(nonzero_node)
    m.graph.node.insert(nonzero_idx, const_node)
    print("  Replaced NonZero with Constant[int64, shape=(1024,1)]")

# Save
onnx.save(m, str(output_path))
print(f"\nSaved: {output_path}")

# Verify no NonZero
for n in m.graph.node:
    if n.op_type == 'NonZero':
        print("ERROR: NonZero still present!")
        break
else:
    print("Verified: No NonZero ops")

onnx.checker.check_model(m)
print("ONNX check: PASSED")
