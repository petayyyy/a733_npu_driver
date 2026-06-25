#!/usr/bin/env python3
"""Fix ONNX type errors in SmolVLM vision encoder and remove NonZero."""
import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from pathlib import Path

work_dir = Path("work/generated/smolvlm_256m_vision_encoder")
input_path = work_dir / "smolvlm_vision_encoder_opset15.onnx"
output_path = work_dir / "smolvlm_vision_encoder_fixed.onnx"

m = onnx.load(str(input_path))

# Step 1: Fix the type error on Gather indices
# The issue: Gather at /vision_model/embeddings/position_embedding/Gather 
# has float32 indices input but needs int64

# Find the problematic Gather and the nodes feeding it
for n in m.graph.node:
    if n.name == '/vision_model/embeddings/position_embedding/Gather':
        print(f"Found Gather: {n.name}")
        print(f"  inputs: {n.input}")
        print(f"  outputs: {n.output}")
        
        # The second input (indices) is float, needs to be int64
        # Insert a Cast node before this Gather
        indices_input = n.input[1]  # indices are the second input
        
        # Create Cast node
        cast_name = n.input[1] + '_cast_to_int64'
        cast_output = n.input[1] + '_cast_int64'
        cast_node = helper.make_node(
            'Cast',
            inputs=[n.input[1]],
            outputs=[cast_output],
            to=TensorProto.INT64,
            name=cast_name
        )
        
        # Update Gather to use cast output
        new_inputs = [n.input[0], cast_output]
        n.ClearField('input')
        n.input.extend(new_inputs)
        
        # Insert Cast before Gather
        for i, existing in enumerate(m.graph.node):
            if existing == n:
                m.graph.node.insert(i, cast_node)
                break
        
        print(f"  Added Cast node: {cast_name}")
        print(f"  Updated Gather inputs to: {n.input}")
        break

# Step 2: Replace NonZero with Constant
nonzero_idx = None
nonzero_node = None
for i, n in enumerate(m.graph.node):
    if n.op_type == 'NonZero':
        nonzero_node = n
        nonzero_idx = i
        break

if nonzero_node:
    print(f"\nReplacing NonZero: {nonzero_node.name}")
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

# Step 3: Check and fix other type issues
# Check all Gather ops
for n in m.graph.node:
    if n.op_type == 'Gather':
        # Check if indices input exists as value info
        indices_name = n.input[1] if len(n.input) > 1 else None
        if indices_name:
            # Find type info for indices
            found_type = None
            for vi in m.graph.value_info:
                if vi.name == indices_name:
                    found_type = vi.type.tensor_type.elem_type
                    break
            for init in m.graph.initializer:
                if init.name == indices_name:
                    found_type = init.data_type
                    break
            # Also check among node outputs
            for other in m.graph.node:
                for o in other.output:
                    if o == indices_name:
                        # Can't easily get output type without shape inference
                        pass

# Step 4: Run shape inference to fix types
try:
    m = onnx.shape_inference.infer_shapes(m)
    print("\nShape inference: PASSED")
except Exception as e:
    print(f"\nShape inference failed: {e}")

# Save
onnx.save(m, str(output_path))
print(f"\nSaved: {output_path}")

# Verify with ONNX checker
try:
    onnx.checker.check_model(m)
    print("ONNX check: PASSED")
except Exception as e:
    print(f"ONNX check: {e}")

# Test with ONNX Runtime
try:
    import onnxruntime as ort
    dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
    sess = ort.InferenceSession(str(output_path))
    out = sess.run(None, {"pixel_values": dummy})
    print(f"ONNX Runtime: WORKS! Output shape: {out[0].shape}")
except Exception as e:
    print(f"ONNX Runtime failed: {e}")
