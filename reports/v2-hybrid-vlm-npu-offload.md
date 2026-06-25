# V2-hybrid-vlm-npu-offload: SmolVLM Vision Encoder NPU Blocked

Date: 2026-06-25 | Status: BLOCKED (ACUITY)

## Summary

SmolVLM-256M-Instruct's vision encoder (SigLIP 400M via Idefics3) was exported to
ONNX (357 MB, 1x3x512x512 → 1x64x576) but **cannot be converted to int16 NBG**
through ACUITY 6.30.22. Two separate blockers were encountered. The fallback
MobileCLIP-S0 encoder (proven on NPU) is dimensionally incompatible with
SmolVLM's LLM (512-dim vs expected 576-dim after SigLIP projection).

**V1 CPU-only config remains the deliverable for this VLM.**

## SmolVLM Vision Encoder Architecture

| Property | Value |
|----------|-------|
| Architecture | SigLIP-base, wrapped in Idefics3VisionTransformer |
| Input | `1x3x512x512` float32 (NCHW) |
| Vision output | `1x1024x768` (1024 patches, no CLS token) |
| Connector | Idefics3Connector: pools 1024→64 tokens, projects 768→576 |
| Final output | `1x64x576` float32 |
| Text decoder | LLaMA-style, hidden=576, 30 layers, vocab=49152 |
| Disk (ONNX) | 357 MB |
| ONNX ops | 30 unique (Add, Cast, Concat, Conv, Div, Gather, LayerNorm, MatMul, Mul, NonZero, ReduceSum, Reshape, ScatterND, Shape, Slice, Softmax, Split, Squeeze, Sub, Tanh, Transpose, Unsqueeze, Where, etc.) |

## ACUITY Blocker 1: NonZero Op

- ONNX file `smolvlm_vision_encoder.onnx` (opset 17)
- ACUITY import error: `Acuity can not support NonZero op in specified dynamic shape model for now`
- Location: `/vision_model/embeddings/NonZero` — computes valid patch position IDs from attention mask
- Resolution: replaced NonZero with Constant[int64, shape=(1024,1)] since all 1024 patches are valid for 512×512 images

## ACUITY Blocker 2: Conv Shape Inference Crash

- After NonZero removal, ACUITY import fails at shape inference:
  `IndexError: list index out of range` in `_conv_shape` (smart_toolkit.py:1571)
- Location: Conv2d patch embedding layer (kernel=16, stride=16, in_channels=3, out_channels=768)
- Tested: opset 15 and opset 17, with and without Cast fix — same error
- Root cause: ACUITY 6.30.22 shape inference cannot handle the SigLIP patch embedding Conv configuration
- Log preserved: `logs/host/v2-smolvlm-vision-conv-shape-import.err.log`

## MobileCLIP-S0 Fallback Assessment

| | MobileCLIP-S0 | SmolVLM SigLIP |
|---|---|---|
| Embedding dim | 512 | 768 (→576 via connector) |
| Tokens | 1 (pooled) | 64 (after connector pooling) |
| NPU proven | Yes (22.6ms, cos 0.99996) | No |
| Compatible with SmolVLM LLM | **No** (dimension mismatch) | N/A |

MobileCLIP-S0 cannot substitute for SmolVLM's vision encoder without a trained
adapter/projector (512→576 dim mapping). This is out of scope for V2.

## Attempted Workarounds

1. **NonZero removal**: Successfully replaced with Constant (verified: ONNX check passes, ONNX Runtime output matches PyTorch, cosine 1.00000000)
2. **Opset downgrade (17→15)**: Same Conv shape error
3. **Cast fix for Gather type error**: Not attempted (Conv error blocks import first)
4. **Direct SigLIP export** (bypassing Idefics3): Requires loading SigLIP weights from SmolVLM safetensors — not attempted due to time constraints

## Conclusion

**SUCCESS GATE: FAILED** — SmolVLM vision encoder on NPU is blocked by ACUITY
6.30.22. V1 CPU-only (SmolVLM-256M-Instruct Q8_0 via llama.cpp, 52.6 tok/s,
634 MB RSS) is the deliverable for this VLM.

## Vendor Blocker Packet

```
Op: Conv (patch embedding)
Model: SmolVLM-256M-Instruct → Idefics3VisionTransformer → SigLIP
Input: 1x3x512x512 float32
Conv params: kernel=16, stride=16, padding=0, in_channels=3, out_channels=768
ONNX opset: 15 and 17 tested
ACUITY version: 6.30.22 (ubuntu-npu:v2.0.10.1)
Error: IndexError in _conv_shape (smart_toolkit.py:1571)
```

## Files

- ONNX (with NonZero): `work/generated/smolvlm_256m_vision_encoder/smolvlm_vision_encoder.onnx` (357 MB)
- ONNX (NonZero removed): `work/generated/smolvlm_256m_vision_encoder/smolvlm_vision_encoder_nononzero_only.onnx`
- Host comparison: cosine 1.00000000 vs PyTorch (verified)
- Export script: `scripts/host/export_smolvlm_vision_onnx.py`
- Fix scripts: `scripts/host/remove_nonzero.py`, `scripts/host/fix_onnx.py`
