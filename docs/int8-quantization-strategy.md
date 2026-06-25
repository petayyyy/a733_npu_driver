# INT8 Quantization Strategy — Final Assessment

**Status: CLOSED (2026-06-25).** INT8 is not a viable path for LLM inference
on the A733 NPU with ACUITY 6.30.22.

## What was tested

- **PCQ (per-channel int8 weights, int8 activations)**: exports and runs
  mechanically, but produces incoherent text. Activation quantization error
  at full model depth is too severe for coherent generation.
- **W8A16 (per-channel int8 weights, int16 activations)**: host quality is
  0.079 cosine on Qwen (far below the 0.90 threshold). SmoothQuant variants
  blocked by ACUITY quantize-table serialization bug (0-byte `.quantize` file).
- **Per-channel int16**: pegasus 6.30.22 rejects: only INT8 and INT4 qtypes
  are supported for `perchannel_symmetric_affine`. This is chip-gated, not a
  configuration issue.
- **Hybrid quantization (ACUITY w8a16/hybrid)**: reaches `End quantization`
  then hangs during YAML table serialization (0-byte `.quantize` file).
  Affects SmolLM2 and Qwen.

## Why INT8 failed

Activation quantization inside the transformer body dominates quality loss.
Weight-only quantization is less destructive, but the int8 quality cliff is too
steep for real-world LLM decoding at model scale.

The ACUITY hybrid YAML failure is a separate tooling bug: quantization analysis
completes correctly (`.quantize.json` is valid) but the YAML serializer
crashes/hangs when writing a table with hundreds of `dtype_converter` nodes.

## Current production path

**int16 dynamic fixed point** is the only working LLM quant on this toolchain.
It is ~1.45× slower than ideal int8 and produces NBG files ~2× larger, but it
generates coherent text for SmolLM2-135M/360M.

## For Qwen specifically

int16 DFP itself fails on Qwen (cosine 0.236) because Qwen's activation
outliers exceed the dynamic range of single-scale int16. The only format that
preserves Qwen quality is BF16 (cosine 0.991), which doesn't export. No int8
or int16 path closes this gap.

## Verdict

INT8 quantization is not useful for LLM inference on this toolchain.
If a vendor update provides:
- RF8-style per-chain int4 quantization, or
- Per-channel int16 support, or
- Fixed BF16 export

then a lightweight integer quant may become viable. Until then, use int16 for
NPU-LLM (SmolLM2 class only) and fall back to CPU for models that need BF16
precision (Qwen).
