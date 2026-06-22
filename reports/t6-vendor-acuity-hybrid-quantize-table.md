# T6 Vendor Blocker: ACUITY Hybrid Quantize Table Dump

Date: 2026-06-23

## Summary

ACUITY hybrid quantization for the full SmolLM2-135M W=32 graph consistently
reaches `End quantization...`, starts dumping the rewritten YAML quantize table,
truncates the target `.quantize` file to `0` bytes, and then remains CPU-active
without producing an inference/exportable package.

This blocks the T5 recovery paths that require `w8a16` or hybrid quantization.
The same graph imports, quantizes, exports, and runs coherently in int16.

## Environment

```text
Docker image: ubuntu-npu:v2.0.10.1
ACUITY path: /root/acuity-toolkit-whl-6.30.22/bin
Target: VIP9000NANODI_PLUS_PID0X1000003B
Model: SmolLM2-135M-Instruct fixed W=32 decoder graph
ONNX: work/generated/smollm2_135m_w32/real_llm.onnx
Dataset: work/generated/smollm2_135m_w32_calib/dataset.txt
Input: token_ids, int32, 1x32
Output: logits, 1x1x49152
```

## Repro 1: Calibrated PCQ Seed + Hybrid

Command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_hybrid_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --hybrid \
  --hybrid-seed-quantize work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_calib/smollm2_135m_w32_calib_pcq.quantize
```

Observed:

```text
ACUITY loads existing quantization tensor table.
ACUITY emits smollm2_135m_w32_hybrid_pcq_pcq.quantize.json.
The JSON graph contains 589 dtype_converter ops.
ACUITY logs End quantization...
ACUITY logs Dump net quantize tensor table to .../smollm2_135m_w32_hybrid_pcq_pcq.quantize.
The YAML .quantize file becomes 0 bytes and stays 0 bytes while ACUITY remains CPU-active.
No inference/export package is produced.
```

Logs:

```text
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.log
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.err.log
```

Artifacts:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_hybrid_pcq/smollm2_135m_w32_hybrid_pcq_pcq.quantize
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_hybrid_pcq/smollm2_135m_w32_hybrid_pcq_pcq.quantize.json
```

## Repro 2: Mixed PCQ Seed + Hybrid

Command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_mixed_hybrid_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --hybrid \
  --hybrid-seed-quantize work/generated/smollm2_135m_w32_mixed_pcq/smollm2_135m_w32_mixed_pcq_pcq.quantize
```

Observed:

```text
ACUITY loads existing mixed quantization tensor table.
ACUITY emits smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.json.
The JSON graph contains 587 dtype_converter ops.
ACUITY logs End quantization...
ACUITY logs Dump net quantize tensor table to .../smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.
The YAML .quantize file becomes 0 bytes and remains 0 bytes.
After an extra 90 seconds, Docker was still CPU-active at about 99.6 percent and 2.6 GiB RSS.
No inference/export package is produced.
```

Logs:

```text
logs/host/t5-smollm2-w32-mixed-hybrid-pcq-convert.log
logs/host/t5-smollm2-w32-mixed-hybrid-pcq-convert.err.log
```

Artifacts:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_mixed_hybrid_pcq/smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_mixed_hybrid_pcq/smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.json
```

## Expected Behavior

ACUITY should either:

- write a valid non-empty YAML `.quantize` table and proceed to inference/export, or
- fail with a non-zero exit code and a diagnostic explaining the unsupported hybrid graph/table state.

## Actual Behavior

ACUITY reports quantization completion, truncates the target YAML `.quantize`
file to zero bytes, and stays CPU-active indefinitely. The process does not
return a diagnostic and does not produce an NBG package.

## Control Results

The same SmolLM2 W=32 graph is valid on this toolchain in other modes:

```text
int16 export: succeeds, NBG 280,882,632 bytes
int16 board run: coherent, first six generated tokens match FP oracle
pcq export: succeeds, NBG about 153,984,304 bytes
pcq board run: executes mechanically but generates incoherent tokens
mixed PCQ seed export: succeeds, NBG 205,233,968 bytes, output int16 fl=10
mixed PCQ board run: executes mechanically but generates incoherent tokens
```

This points to a hybrid quantize-table serialization or rewrite bug rather than
a basic ONNX import, VIPLite export, or board-runtime issue.
