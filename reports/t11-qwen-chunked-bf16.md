# T11 Qwen Chunked BF16

Date: 2026-06-24

Task: test whether Qwen2.5-0.5B W=32 BF16 NBG export is blocked by the
151936-wide vocabulary dimension, and if needed try ACUITY native hybrid with
lm_head/outlier layers promoted to BF16.

## Summary

Verified: the ONNX graph can be generated with a chunked lm_head and no final
151936-wide output tensor. The no-concat graph exposes three outputs:

- `logits_chunk0`: 50646 values
- `logits_chunk1`: 50646 values
- `logits_chunk2`: 50644 values

Verified: ONNX Runtime for the no-concat chunked graph matches the FP oracle:
logits cosine `0.9999999999967962`, top-1 match `198`, max abs diff
`0.000029087`.

Blocked: ACUITY BF16 export still fails. Chunking the output alone does not
clear `vnn_VerifyGraph -3 / 64768`; chunking the token embedding table too
moves the failure to a BF16 `DATACONVERT` setup failure (`65280`).

Blocked: ACUITY hybrid with selected BF16 lm_head/outlier layers reaches host
inference, but export fails at an illegal BF16-to-DFP-INT16 `PERMUTE`
boundary. Host-only quality also fails the gate: logits cosine
`0.2540464987`, top-1 `266` vs oracle `198`.

Result: no T11 Qwen NBG candidate passed export plus host quality. The Orange
Pi was not used.

## Changes

Verified code changes:

- `scripts/host/make_real_llm_onnx.py` adds `--lm-head-chunk-size`,
  `--lm-head-output-mode concat|chunks`, and optional token embedding chunking.
- `scripts/host/compare_onnxruntime_to_oracle.py` concatenates chunked logits
  before comparing to the FP oracle.
- `scripts/host/compare_acuity_host_to_oracle.py` supports chunked host outputs
  and optional `nbg_meta.json` output-name mapping.
- `scripts/host/package_acuity_nbg.py` packages ACUITY host outputs and NBG
  artifacts for board runs.
- `scripts/host/make_qwen2_chunked_hybrid_seed.py` creates ACUITY hybrid seed
  quantize tables from int16 and BF16 quantize tables plus entropy rankings.

Verified syntax check:

```text
python -m py_compile scripts\host\make_real_llm_onnx.py scripts\host\compare_onnxruntime_to_oracle.py scripts\host\compare_acuity_host_to_oracle.py scripts\host\package_acuity_nbg.py scripts\host\make_qwen2_chunked_hybrid_seed.py
```

## Gate 1

Verified chunked ONNX artifacts:

- `work/generated/qwen25_05b_w32_chunked_bf16/real_llm.onnx`
- `work/generated/qwen25_05b_w32_chunked_bf16_no_concat/real_llm.onnx`
- `work/generated/qwen25_05b_w32_chunked_bf16_embed_chunks/real_llm.onnx`

Verified no-concat ORT vs FP:

- Log: `logs/host/t11-qwen25-w32-chunked-no-concat-onnxruntime-vs-fp.log`
- JSON: `logs/host/t11-qwen25-w32-chunked-no-concat-onnxruntime-vs-fp.json`
- Metrics: logits cosine `0.9999999999967962`, top-1 match `198`, max abs diff
  `2.9087066650390625e-05`.

Verified ACUITY BF16 export failures:

- Chunked lm_head with final Concat:
  `logs/host/t11-qwen25-w32-chunked-bf16-convert.retry1.err.log`
  failed at `vnn_VerifyGraph` status `-3`, `Fatal model generation error:
  64768`.
- Chunked lm_head with chunk outputs and no final Concat:
  `logs/host/t11-qwen25-w32-chunked-no-concat-bf16-convert.err.log`
  failed at `vnn_VerifyGraph` status `-3`, `Fatal model generation error:
  64768`.
- Chunked token embedding table plus chunked lm_head outputs:
  `logs/host/t11-qwen25-w32-chunked-embed-chunks-bf16-convert.err.log`
  failed at `Check node[13] DATACONVERT fail`, `Fatal model generation error:
  65280`.

Conclusion: the original full-width logits tensor is not the only BF16 export
blocker.

## Gate 2

Verified hybrid seed:

- Seed summary:
  `work/generated/qwen25_05b_w32_chunked_hybrid_seed/qwen25_05b_w32_chunked_bf16_no_concat_int16.json`
- Final hybrid table copy:
  `work/generated/qwen25_05b_w32_chunked_hybrid_seed/qwen25_05b_w32_chunked_bf16_no_concat_int16.hybrid.quantize`
- Selected BF16 layers: `fullconnect_1979`, `fullconnect_1982`,
  `fullconnect_1985`, `fullconnect_2084`, `fullconnect_2102`,
  `fullconnect_2168`, `fullconnect_2213`, `fullconnect_2318`,
  `fullconnect_2342`, `fullconnect_2483`, `fullconnect_2489`.
- Quantize log:
  `logs/host/t11-qwen25-w32-chunked-hybrid-int16-bf16top8-quantize.retry2.err.log`
  completed `Error(0),Warning(60)` and wrote a `470740` byte quantize table.

Verified hybrid inference/export:

- Log:
  `logs/host/t11-qwen25-w32-chunked-hybrid-int16-bf16top8-infer-export.err.log`
- Host inference completed `Error(0),Warning(11)`.
- Host-only compare:
  `logs/host/t11-qwen25-w32-chunked-hybrid-int16-bf16top8-host-vs-fp.json`
  reports logits cosine `0.25404649873079016`, top-1 mismatch `266` vs `198`.
- Export warned that `bfloat16-bfloat16` hybrid could not be applied through the
  documented hybrid path, then failed at:

```text
Inputs/Outputs data type not support:  BFLOAT16, DFP INT16
Check node[24] PERMUTE fail
Fatal model generation error: 65280
```

Conclusion: ACUITY can preserve BF16 qparams in the seed/table, but the export
graph still creates an unsupported BF16/DFP-INT16 boundary.

## Orange Pi

Verified not run: no package passed both host export and quality gates, so no
T11 package was uploaded to `192.168.31.225` and no board power-cycle/reset was
requested.

## Vendor Blocker

Verified blocker set for a vendor ticket:

- Host-coherent chunked ONNX proof:
  `logs/host/t11-qwen25-w32-chunked-no-concat-onnxruntime-vs-fp.json`
- BF16 no-concat export blocker:
  `logs/host/t11-qwen25-w32-chunked-no-concat-bf16-convert.err.log`
- Strict token+lm_head chunk BF16 DataConvert blocker:
  `logs/host/t11-qwen25-w32-chunked-embed-chunks-bf16-convert.err.log`
- BF16-hybrid PERMUTE dtype blocker:
  `logs/host/t11-qwen25-w32-chunked-hybrid-int16-bf16top8-infer-export.err.log`

Assumption: next productive path is vendor guidance or an architecture-level
split that avoids BF16/DFP boundaries inside ACUITY-generated PERMUTE/SLICE
nodes; plain int16, full FP16, and unchunked full BF16 remain confirmed dead
ends from T8/T9/T11.
