# Q2 Qwen Block NBG

Date: 2026-06-25. Host ACUITY experiments for per-decoder-block Qwen2.5-0.5B NBG compilation.

## Gate 1 -- Single Block (PASSED)

### Method

1. Extended `make_real_llm_onnx.py` with `--export-block N` to export a single
   Qwen2.5 decoder block as a standalone ONNX graph.
   - Input: `hidden_in` (1x32x896 float32)
   - Output: layer output (1x32x896 float32)
   - All weights, RoPE tables, causal mask, shape constants baked as initializers.
   - ONNX size: ~57 MB initializers per block (~14M params in FP32).
2. Imported block 0 ONNX into ACUITY with `--input-size-list 32,896` (comma-separated
   for 3D tensor support).
3. Quantized with int16 dynamic_fixed_point.
4. Compiled to NBG via `pegasus_export_ovx_nbg.sh`.
5. Validated host quality: ACUITY int16 host output vs ONNX Runtime FP32 output.

### Results

| Metric | Value |
|--------|-------|
| NBG export status | Error(0), Warning(0) |
| Simulator create network | 135 ms |
| Simulator verify graph | 8,944 ms |
| Simulator one run | 199.7 ms |
| NBG file size | 23,718,496 bytes (22.6 MB) |
| Host cosine (int16 vs FP32) | **0.999965** |
| Host max abs diff | 0.0144 |
| Host mean abs diff | 0.00107 |

### Verdict

**Gate 1 PASSES**. A single Qwen2.5 decoder block exports to NBG with near-perfect
host quality (cosine > 0.999). The aggregate-graph limit (vnn_VerifyGraph -3) is
confirmed to be an aggregate issue, not a per-op or per-block issue. Splitting into
per-block NBGs sidesteps the limit.

## Gate 2 -- Chain All 24 Blocks (IN PROGRESS)

### ONNX Generation

All 26 stage ONNX files generated:
- 24 decoder blocks (layer 0-23): ~57 MB initializers each
- 1 embedding stage (`--export-embedding`): token_ids -> hidden0, ~519 MB initializers
  (chunked token embedding, 3 chunks of 50646 vocab)
- 1 final stage (`--export-final`): hidden24 -> logits, ~519 MB initializers
  (final RMSNorm + chunked lm_head, 3 chunks)

### Estimated NBG sizes

| Stage | Est. NBG size |
|-------|--------------|
| Per decoder block | ~22.6 MB |
| 24 blocks total | ~543 MB |
| Embedding | ~260 MB (int16 weights) |
| Final (norm + lm_head) | ~260 MB (int16 weights) |
| **Total** | **~1,063 MB** |

### Compilation status

- Block 0: Compiled, NBG 22.6 MB, host cosine 0.999965
- Block 1: Compiling
- Blocks 2-23: ONNX ready, pending compilation
- Embedding: ONNX ready, pending compilation
- Final: ONNX ready, pending compilation

### Next steps for Gate 2

1. Complete compilation of all 26 NBGs.
2. Implement a chained runner:
   - Sequential vpm_run per NBG with host-side tensor memcpy between stages.
   - Fixed-window decode loop: embedding -> blocks 0-23 -> final -> argmax -> slide window.
   - CPU limited to tokenization, tensor passing, sampling, detokenization.
3. Validate full-model coherence on host (ACUITY simulator cross-check).
4. Port to Orange Pi Zero 3W (192.168.31.225) and measure tok/s, RSS, prefill.

### Risk assessment

- **NBG load overhead**: Each NBG is 22.6-260 MB. Loading 26 NBGs sequentially
  per token would be ~26 * load_time per token. With measured load time of
  ~1.35 ms/MB, the embedding NBG alone would take ~350 ms to load. This is fatal
  for decode speed unless NBGs can be loaded once and reused (Multi-Graph).
- **Orange Pi RAM**: 5.7 GB total, ~4-5 GB available. The 1 GB total NBG size
  should fit, but all 26 NBGs must be loaded simultaneously for efficient chaining.
  The T1 persistent runner pattern (load-once, reuse) is mandatory.
- **VIPLite Multi-Graph**: The SDK supports multi-graph execution which could
  chain blocks without intermediate host memcpy. This needs investigation in the
  C runner.

## Changes

- `scripts/host/make_real_llm_onnx.py`: Added `--export-block N`,
  `--export-embedding`, `--export-final` modes for per-stage ONNX export.
- `scripts/host/convert_onnx_to_nbg.sh`: Added `perchannel_int16` quant type
  (from Q1, kept for reference).

## Logs

- `logs/host/q2-gate1-block0-int16.log` - Block 0 full conversion
- `logs/host/q2-gate1-block0-int16-v2.log` - Block 0 conversion v2 (with inputmeta fix)
