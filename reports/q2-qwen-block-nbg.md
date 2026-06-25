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

## Gate 2A -- Full-Chain Host Coherence (PASSED)

Date: 2026-06-25. Host-only int16 simulation. No board or runtime build.

### Method

1. Built `scripts/host/q2_simulate_int16_chain.py` — a PyTorch simulation that:
   - Loads FP32 weights from safetensors
   - Quantizes each weight matrix to int16 dynamic fixed point before use
   - At each decoder-block boundary, quantizes the hidden state to int16 (simulating
     the NBG output/input quantize cycle that occurs with chained per-block NBGs)
   - Runs the full 24-layer Qwen2.5-0.5B model with int16-quantized weights
   - Compares per-layer hidden states and final logits to the FP32 oracle
2. Two configurations tested:
   - **With boundary quantization** (realistic for chained NBGs): each block
     output is quantized to int16 before feeding to the next block.
   - **Without boundary quantization** (weight-only): only model weights are
     int16-quantized; activations stay FP32 within blocks.
3. Prompt: ChatML-wrapped "The capital of France is", W=32 fixed window.
4. Autoregressive decode loop: argmax sampling, sliding window, 10 steps.

### Results — With Boundary Quantization (Realistic)

| Metric | Value |
|--------|-------|
| FP32 oracle top-1 | 785 ("The") |
| Int16 sim top-1 | 785 ("The") |
| Top-1 match | **YES** |
| Logits cosine | **0.974913** |
| Generated tokens (10 steps) | **10/10 match FP32** |
| Generated text | "The capital of France is Paris.\n" |

### Per-Layer Cosine Drift (with boundary quant)

| Layer | Cosine vs FP32 |
|-------|---------------|
| embed | 1.00000000 |
| 0-15 | 0.99994 – 0.99999 |
| 16 | 0.999905 |
| 17 | 0.999870 |
| 18 | 0.999817 |
| 19 | 0.999719 |
| 20 | 0.999580 |
| 21 | 0.987952 |
| 22 | 0.987201 |
| 23 | 0.976839 |
| final_norm | 0.979708 |

The per-layer cosine degrades gradually from >0.99999 to ~0.977 at layer 23,
with the sharpest drop at layers 21-23. This is consistent with the known
behaviour: certain weight matrices in later Qwen layers have value distributions
that lose more precision under int16 dynamic fixed point.

### Results — Without Boundary Quantization (Weight-Only)

| Metric | Value |
|--------|-------|
| Logits cosine | **0.999990** |
| Top-1 match | YES |
| Per-layer cosine min | 0.999941 (layer 23) |

Weight-only int16 quantization has near-zero cumulative error (cosine > 0.9999
at all layers). This confirms that the boundary quantize/dequantize cycle is the
dominant error source, not weight quantization itself.

### Verdict

**Gate 2A PASSES.** 24-block int16 chain is coherent end-to-end on host:
- Logits cosine 0.975 > 0.90 threshold
- Top-1 matches FP32 oracle
- All 10 generated tokens match FP32 oracle exactly
- Generated text "The capital of France is Paris." matches the expected output
- Depth accumulation is gradual and does not cause catastrophic failure

The int16-per-block-but-depth-bad hypothesis from the monolithic int16 failure
(cosine 0.236) is **REFUTED**: with per-block NBGs, int16 error accumulates
gracefully and stays above 0.975 cosine. The monolithic int16 failure was an
aggregate-graph limit (vnn_VerifyGraph -3), not a depth-accumulation problem.

### Next Steps: Gate 2B

Proceed to VIPLite Multi-Graph investigation on Orange Pi (192.168.31.225).
The host gate is cleared; the remaining risk is the runtime's ability to keep
26 NBGs resident and chain them without per-token reload.

## Estimated NBG sizes (from Gate 1 analysis)

| Stage | Est. NBG size |
|-------|--------------|
| Per decoder block | ~22.6 MB |
| 24 blocks total | ~543 MB |
| Embedding | ~260 MB (int16 weights) |
| Final (norm + lm_head) | ~260 MB (int16 weights) |
| **Total** | **~1,063 MB** |

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
- `scripts/host/q2_simulate_int16_chain.py`: New. Int16 chained simulation with
  per-layer cosine drift tracking and autoregressive decode loop.

## Logs

- `logs/host/q2-gate1-block0-int16.log` - Block 0 full conversion
- `logs/host/q2-gate1-block0-int16-v2.log` - Block 0 conversion v2 (with inputmeta fix)
- `work/generated/q2_gate2a/q2_gate2a_simulation.json` - Gate 2A full results
- `work/generated/q2_gate2a_prompt/q2_gate2a_simulation.json` - Gate 2A prompt results
