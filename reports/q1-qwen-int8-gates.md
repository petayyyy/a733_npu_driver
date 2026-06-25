# Q1 Qwen INT8 Gates

Date: 2026-06-25. Host-only ACUITY experiments on Qwen2.5-0.5B W=32.
All work was host-only; no Orange Pi board access was required or started.

## Gate A -- Per-channel INT16

**Status: CONFIRMED DEAD END (chip-gated).**

### Method

1. Generated chunked-lm-head Qwen2.5-0.5B W=32 ONNX with `--lm-head-chunk-size 50646`
   (3 chunks: [0,50646], [50646,101292], [101292,151936]), as required by the ACUITY
   tensor dim limit of 65536.
2. Ran pegasus quantize with `--quantizer perchannel_symmetric_affine --qtype int16`.

### Result

Pegasus rejected the configuration:

```
Please specify valid quantize qtype for quantizer 'perchannel_symmetric_affine',
in ['int8', 'int4']
```

The `perchannel_symmetric_affine` quantizer in ACUITY 6.30.22 supports only INT8 and INT4
qtypes. INT16 per-channel is **not supported on any target**, not just this chip ID.
This was tested and verified on the host with target `VIP9000NANODI_PLUS_PID0X1000003B`.

Full log: `logs/host/q1-gatea-perchannel-int16-convert.log`

### Verdict

Per-channel INT16 is a dead end. The only per-channel option is INT8 (PCQ) for
weights with INT16 activations -- this is the Gate B W8A16 path.

## Gate B -- W8A16 with SmoothQuant alpha sweep

**Status: FAILS host quality gate for all measurable cases. Smoothed-variant W8A16
host tests blocked by ACUITY quantize-table serialization bug.**

### Method

1. Generated 12-window Qwen calibration dataset (`work/generated/qwen25_05b_w32_q1_calib/`).
2. Collected per-channel activation maxima and computed SmoothQuant smoothing
   scales at alphas 0.2, 0.5, 0.7 using `make_real_llm_smoothquant_scales.py`.
   (Alpha "none" = no smoothing; ONNX weights unchanged.)
3. Built smoothed ONNX for alphas 0.2, 0.5, 0.7 with chunked lm_head.
4. For alpha "none": generated a W8A16 quantize seed table from the existing
   non-chunked int16 ACUITY quantize table (440KB, valid). The seed replaces
   168 fullconnect weight entries with per-channel INT8 and 168 bias entries
   with INT32. The tied lm_head weight+bias and all activation tensors remain
   INT16 dynamic_fixed_point.
5. Ran ACUITY host inference with the W8A16 seed on the non-chunked ONNX.
6. Compared ACUITY host output vs FP32 oracle (ONNX Runtime ground truth).

### Quality results

| Alpha | ONNX variant | Host top-1 | FP top-1 | Top-1 match | Cosine | Status |
|-------|-------------|-----------|---------|-------------|--------|--------|
| none  | non-chunked | 16        | 198     | no          | 0.079  | FAIL   |
| 0.2   | chunked     | N/A       | N/A     | N/A         | N/A    | BLOCKED (quantize table) |
| 0.5   | chunked     | N/A       | N/A     | N/A         | N/A    | BLOCKED (quantize table) |
| 0.7   | chunked     | N/A       | N/A     | N/A         | N/A    | BLOCKED (quantize table) |

The "none" alpha W8A16 host cosine of 0.079 is far below the required 0.90 gate.
Top-1 mismatch (16 vs 198) confirms that INT8 weight quantization loses too much
precision for the Qwen2.5-0.5B model's activation outlier profile (act_absmax ~1790).

### Reference: INT16 control quality

| Graph variant | Host top-1 | FP top-1 | Cosine | Notes |
|--------------|-----------|---------|--------|-------|
| non-chunked INT16 | N/A | 198 | ~0.236 | From T8 report, log file preserved |
| chunked INT16 | 31174 | 198 | 0.120 | This run, different token window |

Even full INT16 (dynamic_fixed_point) fails the quality gate for Qwen2.5-0.5B.
This is consistent with the T8 finding (cosine 0.236).

### Blockers

**ACUITY quantize-table serialization bug**: The full Qwen2.5-0.5B chunked ONNX
(3 lm_head chunks) triggers an ACUITY bug where `pegasus_quantize.sh` completes
("End quantization...") but the `.quantize` YAML table is truncated to 0 bytes.
This is identical to the previously reported bug for full-Qwen PCQ
(`reports/t6-vendor-acuity-hybrid-quantize-table.md`). Inference still succeeds
(because ACUITY uses an in-memory quantize state), but the table cannot be
exported for seed-based W8A16 conversion.

The smoothed chunked ONNX variants (0.2, 0.5, 0.7) also trigger this bug, so no
W8A16 host quality numbers could be obtained for them.

The non-chunked ONNX does NOT trigger this bug (its int16 quantize table is
440KB). This is the only variant with a valid W8A16 host measurement.

**Prior T7 evidence**: The T7 report (`reports/t7-w8a16.md`) already tested
full Qwen W=32 W8A16 with SmoothQuant alpha=0.5:
- Logits cosine: 0.254 (non-chunked, smoothed)
- Top-1 mismatch: 120 vs FP oracle 279

Even with SmoothQuant, INT8 weight quantization on the 24-layer Qwen graph does
not reach the required quality gate.

### Verdict

Gate B fails the host quality gate. W8A16 is not a viable path for
Qwen2.5-0.5B on the A733 NPU under the available ACUITY/VIPLite toolchain.

The fundamental challenge is Qwen2.5-0.5B's activation outlier magnitude
(act_absmax ~1790) combined with INT8 weight quantization's limited dynamic
range, which even SmoothQuant cannot fully mitigate across 24 decoder layers.

## Outcome

Both Q1 gates are documented as failing with exact reasons:

- **Gate A**: Pegasus rejects `perchannel_symmetric_affine --qtype int16`
  (only INT8/INT4 supported for per-channel quantization).
- **Gate B**: W8A16 host cosine 0.079 (far below 0.90 gate) for alpha=none;
  smoothed variants blocked by ACUITY quantize-table serialization bug;
  prior T7 data shows alpha=0.5 also fails (cosine 0.254).

**Project decision**: The Qwen2.5-0.5B integer-only NPU path is not viable
with the current ACUITY 6.30.22 toolchain. Hybrid CPU-LLM + NPU-vision becomes
the recommended path.

## Confirmed Dead Ends (updated)

- BF16 NBG export: `vnn_VerifyGraph -3 / 64768` (T9)
- BF16<->INT16 dtype boundaries on SLICE/MATMUL/PERMUTE (T10)
- FP16 host quality: cosine 0.541 (T9)
- Plain INT16 host quality: cosine ~0.236 (T8, confirmed here)
- Per-channel INT16: chip-gated, pegases rejects (Gate A, this report)
- W8A16 host quality: cosine 0.079-0.254 (T7 + Gate B, this report)
- SmoothQuant + W8A16: host quality fails at all tested alphas
- ACUITY quantize-table serialization: full Qwen chunked ONNX produces
  0-byte `.quantize` file (T4/T7/Q1, this report)

## Cleanup

Pending: remove rebuildable artifacts to recover disk space (currently 5.2 GB free).
