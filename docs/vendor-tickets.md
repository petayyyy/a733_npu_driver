# Vendor Tickets

Consolidated blocker packets ready to file with Radxa/Allwinner/VeriSilicon.
Each contains the exact error, op, shapes, toolchain version, and reproducer
log path.

## Ticket 1: BF16 NBG Export — vnn_VerifyGraph -3 / 64768

**Priority**: Critical (blocks Qwen2.5-0.5B on NPU)

**Model**: Qwen2.5-0.5B-Instruct, W=32, fixed-shape decoder (24 layers, hidden=896,
14 q heads, 2 kv heads, vocab=151936)

**Toolchain**: ACUITY 6.30.22, Vivante IDE 5.11.0, Docker `ubuntu-npu:v2.0.10.1`.
Target: `VIP9000NANODI_PLUS_PID0X1000003B`. Board: Orange Pi Zero 3W (A733),
VIPLite 2.0.3.2-AW-2024-08-30.

**What works**: BF16 quantization and host inference pass the FP oracle gate:
logits cosine 0.991, top-1 match.

**What fails**: NBG generation. Error:

```
E [main.c:vnn_VerifyGraph:93] CHECK STATUS(-3:The requested set of parameters
  produce a configuration that cannot be supported.)
Fatal model generation error: 64768
Error(1),Warning(0)
```

No node name emitted. Export directory `_nbg_unify` is missing.

**Variants tested (all fail)**:
- Full monolithic BF16: `vnn_VerifyGraph -3 / 64768`
- Chunked lm_head (3 chunks ≤50646, with Concat): same `64768`
- Chunked lm_head without Concat (3 separate outputs): same `64768`
- Chunked lm_head + chunked token embedding: moves to `DATACONVERT node[13], 65280`

**Control**: int16 NBG export succeeds for the same ONNX (but quality fails:
cosine 0.236). The ONNX builder is verified correct (ORT vs FP oracle cosine
1.000000000).

**Logs**: `logs/host/t9-qwen25-05b-w32-bf16-convert.err.log`,
`logs/host/t11-qwen25-w32-chunked-no-concat-bf16-convert.err.log`,
`logs/host/t11-qwen25-w32-chunked-embed-chunks-bf16-convert.err.log`

**Reports**: [t9](../reports/t9-qwen-bf16.md), [t10](../reports/t10-qwen-mixed-bf16.md),
[t11](../reports/t11-qwen-chunked-bf16.md)

**Request**: Enable BF16 NBG generation for this graph. Either fix `vnn_VerifyGraph`
to accept the BF16 body graph, or document the legal BF16→INT16 boundary placement
so we can manually insert DataConvert nodes.

---

## Ticket 2: ACUITY Hybrid Quantize-Table Hang

**Priority**: High (blocks W8A16 recovery paths)

**Model**: SmolLM2-135M-Instruct, W=32 (also affects Qwen2.5-0.5B)

**Error**: Hybrid quantize (`pegasus_quantize_hybird.sh`) reaches
`End quantization...`, starts `Dump net quantize tensor table to ... .quantize`,
truncates YAML `.quantize` table to 0 bytes, then hangs CPU-active indefinitely
(no exit, no error).

The intermediate `.quantize.json` is produced correctly (contains the
`dtype_converter` nodes). Only the YAML serialization fails.

**Reproducers**:
- Seeded hybrid from calibrated PCQ table (589 dtype_converters)
- Seeded hybrid from mixed PCQ/int16 table (587 dtype_converters)

**Logs**: `logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.log`,
`logs/host/t5-smollm2-w32-mixed-hybrid-pcq-convert.log`

**Report**: [t6](../reports/t6-vendor-acuity-hybrid-quantize-table.md)

**Request**: Fix YAML `.quantize` table serialization for graphs with large
numbers of `dtype_converter` nodes. Alternatively, document the supported
converter per-graph limit.

---

## Ticket 3: INT16 Depth Collapse on Block-Chained Qwen

**Priority**: Medium (academic interest; hybrid CPU approach works)

**Model**: Qwen2.5-0.5B, 26 NBGs chained (1 embedding + 24 blocks + 1 final),
int16 dynamic fixed point. Total NBG ~1,063 MB.

**Host simulation**: Cosine 0.975, full token match (10/10 tokens match FP oracle).

**Hardware**: All 26 NBGs load and chain (load-once, stable timing, 6.6 tok/s).
But generated tokens are degenerate (empty/invalid token IDs). Hidden state
magnitudes reach 1592 forcing `dfp=4` (granularity 0.0625) on layers 4-20;
24 chained quantize-dequantize cycles compound error far beyond host simulation.

**Report**: [q2](../reports/q2-qwen-block-nbg.md)

**Request**: Investigate whether a wider-exponent DFP format or a
per-layer scale calibration pass could reduce boundary quantization error
across chained int16 NBGs.

---

## Ticket 4: SmolLM2-1.7B gen_nbg Segfault

**Priority**: Medium

**Model**: SmolLM2-1.7B-Instruct, all windows (32/64/128/256). ONNX external-data
graph ~6.85 GB.

**ONNX builder gate**: Passes (ORT vs FP oracle cosine ~1.0, top-1 match for
all windows).

**NBG export**: `gen_nbg` segfaults, produces 0-byte `network_binary.nb`.
ACUITY import, quantize, and host inference complete successfully. The crash
occurs during the C VIP graph compilation phase.

**Report**: [b1b](../reports/b1b-benchmark-matrix.md)

**Request**: Investigate gen_nbg stability for large ONNX external-data graphs
(>6 GB FP32 initializer payload). Possible root causes: out-of-memory in
gen_nbg process, external-data file handling race, or a graph-size limit in the
VIP compiler.

---

## Ticket 5: SmolVLM SigLIP Conv Shape Inference Crash

**Priority**: Low (workaround: use CPU VLM)

**Model**: SmolVLM-256M-Instruct → Idefics3VisionTransformer → SigLIP encoder.
ONNX: 357 MB, 1×3×512×512 → 1×64×576, opset 17.

**Error**: ACUITY import crashes with:

```
IndexError: list index out of range
File .../smart_toolkit.py, line 1571, in _conv_shape
```

**Location**: Patch embedding Conv2d: kernel=16, stride=16, padding=0,
in_channels=3, out_channels=768.

**Tested**: opset 17 and 15, with and without Cast fix. NonZero op was removed
(verified ONNX RT cosine 1.00000000). Same Conv error in all variants.

**Report**: [v2](../reports/v2-hybrid-vlm-npu-offload.md)

**Request**: Investigate Conv2d shape inference for patch embedding with
kernel=16, stride=16 on 512×512 inputs. Possibly an unhandled edge case in
`_conv_shape`.

---

## Ticket 6: Per-Channel INT16 Quantizer Rejected

**Priority**: Feature request (would enable Qwen int16 quality path)

**Command**: `pegasus quantize --quantizer perchannel_symmetric_affine --qtype int16`

**Error**: Pegasus 6.30.22 rejects:

```
Please specify valid quantize qtype for quantizer 'perchannel_symmetric_affine',
in ['int8', 'int4']
```

**Report**: [q1](../reports/q1-qwen-int8-gates.md)

**Request**: Add int16 as a supported qtype for `perchannel_symmetric_affine`
quantizer. This would allow per-channel symmetric int16 weights with int16
activations, potentially holding Qwen's outliers without BF16.

---

## Toolchain Versions (all tickets)

- ACUITY: 6.30.22 (`/root/acuity-toolkit-whl-6.30.22/bin`)
- Docker: `ubuntu-npu:v2.0.10.1`
- Vivante IDE: 5.11.0 (from pegasus output)
- VIPLite runtime: 2.0.3.2-AW-2024-08-30 (board)
- Target: `VIP9000NANODI_PLUS_PID0X1000003B`
- Board: Orange Pi Zero 3W, Allwinner A733, kernel 6.6.98-sun60iw2

## Reference: TIM-VX Legal Dtype Boundaries

Verified from github.com/VeriSilicon/TIM-VX, `src/tim/vx/internal/src/ops/`:

- **MATMUL**: BF16 only as BF16,BF16→BF16 (no mixed); int16 only as INT16,INT16→INT16
- **SLICE**: BF16→BF16, INT16→INT16. NOT BF16→INT16
- **DATACONVERT**: INT16→BF16, BF16→F16/F32, F16→INT16. NOT direct BF16→INT16
- **PERMUTE**: same-type only

The legal bridge is BF16 → DataConvert(F16) → DataConvert(INT16), but ACUITY
6.30.22 does not emit standalone DataConvert nodes before SLICE/MATMUL/PERMUTE
in BF16-heavy graphs.
