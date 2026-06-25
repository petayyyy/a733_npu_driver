# A733 / Vivante VIP9000 NPU — Blocker List

Every blocker verified on hardware or host with preserved logs.
Source of truth for what does NOT work and why.

## Setup

- SoC: Allwinner A733. NPU: Vivante VIP9000, cid `0x1000003b`, single core, ~1.0 GHz, ~3 TOPS INT8
- Board: Orange Pi Zero 3W, 6 GB RAM, kernel 6.6.98, VIPLite 2.0.3.2
- Toolchain: ACUITY 6.30.22 (`ubuntu-npu:v2.0.10.1`), Vivante IDE 5.11.0
- Flow: ONNX → pegasus import/quantize/inference/export → NBG → VIPLite
- "int16" = dynamic fixed point (single power-of-2 scale per tensor), NOT IEEE fp16

## Blocker 1 — Qwen2.5-0.5B monolithic NBG: every config fails

Qwen2.5-0.5B (24 layers, hidden 896, vocab 151936) has activation outliers
(act_absmax ~1790, RMS-squared tensors ~2.65e6). Every monolithic config fails:

| Config | Host cosine | Export? | Board coherent? | Report |
|---|---|---|---|---|
| int16 DFP | 0.236 | Yes | No (cosine too low) | [t8](../reports/t8-qwen-int16-port.md) |
| FP16 | 0.541 | Yes | No (cosine too low) | [t9](../reports/t9-qwen-bf16.md) |
| BF16 | **0.991** | **No** (`vnn_VerifyGraph -3 / 64768`) | N/A | [t9](../reports/t9-qwen-bf16.md) |
| W8A16 (no smoothing) | **0.079** | Yes | No (cosine too low) | [q1](../reports/q1-qwen-int8-gates.md) |
| W8A16 + SmoothQuant (all α) | — | Blocked (quantize-table bug) | N/A | [t7](../reports/t7-w8a16.md), [q1](../reports/q1-qwen-int8-gates.md) |
| Per-channel int16 | — | Pegasus rejects: only INT8/INT4 | N/A | [q1](../reports/q1-qwen-int8-gates.md) |
| Chunked lm_head BF16 | — | Same `vnn_VerifyGraph -3 / 64768` | N/A | [t11](../reports/t11-qwen-chunked-bf16.md) |
| Mixed BF16/int16 boundaries | top-1 match, no cosine | Fails at SLICE/MATMUL/PERMUTE dtype boundaries (65280) | N/A | [t10](../reports/t10-qwen-mixed-bf16.md) |

### Root cause

BF16 is the only format that preserves Qwen quality (cosine 0.991), but the NBG
compiler cannot lay out the BF16-heavy graph. TIM-VX source (github.com/VeriSilicon/TIM-VX)
confirms illegal dtype boundaries: MATMUL requires BF16→BF16 only; SLICE
rejects BF16→INT16; DATACONVERT supports BF16→F16→INT16 but ACUITY doesn't
emit standalone DataConvert nodes before SLICE/MATMUL/PERMUTE.

The 65536 dimension limit (Qwen vocab 151936) exacerbates but is not the only
barrier — chunked lm_head BF16 fails with the same `64768` even with all dims
<65536.

## Blocker 2 — Qwen2.5-0.5B block-chain: mechanical pass, quality fails

Per-decoder-block NBGs (26 total: 1 embedding + 24 blocks + 1 final) export
correctly with int16. Host simulation predicts cosine 0.975 with full token
match. On hardware (Q2 Gate 2C):

- 26 NBGs load, chain, and run at 6.6 tok/s (load-once, stable timing)
- Output: **all degenerate tokens** (empty/invalid)
- Root cause: int16 dynamic-fixed-point collapses over 24 chained
  quantize-dequantize cycles on hardware, worse than host simulation predicted
- Report: [q2](../reports/q2-qwen-block-nbg.md)

## Blocker 3 — SmolLM2-1.7B NBG export crashes

All windows (32/64/128/256) pass FP builder gate (cosine ~1.0). NBG export
fails during `gen_nbg`: segfault → 0-byte `network_binary.nb` from 6.85 GB
external-data ONNX graph.

Report: [b1b](../reports/b1b-benchmark-matrix.md)

## Blocker 4 — SmolVLM SigLIP vision encoder on NPU (TOOLCHAIN RESOLVED, E2E PENDING)

SmolVLM-256M SigLIP encoder (Idefics3 wrapper): ONNX exports (357 MB,
1×3×512×512 → 1×64×576) but ACUITY 6.30.22 crashes at Conv shape inference
(`_conv_shape`, smart_toolkit.py:1571 — IndexError) on the patch embedding
Conv (kernel=16, stride=16, in=3, out=768).

**V2b/V2c update**: Conv→Reshape+MatMul rewrite **succeeded**. NBG exports
(271 MB, Error 0), runs on Orange Pi NPU at 5.94 sec/inference (verified:
`cid=0x1000003b`, `vpm run ret=0`). Rebuilt with real-image calibration
(fl=15, range [-1,1]). Toolchain path fully proven. End-to-end VLM accuracy
pending embedding injection into llama.cpp decoder. Not resolved until
answers validated.

Reports: [v2](../reports/v2-hybrid-vlm-npu-offload.md), [v2b](../reports/v2b-smolvlm-vision-npu-retry.md), [v2c](../reports/v2c-vlm-npu-e2e.md)

## Blocker 5 — No KV-cache

NBG graphs are static-shape. No incremental attention: every token recomputes
the full window. Coherence holds at W=32-64, breaks at W≥128. TIM-VX has
VARIABLE tensor but no working KV-cache precedent on VIP9000.

## Blocker 6 — ACUITY hybrid quantize-table hang

Hybrid/w8a16 quantize reaches `End quantization...` then truncates YAML
`.quantize` table to 0 bytes and hangs CPU-active. Blocks all T5 recovery
paths requiring hybrid qtypes. Affects SmolLM2 and Qwen.

Report: [t6](../reports/t6-vendor-acuity-hybrid-quantize-table.md)

## Blocker 7 — ACUITY host int16 cosine is unreliable

`pegasus inference` int16 host cosine gave false negatives (failed known-good
SmolLM2-135M/W32). Use on-board generation for coherence gate, FP oracle
(ONNX Runtime) for builder correctness.

## Confirmed dead ends (do not retry)

| Path | Reason |
|---|---|
| BF16 NBG export | `vnn_VerifyGraph -3 / 64768` on every Qwen variant |
| Per-channel int16 | Pegasus rejects: only INT8/INT4 supported |
| W8A16 | Cosine 0.079 (no smoothing); smoothing blocked by quantize-table bug |
| QDQ-ONNX import | ACUITY ignores QDQ scales |
| TVM/TIM-VX hand-built | Lack RMSNorm/Gather/Slice/SwiGLU coverage |
| 32k Qwen CPU context | ~69 min first-token; impractical |

## Reference

Rockchip RK3588 (6 TOPS, RKLLM with KV-cache + int4/int8) runs full small LLMs/
VLMs (Qwen2.5, InternVL) natively. The A733 at 3 TOPS without KV-cache or
working LLM BF16 cannot match this. The equivalent A733 architecture is hybrid:
NPU vision + CPU LLM.
