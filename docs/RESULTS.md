# Results

All measurements on Orange Pi Zero 3W (6 GB LPDDR5, kernel 6.6.98-sun60iw2,
2×Cortex-A76 + 6×Cortex-A55) unless noted otherwise. Tags: **[verified]** = measured
on board; **[estimate]** = derived from scaling or tool simulator timings.

---

## Table 1: LLM on NPU (int16, fixed-window, NPU-only)

SmolLM2 runs through `npu_lm_runner --protocol`, no KV-cache. Prompt: SmolLM2
chat template `The capital of France is`, rightmost tokens for W=32, zero
left-padding for larger windows. `prefill tok/s = W / first_token_ms`.

| Model | W | Status | Coherent | Decode tok/s | First-token ms | Peak RSS MiB | NBG MB |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 32 | OK [verified] | yes (16/16 FP prefix) | 20.7 | 48 | 272 | 281 |
| SmolLM2-135M | 64 | OK [verified] | weak | 14.0 | 72 | 274 | 282 |
| SmolLM2-135M | 128 | exported [verified] | no | 6.0 | 166 | 282 | 287 |
| SmolLM2-135M | 256 | exported [verified] | no | 1.2 | 860 | 375 | 337 |
| SmolLM2-360M | 32 | OK [verified] | yes | 8.4 | 114 | 646 | 673 |
| SmolLM2-360M | 64 | OK [verified] | yes | 4.9 | 212 | 649 | 675 |
| SmolLM2-360M | 128 | exported [verified] | no | 2.0 | 502 | 681 | 693 |
| SmolLM2-360M | 256 | exported [verified] | no | 1.2 | 834 | 711 | 709 |
| SmolLM2-1.7B | all | NBG export FAILS [verified] | — | — | — | — | — |
| Qwen2.5-0.5B (monolithic) | all | NBG export FAILS or incoherent [verified] | — | — | — | — | — |
| Qwen2.5-0.5B (block-chain, 26 NBG) | 32 | mechanical PASS, quality FAIL [verified] | no | 6.6 | 150 | — | ~1,063 |

### Qwen block-chain detail (Q2 Gate 2C)

The 26-NBG block-chain runs mechanically (load-once, stable timing) but
degenerates to empty/invalid tokens at 6.6 tok/s. Host simulation predicted
cosine 0.975 and full token match, but hardware int16 dynamic-fixed-point
accumulates 24 boundary quantize-dequantize cycles causing total collapse.
See [reports/q2-qwen-block-nbg.md](../reports/q2-qwen-block-nbg.md).

Notes:

- SmolLM2-1.7B: `gen_nbg` segfault → 0-byte NBG from 6.85 GB ONNX
  (see [reports/b1b-benchmark-matrix.md](../reports/b1b-benchmark-matrix.md)).
- Qwen monolithic: int16 cosine 0.236, FP16 0.541, BF16 `vnn_VerifyGraph -3`,
  W8A16 cosine 0.079, per-channel int16 rejected by pegasus (only INT8/INT4).
  Every config fails. (See [reports/t8](../reports/t8-qwen-int16-port.md) through
  [reports/q1](../reports/q1-qwen-int8-gates.md).)
- At W≥128 the model has no real context memory; performance declines sharply.

---

## Table 2: LLM on CPU (llama.cpp, real KV-cache)

Qwen2.5-0.5B-Instruct GGUF on Orange Pi Zero 3W, llama.cpp `be4a6a6`, native
A76 dotprod. First-token times are estimated from benchmark throughput rows
except the real-chat lines which are directly measured.

### B4b CPU utilization sweep (ctx 2048, Q8_0, decode 128 tokens)

| `-t` | Cores | Core type | Prefill tok/s | Decode tok/s | Avg CPU% | % of 8 cores | Peak RSS |
|---|---|---|---|---|---|---|---|
| 1 | 0 | A55 | 18.2 | 6.3 | ~100% | 13% | 1,188 MiB |
| 2 | 0,1 | 2×A55 | 36.7 | 10.7 | ~199% | 25% | 1,195 MiB |
| 3 | 0-2 | 3×A55 | 54.7 | 13.7 | ~300% | 38% | 1,189 MiB |
| 4 | 0-3 | 4×A55 | 71.4 | 14.8 | ~398% | 50% | 1,198 MiB |
| 1 | 6 | A76 | 64.8 | 16.5 | ~100% | 13% | 1,106 MiB |
| **2** | **6,7** | **2×A76** | **128.7** | **18.0** | **~199%** | **25%** | **1,109 MiB** |
| 4 | 4-7 | 2×A76+2×A55 | 129.6 | 16.8 | ~391% | 49% | 1,141 MiB |
| 6 | 2-7 | 2×A76+4×A55 | 157.3 | 16.1 | ~590% | 74% | 1,201 MiB |
| 8 | 0-7 | all 8 | 161.9 | 13.7 | ~741% | 93% | 1,196 MiB |

Key findings: 1×A76 (16.5 tok/s) nearly saturates memory bandwidth; adding 2nd
A76 yields only +9% (18.0 tok/s). A55 single-core is 2.6× slower (6.3 tok/s).
8 threads degrade decode vs 2 A76. **Best config: `-t 2 taskset -c 6,7` at
18.0 tok/s, 25% CPU, 6 A55 cores free.** See [reports/b4b-cpu-utilization.md](../reports/b4b-cpu-utilization.md).

### Context sweep (taskset -c 6,7, -t 2)

| Quant | Context | Decode tok/s (tg64) | Prefill tok/s | Measured first-token | Peak RSS MiB | Status |
|---|---|---|---|---|---|---|
| Q8_0 | 2,048 | 11.7 | 47.8 | ~3 s (chat) | 1,192 | [verified] |
| Q8_0 | 8,192 | 11.5 | 22.1 | ~6 min [estimate] | 1,201 | [verified] |
| Q8_0 | 16,384 | 12.1 (tg64) / 2.2 (real chat) | 13.3 | ~18 min (chat measured) | 1,306 | [verified] |
| Q8_0 | 32,768 | ~12.1 [estimate] | ~8.0 [estimate] | ~69 min [estimate] | ~1,516 [estimate] | impractical |
| Q4_K_M | 2,048 | 10.9 | 18.0 | ~3 s (chat) | 734 | [verified] |
| Q4_K_M | 8,192 | 10.7 | 12.6 | ~11 min [estimate] | 737 | [verified] |
| Q4_K_M | 16,384 | 11.0 | 9.2 | ~30 min [estimate] | 838 | [verified] |

Q8_0 is faster AND higher-quality than Q4_K_M on this board. Long context
works (16k) but decode drops to 2.2 tok/s in real chat and tail retrieval
is unreliable.

## Table 3: VLM Results

### VLM on NPU

| Component | Input | Output | Latency | Peak RSS MiB | NBG MB | Cosine vs ACUITY |
|---|---|---|---|---|---|---|
| MobileCLIP-S0 [verified] | 1×3×256×256 | 1×512 embedding | 22.6 ms | 14 | 19 | 0.99996 |
| Tiny VLM bridge (PoC) [verified] | embed + 4 tokens | logits | 0.063 ms | 2 | 0.094 | 0.99999 |

### VLM on CPU

| Model | Quant | Decode tok/s | Peak RSS | Accuracy | Status |
|---|---|---|---|---|---|
| SmolVLM-256M-Instruct | Q8_0 | 52.6 | 634 MB | Accurate (dog/cat/moon landing) | [verified] |
| SmolVLM-500M-Instruct | Q8_0 | 22.3 | ~1.2 GB | Accurate (more detail) | [verified] |

SmolVLM-256M Q8_0 is the recommended VLM: fast (52.6 tok/s), accurate, 634 MB
RSS leaves ~2.3 GB for ROS2. Its SigLIP vision encoder cannot be exported to
NPU (ACUITY Conv shape crash, see V2 report).

### VLM on NPU (attempted, blocked)

| Component | Blocker | Status |
|---|---|---|
| SmolVLM SigLIP encoder | ACUITY `_conv_shape` IndexError (patch embedding Conv) | [verified blocked] |

## Table 4: Hardware / NPU Facts (verified)

| Fact | Detail | Source |
|---|---|---|
| NPU | Vivante VIP9000, cid `0x1000003b`, single core | G1 |
| Clock | ~1.0 GHz, ~1500 MAC/cycle, ~3 TOPS INT8 | staff note |
| Data types | native INT8/INT16/FP16/BF16 (BF16 host-only; export blocked) | G2/T9 |
| int16 vs uint8 | ~1.45× slower, ~2× working memory (measured on Inception) | G2 |
| NBG load | ~1.35 ms/MB (MobileCLIP 19 MB → 25.6 ms) | G3a |
| NPU decode BW | ~6 GB/s effective (measured from 360M/W256 profile time) | b1b |
| ACUITY dimension limit | 65536 per dimension | T11 |
| Per-channel int16 | NOT in ACUITY 6.30.22 (only INT8/INT4 for `perchannel_symmetric_affine`) | Q1 |
| VIPLite version | 2.0.3.2-AW-2024-08-30 | every run |
| ACUITY version | 6.30.22 (`ubuntu-npu:v2.0.10.1`) | T0+ |

---

## Coherence Cliff

On-board coherence holds at **W=32** and **W=64** for SmolLM2-135M/360M int16.
At **W≥128** both models break coherence: no KV-cache forces full-window
recompute per token with zero-padded diluted attention. Decode drops below
~3 tok/s by W=128 (W=256: 1.2 tok/s). The sweet spot is W=32-64.

---

## Why These Limits

**Static-shape NBG → no KV-cache.** The toolchain builds fixed-shape graphs;
every generated token requires full `W`-token recompute. VIPLite/NBG do not
expose dynamic-shape KV state.

**Memory-bound decode on 32-bit LPDDR5.** NPU decode bandwidth is ~6 GB/s;
each forward pass is dominated by weight reads from the NBG in system RAM.

**int16 dynamic-fixed-point fails on Qwen activation outliers.** Qwen's
act_absmax ~1790, RMS-squared tensors ~2.65e6. int16 DFP cosine 0.236, FP16
0.541, W8A16 0.079. BF16 fixes host quality (>0.99) but won't compile
(`vnn_VerifyGraph -3 / 64768`). Per-block NBG chaining avoids the monolithic
limit but collapses at 24-deep chaining on hardware (6.6 tok/s, garbage output).

**No working LLM INT4/int8.** ACUITY exposes `pcq` (per-channel int8) but it
produces incoherent output for real LLM vocab/logits. Per-channel int16 is
rejected by pegasus (only INT8/INT4 for `perchannel_symmetric_affine`).

**SmolLM2-1.7B: NBG export crashes.** `gen_nbg` segfault on 6.85 GB
external-data ONNX — 0-byte NBG.

---

## Recommendations by Use Case

| Use case | Path | Throughput | Notes |
|---|---|---|---|
| Fast NPU chat | SmolLM2-135M W=32 int16 on NPU | 21 tok/s [verified] | CPU cores free for ROS2 |
| Smarter NPU chat | SmolLM2-360M W=32 int16 on NPU | 8 tok/s [verified] | Larger NBG, still runs |
| Chat with real context | Qwen2.5-0.5B Q8_0 on CPU (2×A76) | 18 tok/s [verified] | 6 A55 cores free, 8k default ctx |
| Image chat (lightweight) | SmolVLM-256M-Instruct Q8_0 on CPU | 53 tok/s [verified] | 634 MB RSS, leaves 2.3 GB for ROS2 |
| VLM vision offload | MobileCLIP-S0 on NPU | 22.6 ms/frame [verified] | Production-quality encoder |
| **Hybrid assistant (recommended)** | NPU vision + CPU LLM | 22.6 ms/vision + 18 tok/s | Best realistic A733 path |

The **hybrid NPU-vision + CPU-LLM** path is the recommended architecture: NPU
handles vision encoding while CPU runs Qwen/SmolVLM with real KV-cache, keeping
the 6 A55 cores free for ROS2.

---

## Blocked / Vendor-Gated

| Blocker | Description | Reports |
|---|---|---|
| Qwen BF16 export | Host quality passes (cosine >0.99), NBG blocked by `vnn_VerifyGraph -3 / 64768` | [t9](../reports/t9-qwen-bf16.md), [t10](../reports/t10-qwen-mixed-bf16.md), [t11](../reports/t11-qwen-chunked-bf16.md) |
| Qwen int16 quality | Cosine 0.236; per-block chain degenerates (cosine 0.975 host → garbage on hardware) | [t8](../reports/t8-qwen-int16-port.md), [q2](../reports/q2-qwen-block-nbg.md) |
| Qwen W8A16 quality | Cosine 0.079; SmoothQuant variants blocked by ACUITY quantize-table bug | [t7](../reports/t7-w8a16.md), [q1](../reports/q1-qwen-int8-gates.md) |
| Per-channel int16 | Pegasus rejects: only INT8/INT4 for `perchannel_symmetric_affine` | [q1](../reports/q1-qwen-int8-gates.md) |
| 65536 dimension limit | Qwen vocab 151936 > 65536; chunking works on ORT but not ACUITY BF16 export | [t11](../reports/t11-qwen-chunked-bf16.md) |
| SmolLM2-1.7B NBG export | `gen_nbg` segfault, 0-byte NBG from 6.85 GB ONNX | [b1b](../reports/b1b-benchmark-matrix.md) |
| SmolVLM SigLIP on NPU | ACUITY Conv shape crash (`_conv_shape` IndexError) | [v2](../reports/v2-hybrid-vlm-npu-offload.md) |
| ACUITY hybrid quantize-table | Reaches `End quantization`, truncates `.quantize` to 0 bytes, hangs | [t6](../reports/t6-vendor-acuity-hybrid-quantize-table.md) |

All blockers verified on hardware. Vendor ticket summaries at [docs/vendor-tickets.md](vendor-tickets.md).

---

## Comparison to RK3588

The RK3588 (6 TOPS, RKLLM with KV-cache + int4/int8) runs full LLMs/VLMs like
InternVL. The A733 (3 TOPS, no KV-cache, no working LLM int4/int8 export, BF16
blocked) cannot match this with NPU-only LLM inference. **The realistic
A733 equivalent is NPU-vision + CPU-LLM hybrid.**

A733 advantage: 21 tok/s SmolLM2-135M int16 on NPU preserves CPU cores, and
MobileCLIP-S0 vision at 22.6 ms is a solid offload target.

---

*Numbers sourced from reports/ and board runlogs. Cross-reference individual
reports for full methodology, logs, and per-step comparisons.*
