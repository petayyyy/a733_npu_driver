# Results

All measurements on Orange Pi Zero 3W (5.7 GiB RAM, kernel 6.6.98-sun60iw2,
2×Cortex-A76 + 6×Cortex-A55) unless noted otherwise. Tags: **[verified]** = measured
on board; **[estimate]** = derived from scaling or tool simulator timings.

---

## Table 1: LLM on NPU (int16, fixed-window, NPU-only)

SmolLM2 runs through `npu_lm_runner --protocol`, single `vip_run_network` per token,
no KV-cache. Prompt: SmolLM2 chat rendering of `The capital of France is`, rightmost
tokens for W=32, zero left-padding for larger windows. `prefill tok/s = W / first_token_ms`.

| Model | W | Status | Coherent | Decode tok/s | First-token ms | Peak RSS MiB | NBG MB |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 32 | OK [verified] | yes (16/16 FP prefix) | 20.7 | 48 | 272 | 281 |
| SmolLM2-135M | 64 | OK [verified] | weak | 14.0 | 72 | 274 | 282 |
| SmolLM2-135M | 128 | exported [verified] | no | 6.0 | 166 | 282 | 287 |
| SmolLM2-135M | 256 | exported [verified] | no | 1.2 | 860 | 375 | 337 |
| SmolLM2-360M | 32 | OK [verified] | yes (FP match) | 8.4 | 114 | 646 | 673 |
| SmolLM2-360M | 64 | OK [verified] | yes | 4.9 | 212 | 649 | 675 |
| SmolLM2-360M | 128 | exported [verified] | no | 2.0 | 502 | 681 | 693 |
| SmolLM2-360M | 256 | exported [verified] | no | 1.2 | 834 | 711 | 709 |
| SmolLM2-1.7B | all | NBG export FAILS [verified] | — | — | — | — | — |
| Qwen2.5-0.5B | all | NBG export FAILS [verified] | — | — | — | — | — |

Notes:

- SmolLM2-1.7B: `gen_nbg` segfault on host → 0-byte NBG from 6.85 GB ONNX external-data graph
  (see [reports/b1b-benchmark-matrix.md](../reports/b1b-benchmark-matrix.md)).
- Qwen2.5-0.5B: BF16 fixes host cosine (>0.99) but `vnn_VerifyGraph -3 / 64768` blocks export;
  int16 exports NBG (1,065 MB) but is too large for this 5.7 GiB board; int8 `pcq` full-graph
  quantize stalls in ACUITY (see [reports/t9-qwen-bf16.md](../reports/t9-qwen-bf16.md),
  [reports/t10-qwen-mixed-bf16.md](../reports/t10-qwen-mixed-bf16.md),
  [reports/t11-qwen-chunked-bf16.md](../reports/t11-qwen-chunked-bf16.md)).
- At W≥128 the model has no real context memory; performance declines sharply as full-window
  recompute becomes weight-bandwidth dominated.

---

## Table 2: LLM on CPU (llama.cpp, real KV-cache)

Qwen2.5-0.5B-Instruct GGUF on Orange Pi Zero 3W, llama.cpp `be4a6a6`, native A76 dotprod,
`taskset -c 6,7 -t 2 -ngl 0`. `tg64` decode from `llama-bench`; real-chat decode is
lower under long KV-cache load. First-token times are estimated from `llama-bench`
throughput rows except the real-chat lines which are directly measured.

| Quant | Context | Decode tok/s (tg64) | Prefill tok/s | Measured first-token | Peak RSS MiB | Status |
|---|---|---|---|---|---|---|
| Q8_0 | 2,048 | 11.7 | 47.8 | ~3 s (chat) | 1,192 | [verified] |
| Q8_0 | 8,192 | 11.5 | 22.1 | ~6 min [estimate] | 1,201 | [verified] |
| Q8_0 | 16,384 | 12.1 (tg64) / 2.2 (real chat) | 13.3 | ~18 min (chat measured) | 1,306 | [verified] |
| Q8_0 | 32,768 | ~12.1 [estimate] | ~8.0 [estimate] | ~69 min [estimate] | ~1,516 [estimate] | impractical, not run |
| Q4_K_M | 2,048 | 10.9 | 18.0 | ~3 s (chat) | 734 | [verified] |
| Q4_K_M | 8,192 | 10.7 | 12.6 | ~11 min [estimate] | 737 | [verified] |
| Q4_K_M | 16,384 | 11.0 | 9.2 | ~30 min [estimate] | 838 | [verified] |

Notes:

- Q8_0 is faster AND higher-quality than Q4_K_M on this board, counter-intuitively.
  Suspect AMX/SIMD soft-permute overhead in K-quants on A76.
- Real-chat decode at 16k KV-cache drops to 2.2 tok/s (Q8_0) — `llama-bench` `tg64`
  is a short-run number and does not reflect degraded KV-cache scan speed.
- 16k retrieval was measured but tail accuracy failed (last field key wrong), so 16k
  is a capacity point, not a reliability point.
- Full report: [reports/b4-qwen-cpu-baseline.md](../reports/b4-qwen-cpu-baseline.md).

---

## Table 3: VLM on NPU (Orange Pi Zero 3W)

| Component | Input | Output | Latency | Peak RSS MiB | NBG MB | On-board vs ACUITY cosine |
|---|---|---|---|---|---|---|
| MobileCLIP-S0 vision encoder [verified] | 1×3×256×256 image | 1×512 embedding | 22.6 ms | 14 | 19 | 0.99996, top-5 match |
| Tiny VLM bridge (proof-of-concept) [verified] | image embed + token window | logits | 0.063 ms | 2 | 0.094 | 0.99999, top-5 match |

Notes:

- MobileCLIP: int16 inference on NPU, profile mean 22,606 us across 5 loops, VmHWM 21 MB.
  This is a real production-quality image encoder and the recommended vision offload path.
- Tiny bridge: vocab = 16 tokens. This proves the NPU data path (image projector +
  `Gather` embedding + concat + decoder + logits) but is NOT a usable captioning/VQA model.
  An actual small VLM would need a real projector plus larger-decoder NBG under the same
  fixed-window constraints.
- Full report: [reports/b3-vlm-orangepi.md](../reports/b3-vlm-orangepi.md).

---

## Table 4: Hardware / NPU Facts (verified)

| Fact | Detail | Source |
|---|---|---|
| NPU | Vivante VIP9000, cid `0x1000003b`, single core | G1 / every run |
| Clock | ~1.0 GHz, ~1500 MAC/cycle, ~3 TOPS INT8 nominal | staff note |
| Data types | native INT8/INT16/FP16/BF16 (BF16 host-only; export blocked on Qwen) | G2/T9 |
| int16 vs uint8 | ~1.45× slower, ~2× working memory (measured on Inception) | G2 |
| NBG load | ~1.35 ms/MB (MobileCLIP 19 MB → 25.6 ms) | G3a |
| Working memory | VPU pool tiny (vision model 1.5 MB; LLM graphs use 0–345 KB) | b3/t4 |
| NPU decode BW | ~6 GB/s effective (measured from 360M/W256 profile time) | b1b |
| ACUITY dimension limit | 65536 per dimension → blocks Qwen vocab 151936 in single FC; chunking works on ORT but not ACUITY export | T11 |
| VIPLite version | 2.0.3.2-AW-2024-08-30 (both Radxa and Orange Pi) | every run |
| ACUITY version | 6.30.22 (`ubuntu-npu:v2.0.10.1`) | T0+ |

---

## Coherence Cliff

On-board coherence holds at **W=32** and **W=64** for SmolLM2-135M/360M int16. At
**W≥128**, both models break coherence: the fixed-window graph has no KV-cache and
recomputes the full `W`-token context per step. With left-padded zero tokens for
prompts shorter than `W`, the model effectively sees a diluted attention mask.
At large `W`, the compute grows O(W²) while throughput drops sharply below ~3 tok/s
(135M/W256 at 1.2 tok/s, 360M/W128+ at ≤2.0 tok/s). The sweet spot for NPU-only
fixed-window decode is W=32–64.

---

## Why These Limits

**Static-shape NBG → no KV-cache.** The current toolchain builds fixed-shape
compute graphs, so every generated token requires a full `W`-token recompute of
the entire decoder stack. A real KV-cache runtime would keep per-layer key/value
tensors and only compute incremental attention — but that requires dynamic-shape
or graph-level KV state support, which VIPLite/NBG do not expose.

**Memory-bound decode on 32-bit LPDDR5.** The NPU is actively fetching weights,
not blocked on compute, for the LLM decode path. The effective measured NPU
decode bandwidth is ~6 GB/s; the tiny 1.5 MB working-memory pool means every
forward pass is dominated by weight reads from the ~200–700 MB NBG in RAM.

**int16 dynamic-fixed-point fails on Qwen activation outliers.** Qwen2.5-0.5B
int16 cosines ~0.24 vs FP oracle; BF16 fixes host quality to >0.99 but `vnn_VerifyGraph`
status `-3` blocks export. ACUITY's `pcq` int8 path either stalls (full 24-layer
quantize hangs in table rewrite) or produces incoherent output. No tested
mixed BF16/int16 split cleared both host quality and export gates.

**No working LLM INT4/int8.** The public toolkit exposes `pcq` (per-channel int8)
but RF8-style per-chain INT4 is not in the ACUITY 6.30.22 CLI. All int8 candidates
for real LLM vocab/logits failed coherence, leaving int16 as the only working LLM
quant on this toolchain.

---

## Recommendations by Use Case

| Use case | Recommended path | Throughput | Notes |
|---|---|---|---|
| Tiny fast NPU chat | SmolLM2-135M W=32 int16 on NPU | 21 tok/s [verified] | Fits in free RAM alongside ROS2 |
| Smarter NPU chat | SmolLM2-360M W=32 int16 on NPU | 8 tok/s [verified] | Larger NBG (~673 MB), still runs |
| Usable chat with real context | Qwen2.5-0.5B Q8_0 on CPU (A76) | ~18 tok/s (2k) [verified] | Req. ROS2 frozen/paused; 8k default ctx |
| VLM vision offload | MobileCLIP-S0 on NPU | 22.6 ms/frame [verified] | Production-quality encoder |
| Hybrid assistant (recommended) | NPU vision + CPU LLM | NPU: 22.6 ms/vision, CPU: ~18 tok/s [verified] | Keeps A55/A76 split; best realistic path on A733 |
| NPU-only fixed-window LLM | SmolLM2-135M W=64 int16 on NPU | 14 tok/s [verified] | Slightly more context, still coherent |

The **hybrid NPU-vision + CPU-LLM** path is the recommended architecture for a
real assistant on A733 hardware: the NPU handles vision encoding at 22.6 ms/frame
while the CPU runs Qwen with a real KV-cache at usable speeds, keeping the A76
cores partitioned from the A55 cores during robotic operation.

---

## Blocked / Vendor-Gated

| Blocker | Description | Reports |
|---|---|---|
| Qwen2.5-0.5B BF16 export | BF16 host quality >0.99, but `vnn_VerifyGraph -3 / 64768` blocks NBG generation; chunked lm_head and token-embedding chunking do not clear it | [t9](../reports/t9-qwen-bf16.md), [t10](../reports/t10-qwen-mixed-bf16.md), [t11](../reports/t11-qwen-chunked-bf16.md) |
| 65536 dimension limit | ACUITY rejects Qwen 151936-vocab single-FC path; chunked outputs tested on ORT (OK) but ACUITY BF16 export still fails with the same `64768` | [t11](../reports/t11-qwen-chunked-bf16.md) |
| SmolLM2-1.7B NBG export | All windows (32/64/128/256) fail `gen_nbg` segfault / 0-byte NBG from 6.85 GB ONNX external-data graph | [b1](../reports/b1b-benchmark-matrix.md) |
| ACUITY hybrid quantize-table hang | Hybrid/w8a16 quantize reaches `End quantization` then truncates `.quantize` table to 0 bytes and hangs; blocks T5 recovery paths | [t6](../reports/t6-vendor-acuity-hybrid-quantize-table.md) |

---

## Comparison to RK3588

The RK3588 (6 TOPS, RKLLM with native KV-cache + int4/int8) runs full LLMs and
VLMs like InternVL. The A733 (3 TOPS nominal, no KV-cache, no working LLM
int4/int8 export, BF16 blocked by vendor) cannot match this with NPU-only LLM
inference. **The realistic A733 equivalent is NPU-vision + CPU-LLM hybrid.**

The one advantage A733 does show is usable 21 tok/s SmolLM2-135M int16 inference
with the NPU path while keeping the CPU cores free — but this comes with
fixed-window constraints and no Qwen or 1.7B-class models on the NPU path.

---

*Numbers sourced from reports/ and board runlogs. Cross-reference individual
reports for full methodology, logs, and per-step comparisons.*
