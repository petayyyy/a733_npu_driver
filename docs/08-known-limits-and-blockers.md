# 08 — Known Limits and Blockers

An honest catalog of what the A733 Vivante VIP9000 NPU toolchain can and
cannot do for LLM/VLM inference as of June 2026. All items are board-verified
or host-verified with preserved logs.

## Fundamental limits

### Fixed-window, no KV-cache

The ACUITY/VIPLite toolchain produces static-shape NBG graphs. There is no
incremental attention computation: every generated token requires a full
`W`-token recompute through all decoder layers. This means:

- Throughput is O(W²) — doubling the window roughly quarters the tok/s
- Coherence breaks above W=64 because the model sees only a sliding window
  with no long-range attention state
- There is no separate "prefill" phase — the first token already pays
  full cost

**If the vendor provides a KV-cache runtime**: the same int16 graphs could
potentially run with incremental decode, improving throughput and enabling
longer coherent context. This is the single biggest missing piece.

### 3 TOPS memory-bound decode

Effective NPU decode bandwidth is measured at ~6 GB/s. The NPU is actively
fetching weights, not blocked on compute. The 1.5 MB working-memory pool
for vision models (0-345 KB for LLM graphs) means every forward pass is
dominated by weight reads from the NBG in system RAM over a 32-bit LPDDR5 bus.

### No working LLM int4/int8

The public ACUITY 6.30.22 toolchain exposes `pcq` (per-channel int8
asymmetric affine) but not RF8-style per-chain int4. Int8 `pcq` exports
and runs for SmolLM2 but produces incoherent output (quantization error in
activations at full model depth). All coherent LLM decodes use int16, which
is ~1.45× slower and ~2× larger than an ideal int8 path.

## Model-specific blockers

### Qwen2.5-0.5B — no NBG export path works

| Attempt | Host quality (cosine vs FP) | NBG export | Blocker |
|---|---|---|---|
| int16 DFP | 0.24 | Exports (1,065 MB) | Fails host quality gate (activation outliers) |
| BF16 | 0.99 (passes) | FAILS | `vnn_VerifyGraph -3 / 64768` |
| FP16 | 0.54 | Exports (991 MB) | Fails host quality gate |
| pcq full | N/A | Never completes | ACUITY quantize hangs after `End quantization` |
| pcq seeded | N/A | Exports (588 MB) | Board killed (OOM on 1 GiB Radxa, untested on 5.7 GiB OPi) |
| Mixed BF16/int16 (v1) | — | FAILS | `SLICE` dtype boundary: BF16 → DFP INT16 |
| Mixed BF16/int16 (v2) | top-1 match | FAILS | `vnn_VerifyGraph -3 / 64768` |
| Mixed BF16/int16 (outliers) | top-1 match | FAILS | `MATRIXMUL` dtype: DFP INT16→BFLOAT16 |
| Chunked BF16 | N/A | FAILS | Same `vnn_VerifyGraph -3 / 64768` with or without Concat |
| Chunked BF16 + embed chunks | N/A | FAILS | `DATACONVERT` node[13]: 65280 |

The **65536 dimension limit** (T11 finding): ACUITY rejects operations
with a dimension >65536. Qwen vocab is 151,936. Chunking the vocab works
in ONNX Runtime (cosine 1.000000000) but does not clear the BF16 export
blocker, indicating the limit is not the only problem.

Vendor blocker logs preserved at:
- `logs/host/t9-qwen25-05b-w32-bf16-convert.err.log`
- `logs/host/t10-qwen25-w32-mixed-bf16-v2-convert.err.log`
- `logs/host/t11-qwen25-w32-chunked-no-concat-bf16-convert.err.log`
- Reports: [reports/t9-qwen-bf16.md](../reports/t9-qwen-bf16.md),
  [reports/t10-qwen-mixed-bf16.md](../reports/t10-qwen-mixed-bf16.md),
  [reports/t11-qwen-chunked-bf16.md](../reports/t11-qwen-chunked-bf16.md)

### SmolLM2-1.7B — no NBG export

All windows (32/64/128/256) pass the ONNX builder gate (cosine ~1.0,
top-1 match) but fail NBG generation during `gen_nbg`:

```
ONNX external-data graph: ~6.85 GB
gen_nbg: segfault → 0-byte network_binary.nb
Board: never run (no valid NBG produced)
```

Vendor blocker logs preserved at the B1 host workspace.
Report: [reports/b1b-benchmark-matrix.md](../reports/b1b-benchmark-matrix.md).

### ACUITY hybrid quantization — table dump hang

Hybrid/w8a16 quantize consistently reaches `End quantization...`, truncates
the YAML `.quantize` table to 0 bytes, then hangs CPU-active. Blocks any
T5 recovery path that requires hybrid qtypes. Affects both SmolLM2 and Qwen.

Vendor blocker report: [reports/t6-vendor-acuity-hybrid-quantize-table.md](../reports/t6-vendor-acuity-hybrid-quantize-table.md).

### BF16 export — general

BF16 host inference works and preserves quality (cosine >0.99 on Qwen,
passing the host gate). But `vnn_VerifyGraph` status `-3 / 64768` blocks
NBG generation for both the full and chunked Qwen graphs and for
mixed-BF16 Qwen graphs that keep BF16 in transformer regions.

## Data type support summary

| Data type | Host import | Host quantize | Host inference | NBG export | Board run | LLM coherence |
|---|---|---|---|---|---|---|
| int16 DFP | Yes | Yes | Yes | Yes | Yes | OK for SmolLM2, fails for Qwen |
| pcq (int8 per-channel) | Yes | Yes (may hang) | Yes | Yes | Yes | No (incoherent) |
| uint8 uniform | Yes | Yes | Yes | Yes | Yes | Not tested on LLM (CNN only) |
| bf16 (bfloat16) | Yes | Yes | Yes | **No** (vnn_VerifyGraph -3) | — | Host: good; board: blocked |
| fp16 (float16) | Yes | Yes | Yes | Yes | Not tested | Host: fails quality (>0.99 threshold) |

## Practical ceiling

| What | Current limit | With vendor fix |
|---|---|---|
| Fast NPU chat | SmolLM2-135M, 21 tok/s, W=32 | — |
| Large NPU chat | SmolLM2-360M, 8 tok/s, W=32 | — |
| Context window | Fixed 32-64 tokens | Potentially thousands (KV-cache) |
| Qwen class on NPU | Blocked (all export paths) | BF16 export fix needed |
| 1.7B class on NPU | Blocked (gen_nbg segfault) | gen_nbg stability needed |
| CPU fallback quality | Qwen-0.5B Q8_0, real KV-cache | — |
| VLM on NPU | Proof of concept (tiny bridge) | Needs VLM-grade decoder + projector |
| Production hybrid | NPU vision + CPU LLM (works now) | — |

## Comparison to RK3588

The RK3588 (6 TOPS, RKLLM with native KV-cache + int4/int8) runs full
LLMs and VLMs like InternVL on the NPU. The A733 (3 TOPS nominal, no
KV-cache, no working LLM int4/int8 export, BF16 blocked by vendor) cannot
match this with NPU-only LLM inference.

**The realistic A733 equivalent is NPU-vision + CPU-LLM hybrid**: offload
vision encoding to the NPU (22.6 ms/frame for MobileCLIP-S0), run the
language model on CPU with a real KV-cache, and keep the NPU free for
robotics workloads during normal operation.
