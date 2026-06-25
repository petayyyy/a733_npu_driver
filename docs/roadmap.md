# A733 VIP9000 Execution Roadmap

Research phase: **COMPLETE** (2026-06-25). The project achieved its core goal:
real LLM/VLM layer compute runs NPU-only on the A733, with verified tooling,
benchmarks, and the hybrid CPU-NPU path recommended for production.

## Gates Completed

| Gate | Date | Status | Summary |
|---|---|---|---|
| G0 | 2026-06-20 | Passed | Board boots, 8 cores, thermals OK |
| G1 | 2026-06-20 | Passed | `/dev/vipcore`, VIPLite 2.0.3.2, YOLOv8n, cid `0x1000003b` |
| G2 | 2026-06-20 | Passed | ACUITY int16/uint8 ONNX→NBG for LeNet, Inception v1 |
| G3a | 2026-06-22 | Passed | Transformer decoder block, tiny LM, VLM bridge, decode loop on NPU |
| G5 | 2026-06-24 | Passed | Orange Pi port: `/dev/vipcore`, NBG binary-compatible, runner rebuilt |
| G6 | 2026-06-25 | Passed | Full benchmark matrix (SmolLM2-135M/360M at W=32/64/128/256), CPU baselines (Qwen, SmolVLM), VLM benchmarks, CPU utilization sweep |
| G7 | 2026-06-25 | Passed | Hybrid VLM: NPU SmolVLM vision + CPU LLM, accurate on 3 test images, 2 A76 cores freed |

### Detailed gate log

See [reports/status.md](../reports/status.md) for the full chronological log.

## What Works (final state)

- **SmolLM2-135M int16 W=32**: 20.7 tok/s, coherent, NPU-only
- **SmolLM2-360M int16 W=32**: 8.4 tok/s, coherent, NPU-only
- **MobileCLIP-S0 vision**: 22.6 ms/frame, cosine 0.99996, NPU-only
- **Interactive chat shell** (B2): streaming tokens, window counter, on Orange Pi
- **Qwen2.5-0.5B Q8_0 CPU**: 18.0 tok/s, 2×A76 (25% CPU), 6 cores free
- **SmolVLM-256M Q8_0 CPU**: 52.6 tok/s, 634 MB RSS, accurate image chat

## What's Vendor-Gated / Blocked

- Qwen2.5-0.5B NPU: every monolithic and block-chain config fails (see blockers.md)
- SmolLM2-1.7B NPU: gen_nbg segfault (0-byte NBG)
- **SmolVLM SigLIP NPU: RESOLVED (V2d)** — Conv→MatMul rewrite + NPU→llama.cpp injection, accurate on 3 test images
- No KV-cache: static-shape NBG, fixed window only

## Remaining Open Items

1. **V2 retry** — Try direct SigLIP export (bypassing Idefics3 wrapper) to
   see if ACUITY handles pure SigLIP Conv. Could unlock SmolVLM vision on NPU.
2. **Vendor tickets** — File the consolidated blocker packets (see
   [vendor-tickets.md](vendor-tickets.md)) with Radxa/Allwinner/VeriSilicon.
   The T6/T9/T10/T11/Q2/V2 reports contain exact reproducer commands.
3. **What each would unlock**:
   - **BF16 export fix (T9)**: Qwen2.5-0.5B runs monolithic NPU-only at
     host-quality cosine 0.991. This is the single biggest unlock.
   - **gen_nbg stability (B1)**: SmolLM2-1.7B on NPU.
   - **Conv shape fix (V2)**: SmolVLM vision encoder on NPU.
   - **KV-cache runtime**: Thousands of tokens of coherent context on NPU
     instead of W≤64 fixed windows.

## Recommended Architecture (production)

**Hybrid: NPU vision + CPU LLM.** Vision encoding on NPU (MobileCLIP-S0,
22.6 ms/frame), language model on CPU (Qwen/SmolVLM via llama.cpp with real
KV-cache). SmolLM2-135M on NPU for fast short responses when CPU cores are
needed elsewhere.

This is the practical ceiling for the current ACUITY 6.30.22 / VIPLite 2.0.3.2
toolchain on this silicon. Without vendor intervention on the BF16 exporter or
a KV-cache runtime, the A733 VIP9000 cannot match the RK3588's LLM/VLM NPU
capabilities.
