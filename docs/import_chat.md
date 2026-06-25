# A733 NPU LLM/VLM — Project Handoff / Knowledge Base

> Paste or import this into a new Claude / Claude Code session to continue the
> A733 NPU LLM/VLM project with full context. This is the final, verified state.
> Repository: github.com/petayyyy/a733_npu_driver

## 0. Your role (read first)

You are advising on running LLM/VLM inference on the NPU of an Allwinner A733
board. The repo is shared memory: each task commits a report under `reports/`.
Always reason from this document + the repo, not from scratch.

## 1. Goal & final verdict

**Goal:** run LLM/VLM with model compute on the A733 NPU.

**VERDICT (2026-06-25): Core goal ACHIEVED as a working proof-of-concept + full
reproducible toolchain.** SmolLM2-135M/360M run NPU-only and coherently
(135M=20.7 tok/s). Interactive chat shell works. MobileCLIP-S0 vision encoder
works on NPU (22.6 ms, cosine 0.99996). **SmolVLM-256M hybrid VLM (NPU vision +
CPU LLM) is a proven deliverable** (V2d, e2e accurate on 3 images, vision 5.94s
on NPU freeing both A76, LLM ~46.5 tok/s). Qwen2.5-0.5B on NPU is vendor-gated
— every config was tried and failed.

**Recommended production path: HYBRID.** NPU for vision + small LLMs; CPU
(llama.cpp) for Qwen-class models and full VLM. This is the path for a real
local assistant on A733 hardware.

## 2. Hardware / environment (verified)

- **SoC:** Allwinner A733
- **NPU:** Vivante VIP9000, cid `0x1000003b`, single core, ~1.0 GHz, ~3 TOPS INT8
- **Native dtypes:** INT8, INT16, FP16, BF16 (BF16 host-only; export blocked)
- **"int16" = dynamic fixed point** (single power-of-2 scale per tensor), NOT IEEE fp16
- **DRAM:** 32-bit LPDDR5, NPU decode BW ~6 GB/s effective
- **Final board:** Orange Pi Zero 3W, 6 GB LPDDR5, kernel 6.6.98-sun60iw2,
  SSH orangepi@192.168.31.225, `/dev/vipcore` 199:0
- **Dev board:** Radxa Cubie A7Z, Debian 11, kernel 5.15.147, ~1 GB RAM
- **Toolchain:** ACUITY 6.30.22 (Docker `ubuntu-npu:v2.0.10.1`), Vivante IDE 5.11.0,
  VIPLite 2.0.3.2-AW-2024-08-30
- **ACUITY tensor dim limit:** 65536 per dimension (Qwen vocab 151936 exceeds it)
- **int16 vs uint8:** ~1.45× slower, ~2× working memory
- **NBG load:** ~1.35 ms/MB
- **NBG files** are binary-compatible across A733 boards (same cid `0x1000003b`)

## 3. What WORKS (all verified on Orange Pi Zero 3W)

### 3a. LLM on NPU (int16, fixed-window, NPU-only)

| Model | W | Coherent | Decode tok/s | First-token ms | RSS MiB | NBG MB |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 32 | yes | 20.7 | 48 | 272 | 281 |
| SmolLM2-135M | 64 | weak | 14.0 | 72 | 274 | 282 |
| SmolLM2-135M | 128 | no | 6.0 | 166 | 282 | 287 |
| SmolLM2-360M | 32 | yes | 8.4 | 114 | 646 | 673 |
| SmolLM2-360M | 64 | yes | 4.9 | 212 | 649 | 675 |
| SmolLM2-360M | 128 | no | 2.0 | 502 | 681 | 693 |

**Coherence CLIFF at W≥128.** No KV-cache → full recompute per token → O(W²)
throughput. Sweet spot: W=32–64.

### 3b. LLM on CPU (llama.cpp, real KV-cache, 2×A76 `taskset -c 6,7`)

| Model | Quant | Context | Decode tok/s | Peak RSS | Notes |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Q8_0 | 2k–8k | 18.0 | 1,109 MB | Fast; 6 A55 cores free for ROS2 |
| Qwen2.5-0.5B | Q8_0 | 16k | 2.2 (real chat) | 1,306 MB | ~18 min first-token |
| Qwen2.5-1.5B | Q4_K_M | 8k | 8.5 | 2,041 MB | **Recommended default** for LLM chat |
| Qwen2.5-3B | Q4_K_M | 2k | 4.0 | 3,841 MB | **Experimental, batch-only** — not for interactive chat |
| Qwen2.5-7B | Q2_K | — | 0.05 | 2,839 MB | Impractical |

**Q8_0 beats Q4_K_M** for 0.5B on this board. 2×A76 is optimal — extra cores
hurt (memory-bound). A76 decode is 2.6× faster than A55.

### 3c. Vision on NPU

| Component | Input | Output | Latency | RSS | NBG | Cosine |
|---|---|---|---|---|---|---|
| MobileCLIP-S0 | 1×3×256×256 | 1×512 emb | 22.6 ms | 14 MB | 19 MB | 0.99996 |
| SmolVLM SigLIP (V2d) | 1×3×512×512 int16 | 1×64×576 f32 | 5,958 ms | 21 MB pool | 271 MB | 0.9972 vs ONNX |

SmolVLM SigLIP was originally blocked (ACUITY Conv shape crash). **Resolved in
V2d via Conv→Reshape+MatMul rewrite.** NPU vision frees both A76 cores for ROS2.

### 3d. VLM on CPU (llama.cpp)

| Model | Quant | Decode tok/s | Peak RSS | Accuracy |
|---|---|---|---|---|
| SmolVLM-256M-Instruct | Q8_0 | 52.6 | 634 MB | Accurate (verified on 3 images) |
| SmolVLM-500M-Instruct | Q8_0 | 22.3 | ~1.2 GB | Accurate (more detail) |

SmolVLM-256M is the recommended VLM: fast, accurate, 634 MB RSS leaves >5 GB for
ROS2/picoclaw.

### 3e. Hybrid VLM: NPU vision + CPU LLM (V2d, proven deliverable)

| Metric | CPU-only VLM | Hybrid (V2d) |
|---|---|---|
| Vision latency | ~1–2s (estimated) | 5.94s (measured, **0 CPU**) |
| A76 cores for vision | 2 (fully loaded) | **0** (NPU only) |
| A76 free for ROS2 | 0 | **2** |
| LLM decode tok/s | 52.6 | 46.5 |
| Answer accuracy | accurate | **accurate (verified)** |

E2E validated on 3 V1 test images (dog, cat, moon-landing newspaper). All
answers match CPU-only quality. Vision runs on NPU; LLM runs on CPU via
`A733_NPU_EMBEDDINGS` env var injection (V2c mtmd patch).

## 4. What is BLOCKED / VENDOR-GATED (verified, do NOT retry)

### 4a. Qwen2.5-0.5B on NPU — every config fails

| Config | Host cosine | Export? | Verdict |
|---|---|---|---|
| int16 DFP | 0.236 | Yes | Incoherent |
| FP16 | 0.541 | Yes | Incoherent |
| BF16 | **0.991** | **No** | `vnn_VerifyGraph -3 / 64768` |
| W8A16 (no smoothing) | 0.079 | Yes | Incoherent |
| W8A16 + SmoothQuant | — | Blocked | ACUITY quantize-table bug (t6) |
| Per-channel int16 | — | Blocked | Pegasus rejects (only INT8/INT4) |
| Block-chain 26 NBG (int16) | 0.975 (host) | Yes | **Garbage on hardware** (6.6 tok/s, depth collapse) |

**Root cause:** Qwen's activation outliers (act_absmax ~1790, RMS² ~2.65e6)
exceed int16/FP16 range. BF16 fixes host quality (cosine 0.991) but won't
compile — legal dtype boundaries (TIM-VX) require DataConvert nodes that ACUITY
doesn't insert. 65536 dimension limit on vocab (151936) exacerbates but is not
the only barrier.

**Verdict: Qwen on NPU is genuinely vendor-gated. Do not re-attempt without a
new toolchain build or vendor fix.**

### 4b. Other blockers

| Blocker | Detail |
|---|---|
| SmolLM2-1.7B NBG export | `gen_nbg` segfault, 0-byte NBG from 6.85 GB ONNX |
| >1B VLMs | InternVL3.5-1B loads but OOM/starvation on 6 GB |
| No KV-cache | Static-shape NBG, fixed window only |
| No working LLM INT4/INT8 | PCQ exports but incoherent; per-channel int16 not in ACUITY |
| ACUITY hybrid quantize-table | Truncates to 0 bytes, hangs (t6) |
| ACUITY host int16 cosine | Gave false negatives — use on-board generation for gates |

**Confirmed dead ends:** BF16 NBG export, per-channel int16, W8A16 SmoothQuant,
TVM/TIM-VX hand-built (lacks RMSNorm/Gather/Slice/SwiGLU), 32k Qwen CPU context
(~69 min first-token).

## 5. Key building blocks in the repo (reuse, don't rebuild)

| File | Purpose |
|---|---|
| `scripts/host/make_real_llm_onnx.py` | Fixed-window ONNX from HF weights (SmolLM2, Qwen) |
| `scripts/host/convert_onnx_to_nbg.sh` | ONNX → NBG via ACUITY (uint8\|int16\|pcq\|bf16\|fp16) |
| `scripts/host/compare_*.py` | Host oracle tooling (ORT, ACUITY host vs FP reference) |
| `scripts/board/npu_lm_runner.c` | Persistent VIPLite runner, load-once, stdio protocol |
| `scripts/board/chat_shell.py` | Interactive REPL with streaming tokens |
| `app/vlm_chat.py` | CLI VLM image chat (CPU or NPU+CPU hybrid) |
| `app/llm_chat.py` | CLI LLM text chat (Qwen 0.5B/1.5B/3B, CPU) |
| `docs/09-cli-tools.md` | Install guide for CLI tools on Orange Pi Zero 3W |
| `docs/blockers.md` | Exhaustive blocker list with reports |
| `docs/configurations.md` | All verified configs at a glance |
| `docs/RESULTS.md` | Every number with source |
| `docs/vendor-tickets.md` | 6 consolidated blocker packets ready to file |

## 6. Why the limits exist (so you reason correctly, don't re-litigate)

- **Static-shape NBG → no KV-cache.** Every token recomputes the full window.
  Coherence holds at W≤64, breaks at W≥128. TIM-VX has VARIABLE tensor but no
  working KV-cache precedent on VIP9000.
- **Memory-bound decode.** 32-bit LPDDR5, NPU reads ~6 GB/s from NBG. Throughput
  is dominated by weight-bandwidth, not clock. More compute cores won't help.
- **Qwen activation outliers exceed int16/FP16 range.** BF16 is the only format
  that preserves quality (>0.99 cosine), but ACUITY won't emit the DataConvert
  nodes required by TIM-VX dtype boundary rules. This is a toolchain gap, not an
  NPU hardware limit.
- **int16 DFP collapses over depth.** Even in block-chain mode (26 NBGs,
  per-block export succeeds), 24 chained quantize-dequantize cycles compound
  error beyond host simulation, producing degenerate tokens on hardware.
- **No working LLM INT4/INT8 path.** PCQ (per-channel int8) exports but is
  incoherent. Per-channel int16 is rejected by pegasus. W8A16 needs SmoothQuant
  which is blocked by the ACUITY quantize-table bug (t6).

## 7. Open items

1. **File vendor tickets** — 6 consolidated blocker packets ready in
   `docs/vendor-tickets.md`. This is the only remaining action item.
2. **If vendor fixes BF16 export:** Qwen full model on NPU becomes feasible.
3. **If vendor provides KV-cache runtime:** thousands of tokens of coherent
   NPU context become possible.

The project is otherwise complete. All exploration paths are exhausted; all
working configurations are documented; all blockers are characterized with
reproducers. The CLI tools (`app/vlm_chat.py`, `app/llm_chat.py`) are the
recommended user-facing interface.

## 8. How to work going forward

- One task = one focused session; each commits a `reports/<id>.md`
- Mark every claim **[verified]** or **[estimate]**
- Gate NPU coherence on ACTUAL on-board generation, not pegasus-inference cosine
  (host int16 cosine gave false negatives)
- Only one NPU consumer on `/dev/vipcore` at a time
- CPU (llama.cpp) can run alongside NPU workloads
- Runner must be rebuilt per-board (glibc-matched VIPLite .so)
- Use the CLI tools as the recommended user interface; extend `app/` for new use

## 9. Pointers

| Document | Purpose |
|---|---|
| [README.md](../README.md) | Top-level overview, quick start |
| [docs/RESULTS.md](RESULTS.md) | All numbers with source reports |
| [docs/blockers.md](blockers.md) | Exhaustive blocker list |
| [docs/configurations.md](configurations.md) | All verified configs at a glance |
| [docs/vendor-tickets.md](vendor-tickets.md) | 6 ready-to-file vendor tickets |
| [docs/09-cli-tools.md](09-cli-tools.md) | Install & run CLI tools on board |
| [app/README.md](../app/README.md) | CLI tools reference |
| [reports/b1b-benchmark-matrix.md](../reports/b1b-benchmark-matrix.md) | SmolLM2 NPU benchmarks |
| [reports/b4b-cpu-utilization.md](../reports/b4b-cpu-utilization.md) | Qwen CPU utilization sweep |
| [reports/b5-qwen-size-sweep.md](../reports/b5-qwen-size-sweep.md) | Qwen 0.5B–7B CPU benchmarks |
| [reports/v2d-vlm-npu-mmproj-glue.md](../reports/v2d-vlm-npu-mmproj-glue.md) | Hybrid VLM final deliverable |
| [reports/q2-qwen-block-nbg.md](../reports/q2-qwen-block-nbg.md) | Qwen block-chain failure |
| [reports/q1-qwen-int8-gates.md](../reports/q1-qwen-int8-gates.md) | Qwen W8A16 + per-channel gates |

---

*Numbers sourced from Orange Pi Zero 3W (6 GB, kernel 6.6.98, VIPLite 2.0.3.2).
Cross-reference individual reports for full methodology, logs, and per-step
comparisons. All claims tagged [verified] are measured on hardware.*
