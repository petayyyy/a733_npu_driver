# NPU Inference of LLMs/VLMs on the Allwinner A733 (Vivante VIP9000): Methodology Guide, Roadmap, and Agent Work Breakdown

## TL;DR
- **Getting a small CNN/vision model running on the A733 NPU is "almost certainly achievable" today** — the vendor VIPLite 2.0 / ACUITY (NBG) stack already runs ResNet50, YOLOv5/v8/v11 and YOLACT on the Vivante VIP9000 via `/dev/vipcore` on Radxa's Debian image; this is the realistic proof-of-principle target. **Running a full LLM/VLM *on the NPU* is "research risk," not a turnkey path:** there is no RKLLM-equivalent for Allwinner, the public ACUITY toolkit exposes only uint8/int16/bf16/pcq (int8 per-channel) quantization (no INT4), NBG graphs are static-shape (no dynamic KV-cache), and no one has demonstrated a transformer decoder on any VIP9000-class NPU via the public stack.
- **The hardware is severely memory-bound for decode.** The A733 has a 32-bit LPDDR5 bus; on the final-target Orange Pi Zero 3W (LPDDR5 @ 4800 MT/s ≈ 19.2 GB/s theoretical) a w4 0.5–1.5B model is ceiling-bound to roughly single-digit-to-low-double-digit tokens/s even in the best case, and the Radxa Cubie A7Z dev board's slower measured memory profile is worse. The 3 TOPS INT8 rating is not the bottleneck — bandwidth is.
- **Recommended strategy: a hybrid pipeline.** Put the VLM vision encoder (a CNN/ViT, static-shape) on the NPU via ACUITY/VIPLite — this is where the NPU genuinely helps — and run the LLM decoder on the A76 CPU cores with llama.cpp (q4_0/q4_K). Treat "LLM decode fully on NPU" as an R&D track via TVM-BYOC (VeriSilicon's `vsi_npu`) or etnaviv, not as the primary deliverable.

## Key Findings

### 1. Hardware and NPU-IP (mostly VERIFIED, performance ceilings are ASSUMPTION)
- **SoC:** Allwinner A733, 12 nm, 2× Cortex-A76 @ up to 2.0 GHz + 6× Cortex-A55 @ ~1.8 GHz, IMG BXM-4-64 MC1 GPU (OpenGL ES 3.2, Vulkan 1.3, OpenCL 3.0), RISC-V XuanTie E902 @ 200 MHz. (VERIFIED — Radxa/CNX/Olimex.)
- **NPU:** VeriSilicon Vivante VIP9000 (VIP V8 architecture), rated **3 TOPS INT8**, chip/optimize ID **0x1000003B**, single core (`device[0] core_count=1` in vpm_run logs), native INT8/INT16/FP16/BF16. INT4 is advertised at the IP level but **not** exposed in the public Allwinner toolkit. (VERIFIED — Radxa docs, Frigate writeup, vpm_run logs.)
- **Memory:** LPDDR4/4x/LPDDR5, **32-bit bus**. Radxa Cubie A7A units have measured LPDDR5 @ 1800 MT/s (~7.2 GB/s theoretical). Orange Pi Zero 3W uses LPDDR5 @ 4800 MT/s (~19.2 GB/s theoretical). On-chip SRAM: 192 KB + 512 KB shared. (VERIFIED — Radxa spec, Smart Home Circle review, Lunar Computer review.)
- **Memory-bound decode ceiling (ASSUMPTION / first-order calculation):** decode tok/s ≈ usable_BW ÷ bytes_read_per_token, at an assumed ~65% bandwidth efficiency:
  - **Orange Pi Zero 3W (~12.5 GB/s usable):** 0.5B w4 (~0.3 GB) ≈ 40 tok/s ceiling; 1.1–1.5B w4 (~0.8 GB) ≈ 15 tok/s; 3B w4 (~1.7 GB) ≈ 7 tok/s. At w8, roughly halve these.
  - **Radxa Cubie A7Z (~4.7 GB/s usable, slower measured memory):** roughly 0.4× the above.
  - These are *upper bounds*; measured CPU decode will be lower. KV-cache adds read traffic that grows with context (512 vs 2048 tokens). For calibration: on the 6 TOPS RK3588 with a *mature* NPU LLM stack, TinyLlama 1.1B (w8a8) sustains ~20.2 tok/s and Qwen2 0.5B ~42.6 tok/s — the A733 lacks both the bandwidth and the software to approach this.
- **RAM budget (ASSUMPTION):** weights dominate. 1.5B w4 ≈ 0.8 GB weights; KV-cache for a 1.5B-class model (~28 layers, GQA) at fp16 ≈ tens of MB at 512 ctx, ~100–200 MB at 2048 ctx. A 4 GB board comfortably holds a 1.5B w4 model + context; 3B needs a 6–8 GB SKU.

### 2. SDK and Toolchain (the main question) — VERIFIED
- **What is public:** Vivante **ACUITY Toolkit** (x86 Docker image `ubuntu-npu:v2.0.10`, pegasus scripts), **VIPLite 2.0** runtime (driver string `2.0.3.2-AW-2024-08-30`, confirmed on Cubie A7Z with kernel 5.15.147-12-a733), **awnn** C/C++ API, **vpm_run** test tool, OpenVX codegen, Vivante Unified Driver, plus VeriSilicon's open **TIM-VX** and the `vsi_npu` TVM fork. The Allwinner `ai-sdk` is mirrored at `ZIFENG278/ai-sdk` and `radxa-edge/ai-sdk`; build with `make AI_SDK_PLATFORM=a733 NPU_SW_VERSION=v2.0`.
- **Quantization:** the public ACUITY pipeline supports **uint8, int16, bf16, pcq (int8 per-channel), and mixed** — **no INT4** in the documented `pegasus_quantize.sh` flow. (VERIFIED.) VeriSilicon's "LLM on NPU" marketing (the June 2025 announcement of on-device LLMs with INT4/sparsity) refers to a **40+ TOPS-class part**, not the A733's 3 TOPS VIP9000; the original VIP9000 launch (2019) listed only INT8/INT16/FP16/BF16. INT4 is therefore not available to the user.
- **Transformer/LLM flow:** **geared toward CNNs.** NBG requires **fixed input dimensions** ("NPU inference only accepts fixed input dimensions"), which breaks autoregressive decode with a growing KV-cache. TIM-VX's op list includes Matmul, Softmax, LogSoftmax, FullyConnected, Gather, Moments, Gelu, Tanh, LSTM/GRU, EmbeddingLookup — but **no dedicated LayerNorm/RMSNorm, no RoPE, no fused attention, no KV-cache primitive**; these must be hand-composed. (VERIFIED via TIM-VX docs/Operators.md + targeted sub-investigation.)
- **Op support matrix (LLM/VLM needs vs public stack):**

| Op needed for LLM/VLM | Public A733 stack status |
|---|---|
| Conv2d / ViT patch-embed | ✅ Supported (CNN core strength) |
| MatMul / FullyConnected | ✅ In TIM-VX; ✅ NBG static |
| Softmax / LogSoftmax | ✅ Supported |
| GELU / Tanh / Sigmoid | ✅ Supported |
| LayerNorm | ⚠️ Compose from Moments + affine |
| RMSNorm | ❌ Not a primitive; compose |
| RoPE | ❌ Not present; compose or CPU |
| Scaled-dot-product attention (fused) | ❌ Not present; compose from matmul+softmax |
| KV-cache (dynamic seq len) | ❌ NBG is static-shape |
| INT4 weight quant | ❌ Not in public toolkit |
| Argmax / TopK / NMS post-proc | ❌ Falls back to CPU (Frigate finding) |

- **Where to get what's missing:** (a) **Request from Allwinner/Radxa/VeriSilicon** (see Vendor requests below): an ACUITY LLM/INT4 export plugin, VIPLite dynamic-shape/KV-cache support, a reference transformer NBG. (b) **Custom compile** via **TVM-BYOC `vsi_npu`** (registers VSI-NPU ops, generates NBG codegen + runtime; the fork exists at `VeriSilicon/tvm` branch `vsi_npu`/`upstream/vsi_npu`) or **MLIR-TIM-VX**. (c) **Reverse engineering** via **etnaviv/Teflon** in Mesa.

### 3. OS / Driver / Kernel — VERIFIED
- **NPU works on Debian (not just Tina).** On Radxa's Debian image the NPU node is **`/dev/vipcore`** (NOT `/dev/galcore` or `/dev/rknpu`); VIPLite userspace comes from the board's `npu-runtime` package (`libNBGlinker.so`, `libVIPhal.so`). Verify with `ls -l /dev/vipcore` and a `vpm_run` smoke test; the init log prints `VIPLite driver software version 2.0.3.2-AW-2024-08-30 ... cid=0x1000003b, device_count=1, device[0] core_count=1`. Independent confirmation: a community user ran **YOLOv8/v9/YOLO-NAS** object detection on the A733 NPU under Armbian (Debian 13, vendor kernel 6.6.98-vendor-sun60iw2) at **~25–60 ms/inference** via a custom ctypes plugin (Frigate discussion #23418), with DFL/box-decode + NMS forced onto CPU because "quantizing the post-processing (NMS/TopK/argmax) on the NPU loses accuracy / isn't supported."
- **Kernel coupling:** the NPU KMD and VIPLite blobs are tied to the Allwinner BSP (`sun60iw2` platform). Radxa's `linux-a733` packaging targets kernel **5.15**, and the SDK is documented against 5.15; community boards have also run vendor **6.6.98** kernels with VIPLite 2.0 working. The Orange Pi Zero 3W official image is **Debian 12 / kernel 6.1.31** (vendor fork). **No mainline A733 support exists** (absent from Linux 7.0); early SoC-level patches (CCU, RTC, pinctrl) are tracked on the linux-sunxi mainlining page but none are merged, so both boards depend on vendor kernels indefinitely.
- **Portability Radxa 5.1x/bullseye → Orange Pi 6.1.31/bookworm (ASSUMPTION, supported by evidence):** The **NPU user-space (ACUITY-produced NBG models, awnn/VIPLite `.so`, vpm_run) is kernel-agnostic at the binary level** as long as a compatible Vivante KMD + `/dev/vipcore` is present and the userspace glibc matches. **Transfers as-is:** NBG model files (compiled for cid 0x1000003B — identical silicon), ACUITY workflow, model architecture. **Must be rebuilt/re-sourced:** the **kernel NPU module** (must come from Orange Pi's 6.1.31 BSP), and any C++ awnn/llama.cpp app should be **recompiled against bookworm** (glibc 2.36) rather than reused from bullseye (glibc 2.31). The newer 6.1 kernel neither helps nor hurts NPU throughput (memory-bound); its value is driver hygiene, not speed.

### 4. Prior work and analogs — VERIFIED
- **No LLM has run on any Vivante VIP9000 NPU via the public stack** (A733, T527, i.MX 8M Plus, A311D). Every confirmed public-stack deployment is a CNN; even YOLO post-processing (NMS/argmax/TopK) falls back to CPU.
- **RKLLM (Rockchip) is the reference template but NOT portable as code.** RKLLM = conversion toolkit (HF→`.rkllm`; **w8a8 on RK3588, w4a16 on RK3576**) + runtime (`librkllmrt.so`) + NPU driver (`rknpu` ≥ v0.9.8). Measured rates: on RK3588 (6 TOPS) TinyLlama 1.1B (w8a8) ~20.2 tok/s and Qwen2 0.5B ~42.6 tok/s; on RK3576 (w4a16) Qwen2 0.5B ~34.2 tok/s, TinyLlama 1.1B ~21.3 tok/s, InternLM2 1.8B ~13.7 tok/s (w4a16 also shrinks TinyLlama from ~1.14 GB to ~645 MB). VLMs run as a `.rknn` vision encoder + `.rkllm` decoder (Qwen2-VL-2B, Qwen2.5-VL-3B, InternVL3.5-4B in Qengineering's demos). Community wrappers: **rkllama** (Ollama-style REST/OpenAI API). **What is portable to VIP9000:** the *architecture pattern* (offline AOT compile + lightweight runtime + split vision/LLM), NOT any binary. Allwinner has no rkllm-toolkit equivalent.
- **etnaviv (Mesa) reverse-engineering:** Tomeu Vizoso's open driver runs MobileNetV1 on Vivante NPUs (VIM3/VIPNano-QI, i.MX 8M Plus VIPNano-SI+) via the **Teflon** TFLite delegate — the NN job took 13 ms (~33 ms total), "around 3 times faster than running the same inference on the CPUs on the A311D SoC," later optimized to ~6.6 ms vs the proprietary driver's ~5.5 ms. **Limitations:** only convolution + tensor-add + a few ops are implemented; **no transformer ops, no LLM support**; VeriSilicon hardware docs are unavailable even to Allwinner (per Vizoso). etnaviv is a legitimate mainline path to *CNN* NPU offload but not a near-term LLM path.
- **Other Vivante platforms:** NXP i.MX 8M Plus uses the same VIP9000 family with NXP's VX Delegate (CNN only); NXP's LLM/4-bit support is on newer i.MX 9xx Neutron NPUs, not the Vivante part.

### 5. Models, Quantization, Compilation — recommendation
- **LLM shortlist (CPU decode via llama.cpp; NPU optional for prefill GEMMs):** Qwen2.5-0.5B-Instruct (best fit), SmolLM2-360M/1.7B, TinyLlama-1.1B, Qwen2.5-1.5B-Instruct; Phi-3-mini-3.8B as the upper bound (memory-heavy, slow). Rationale: w4 weights fit in ~0.3–2.2 GB; A76 cores handle decode at usable-if-modest rates.
- **VLM shortlist (hybrid):** SmolVLM-256M/500M, nanoLLaVA (~1B), Qwen2-VL-2B. Pattern: **vision encoder (SigLIP/ViT) → NPU (static shape, NBG)**; projector → NPU or CPU; **LLM decoder → CPU**. This is the most defensible use of the NPU.
- **CPU-fallback strategy:** run RoPE, RMSNorm, softmax, and KV-cache management on CPU (llama.cpp already does this efficiently); reserve NPU for the large static GEMMs (vision encoder; optionally prefill). Use **int16** quantization for NPU CNNs — Frigate showed uint8 dropped a real-scene detection from ~0.80 to ~0.50 confidence, while int16 keeps near-float accuracy.
- **GPU offload:** the IMG BXM-4-64 supports OpenCL 3.0 and llama.cpp has an OpenCL backend, but integrated-GPU bandwidth-sharing usually yields little over CPU on such SoCs — treat as experimental.

### 6. Runtime and Integration — recommendation
- **Engine options:** (a) vendor **awnn/VIPLite** + ACUITY NBG export — best for the vision encoder; (b) **TIM-VX-backed TFLite** delegate or **TVM-BYOC `vsi_npu`** — for custom op graphs; (c) **llama.cpp / MNN / ncnn** — CPU decode; (d) **etnaviv/Teflon** — mainline CNN offload.
- **E2E VLM pipeline:** image → preprocess (CPU/GPU) → ViT encoder NBG on NPU → projector → token embeddings → llama.cpp decoder on CPU → stream tokens.
- **Benchmark plan:** measure prefill tok/s, decode tok/s, first-token latency, peak RSS, watts (USB power meter), and accuracy vs a CPU-only baseline; record the NPU-vs-CPU split per pipeline stage.

## Details: Menu of Goals (achievability tiers)

| Tier | Goal | Evidence basis | Recommendation |
|---|---|---|---|
| **A — Almost certain** | Run a small CNN (ResNet50/MobileNet/YOLOv8n) on the NPU via VIPLite/awnn on Debian; verify `/dev/vipcore`, NBG inference | Vendor docs + Frigate community success (~25–60 ms) | **Do this first — proof of principle** |
| **B — High confidence** | NPU-accelerated VLM **vision encoder** (ViT/SigLIP, static shape) + CPU LLM decoder (llama.cpp) | ACUITY supports CNN/ViT; llama.cpp runs on A76 | **Primary deliverable** |
| **C — Plausible R&D** | Partial LLM **prefill** GEMMs on NPU via TVM-BYOC `vsi_npu`; decode on CPU | TVM `vsi_npu` fork exists; ops composable | Stretch goal |
| **D — Research risk** | Full LLM **decode** on NPU (dynamic KV-cache, RoPE, RMSNorm, INT4) | No precedent on VIP9000; NBG static-shape; no INT4 in toolkit | Only with vendor support or heavy RE |

## Phased Roadmap with Gates and Dependency Graph

**Dependency graph (linear with a fork at Phase 3):**
```
P0 (env) → P1 (NPU bring-up, CNN) → P2 (toolchain/ACUITY) → P3 ┬→ P3a (VLM hybrid, MAIN) → P4 (opt) → P5 (port) → P6 (bench)
                                                              └→ P3b (LLM-on-NPU R&D, optional, time-boxed)
```

- **Phase 0 — Environment (Radxa Cubie A7Z, in hand).** Flash Debian 11 rsdk-r6; confirm boot, A76 governors, thermals. **Gate G0:** board boots; `/proc/cpuinfo` shows 8 cores; thermals stable.
- **Phase 1 — NPU bring-up (CNN PoP).** Verify `/dev/vipcore`; build `ai-sdk` (`AI_SDK_PLATFORM=a733 NPU_SW_VERSION=v2.0`); run `vpm_run` + ResNet50/YOLOv8n NBG. **Gate G1:** NPU inference confirmed (driver banner `cid=0x1000003B`; correct top-5 / detections). *This is the user's #1 success criterion.*
- **Phase 2 — Toolchain.** Stand up ACUITY Docker (`ubuntu-npu:v2.0.10`) on x86; convert a custom ONNX CNN → NBG (int16, NPU_VERSION=v3); run on board. **Gate G2:** custom model converts + runs; accuracy within tolerance vs ONNX (use int16, not uint8).
- **Phase 3a — VLM hybrid (primary).** Export a small ViT vision encoder to NBG; run on NPU; wire to a llama.cpp CPU decoder (SmolVLM/nanoLLaVA-class). **Gate G3a:** end-to-end image→text answer; per-stage timings captured.
- **Phase 3b — LLM-on-NPU R&D (optional, time-boxed).** Attempt TVM-BYOC `vsi_npu` to offload prefill GEMMs of Qwen2.5-0.5B; decode on CPU. **Gate G3b:** any transformer op verified on NPU **OR** a documented blocker for vendor escalation.
- **Phase 4 — Optimization.** Quantization tuning (int16 vs pcq), SRAM caching, CPU thread pinning to A76, fixed-frequency governor. **Gate G4:** ≥20% latency improvement vs naive run.
- **Phase 5 — PORTING to Orange Pi Zero 3W (final target).** Reflash to Orange Pi official image 1.0.4 / kernel 6.1.31 / bookworm. Reuse NBG files as-is; obtain Orange Pi's NPU KMD + VIPLite from its BSP; **recompile** awnn/llama.cpp apps against bookworm glibc; confirm `/dev/vipcore` device-tree/overlay. **Gate G5:** identical NBG runs on Orange Pi NPU; outputs match Radxa within tolerance. *Transfers as-is:* NBG models, ACUITY workflow, architecture. *Rebuilds:* kernel NPU module (from OPi BSP), C++ binaries (bookworm), DT/overlay.
- **Phase 6 — Benchmark & docs.** Full benchmark matrix on both boards; publish methodology. **Gate G6:** reproducible numbers + writeup.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| NPU driver / `npu-runtime` absent or broken on a given image | Med | High | Prefer Radxa rsdk-r6 (known-good); local-build ai-sdk; vpm_run smoke test gate |
| LLM decode infeasible on NPU (static shape, no INT4) | High | High (only if goal D) | Pivot to hybrid (vision on NPU, decode on CPU); set goal B as deliverable |
| Memory bandwidth caps decode to low single digits | High | Med | Use 0.5–1.5B w4 models; manage expectations; pin to A76 |
| Orange Pi 6.1.31 NPU KMD incompatible with Radxa-sourced VIPLite | Med | Med | Source KMD + `.so` from the Orange Pi BSP specifically; recompile userspace on bookworm |
| ACUITY uint8 accuracy loss | Med | Med | Use int16 (Frigate-proven); calibrate with representative dataset |
| Vendor non-responsive to LLM feature requests | High | Low | Treat vendor path as bonus; rely on TVM-BYOC/etnaviv |
| Orange Pi images/links empty (Google Drive) | Med | Low | Use orangepi.org mirrors; keep Radxa as dev platform |

## Work Breakdown by Agents

**Final agent count: 9** — **5 research agents (R1–R5)** + **4 implementation agents (I1–I4)**.

### Research agents (turnkey, copy-paste prompts)

**R1 — NPU SDK & Toolchain Analyst**
> Mission: Produce a definitive capability report on the public Allwinner A733 / Vivante VIP9000 NPU toolchain (ACUITY, VIPLite 2.0, awnn, TIM-VX, TVM `vsi_npu`). Search: docs.radxa.com NPU pages, gitlab.com/tina5.0_aiot, GitHub `ZIFENG278/ai-sdk`, `radxa-edge/ai-sdk`, `VeriSilicon/TIM-VX` (docs/Operators.md), `VeriSilicon/tvm` (branch vsi_npu). Deliver: (a) exact quantization options in pegasus scripts; (b) op-support matrix vs LLM/VLM needs; (c) whether dynamic shapes / KV-cache are possible; (d) a step-by-step ACUITY ONNX→NBG recipe. Success: every claim cited to a primary URL + verbatim quote, each marked verified/assumption.

**R2 — Driver/Kernel/OS Portability Analyst**
> Mission: Map the NPU driver stack across Radxa rsdk (kernel 5.1x/bullseye) and Orange Pi Zero 3W (6.1.31/bookworm). Search: linux-sunxi A733 page, Radxa `linux-a733` packaging (DeepWiki), orangepi.org Zero 3W downloads, Armbian forum A733 threads. Deliver: a `/dev/vipcore` bring-up checklist; a table of what transfers as-is vs rebuilds across kernels; glibc considerations; device-tree/overlay notes. Success: a reproducible bring-up + porting checklist with sources.

**R3 — Analogs & Prior-Art Analyst (RKLLM/etnaviv)**
> Mission: Deep-dive RKLLM (toolkit/runtime/driver, w4a16/w8a8, measured tok/s, rkllama) and etnaviv NPU ML status; extract what is transferable to VIP9000. Search: airockchip/rknn-llm, docs.armsom.org, tomeuvizoso.net, Phoronix etnaviv, Qengineering VLM repos. Deliver: a side-by-side comparison + a "portability to VIP9000" verdict. Success: concrete tok/s figures + a transferability table.

**R4 — Hardware & Performance-Ceiling Analyst**
> Mission: Nail down the A733 memory subsystem and compute memory-bound decode ceilings for 0.5/1.1/1.5/3B at w4/w8, ctx 512/2048, for both boards. Search: A733 datasheet (DRAM section), Radxa/Orange Pi specs, board reviews for measured MT/s. Deliver: a tok/s + RAM table with stated assumptions and efficiency factor. Success: numbers reconcilable with RK3588/RK3576 baselines.

**R5 — Models & Quantization Analyst**
> Mission: Finalize the LLM/VLM shortlist with memory/op justification and a hybrid NPU+CPU op-placement plan. Search: HF model cards (Qwen2.5-0.5B/1.5B, SmolLM2, TinyLlama, SmolVLM, nanoLLaVA), llama.cpp quant docs. Deliver: a ranked model table (params, w4 size, KV size @512/2048, NPU-able layers) + a placement plan. Success: every model fits a stated RAM budget.

### Implementation agents (executor-style, Claude Code/CLI)

**I1 — NPU Bring-up Engineer (Radxa Cubie A7Z)**
> Mission: Achieve first NPU inference. Verify `/dev/vipcore`; clone+build `ai-sdk` with `make AI_SDK_PLATFORM=a733 NPU_SW_VERSION=v2.0`; set `LD_LIBRARY_PATH` to `viplite-tina/lib/aarch64-none-linux-gnu/v2.0`; run `vpm_run` and ResNet50/YOLOv8n NBG demos. Deliver: terminal logs showing the NPU banner + correct outputs. Success = Gate G1.

**I2 — ACUITY Conversion Engineer (x86 host)**
> Mission: Stand up ACUITY Docker `ubuntu-npu:v2.0.10`; convert a custom ONNX CNN and a small ViT vision encoder to NBG (int16, NPU_VERSION=v3); validate via pegasus_inference + on-board vpm_run. Deliver: reproducible conversion scripts + an accuracy report. Success = Gate G2.

**I3 — Hybrid VLM/LLM Integration Engineer**
> Mission: Build the e2e hybrid pipeline — ViT encoder NBG on NPU + llama.cpp CPU decoder (start with Qwen2.5-0.5B / SmolVLM). Build llama.cpp for aarch64 (optionally OpenCL on BXM); wire vision features to the decoder; pin to A76. Deliver: a working image→text demo + per-stage timings. Success = Gate G3a.

**I4 — Porting & Benchmark Engineer (Orange Pi Zero 3W)**
> Mission: Port to Orange Pi official image 1.0.4 / kernel 6.1.31 / bookworm; reuse NBG models; source NPU KMD + VIPLite from the OPi BSP; recompile userspace; run the full benchmark matrix (prefill/decode tok/s, latency, RAM, watts, accuracy) on both boards. Deliver: a comparison report. Success = Gates G5 + G6.

## Open Questions and Vendor Requests
**Open questions:** (1) Exact VIP9000 sub-variant, MAC count, and NPU clock in the A733 (not in public docs; the "NPU Version Comparison Table" did not yield MAC/frequency). (2) Whether VIPLite 2.0 supports any dynamic-shape execution. (3) Whether Orange Pi's 6.1.31 BSP ships a compatible `/dev/vipcore` KMD + `npu-runtime` package. (4) Real measured LPDDR5 MT/s on shipping Orange Pi Zero 3W units.

**Concrete requests to Allwinner/Radxa/VeriSilicon:** (a) an ACUITY LLM/transformer export path + **INT4 (w4a16)** quantization plugin for the A733 VIP9000; (b) VIPLite dynamic-shape / KV-cache reference for autoregressive decode; (c) a reference transformer NBG (e.g., a tiny GPT block) demonstrating attention on VIP9000; (d) the "NPU Version Comparison Table" with MAC/frequency for A733 v3; (e) confirmation that the `npu-runtime` package is available on the Orange Pi Zero 3W BSP.

## Caveats
- **3 TOPS is small and the bus is narrow (32-bit).** Decode is memory-bound; do not expect RK3588-class tok/s. Figures marked "ASSUMPTION" are first-order ceilings, not measured results.
- **There is no RKLLM analog for Allwinner.** Any "LLM on the NPU" claim today is research, not product.
- **The public ACUITY toolkit is CNN-oriented:** no INT4, static shapes, no transformer-specific ops — verified.
- **Both boards depend on vendor kernels** (no mainline A733). Driver portability is plausible but must be validated empirically on the Orange Pi 6.1.31 BSP.
- **Vendor "Llama 2 / LLM on VIP9000" marketing** refers to the IP's theoretical capability with VeriSilicon's full (non-public) stack — and the headline 40+ TOPS LLM announcement is a different, much larger NPU class — not the Allwinner public release on the 3 TOPS A733.