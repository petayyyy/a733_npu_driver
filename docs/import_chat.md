# A733 NPU LLM/VLM — Project Handoff / Knowledge Base

> Paste this into a new Claude or Claude Code session to continue seamlessly.
> This is the distilled, verified final state of the project.
> Repo: github.com/petayyyy/a733_npu_driver

## 0. Your role (read first)
You are advising on running LLM/VLM inference on the NPU of an Allwinner A733 board.
The user runs implementation via Codex in the repo above. The repo is the shared memory:
each task commits a report under reports/ and updates reports/status.md.
Always reason from the repo + this doc.

## 1. Goal & final verdict
Goal: run a real LLM/VLM with model compute on the A733 NPU (NPU-only).
CPU allowed only for orchestration (tokenize, sample/argmax, loop, I/O).

**VERDICT (2026-06-25): Core goal ACHIEVED as a working proof-of-concept.**
SmolLM2-135M/360M run NPU-only and coherently on the Orange Pi Zero 3W
(135M/W32 = 20.7 tok/s). Interactive chat shell works. MobileCLIP-S0 vision
encoder works on NPU (22.6 ms/frame). Qwen2.5-0.5B and SmolLM2-1.7B are
vendor-gated — every config was tried and failed.

**Recommended production path: HYBRID.** NPU for vision + small LLMs; CPU
(llama.cpp) for Qwen-class models and VLM (SmolVLM-256M, 52.6 tok/s).

## 2. Hardware / environment (verified)
- SoC: Allwinner A733. NPU: Vivante VIP9000, cid `0x1000003b`, single core, ~1.0 GHz,
  ~3 TOPS INT8. Native INT8/INT16/FP16/BF16. 32-bit LPDDR5.
- Final board: Orange Pi Zero 3W (A733), 6 GB LPDDR5, kernel 6.6.98-sun60iw2.
  SSH orangepi@192.168.31.225. /dev/vipcore present as major:minor 199,0.
- Dev board: Radxa Cubie A7Z, Debian 11, kernel 5.15.147, ~1 GB RAM.
- Toolchain: ACUITY 6.30.22 (Docker ubuntu-npu:v2.0.10.1), Vivante IDE 5.11.0,
  VIPLite 2.0.3.2-AW-2024-08-30.
  **"int16" = dynamic fixed point (single power-of-2 scale per tensor), NOT IEEE fp16.**

## 3. What WORKS (verified on hardware)

### LLM on NPU (int16, fixed-window):
| Model | W | Coherent | Decode tok/s | First-token | RSS | NBG size |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 32 | yes | 20.7 | 48 ms | 272 MB | 281 MB |
| SmolLM2-135M | 64 | weak | 14.0 | 72 ms | 274 MB | 282 MB |
| SmolLM2-360M | 32 | yes | 8.4 | 114 ms | 646 MB | 673 MB |
| SmolLM2-360M | 64 | yes | 4.9 | 212 ms | 649 MB | 675 MB |
Coherence CLIFF: W≥128 incoherent. No KV-cache → fixed window only.

### LLM on CPU (llama.cpp, real KV-cache):
| Model | Quant | Context | Decode tok/s | CPU usage | Cores free |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Q8_0 | 2k-8k | 18.0 | 2×A76 = 25% | 6 A55 |
| Qwen2.5-0.5B | Q8_0 | 16k | 2.2 (real) | 2×A76 | 6 A55 |

### VLM on NPU:
- **MobileCLIP-S0**: 1×3×256×256 → 1×512, 22.6 ms, cosine 0.99996. Works.
- **Tiny VLM bridge** (PoC only): vocab=16, proves the data path.

### VLM on CPU:
| Model | Quant | Decode tok/s | RSS | Accuracy |
|---|---|---|---|---|
| SmolVLM-256M | Q8_0 | 52.6 | 634 MB | Accurate (read moon-landing newspaper) |
| SmolVLM-500M | Q8_0 | 22.3 | ~1.2 GB | Accurate (more detail) |

## 4. What is BLOCKED / VENDOR-GATED (verified)

### Qwen2.5-0.5B on NPU — every config fails:
- **int16 DFP**: cosine 0.236 (quality fail)
- **FP16**: cosine 0.541 (quality fail)
- **BF16**: cosine 0.991 (quality OK) but `vnn_VerifyGraph -3 / 64768` (export fail)
- **W8A16**: cosine 0.079 (no SmoothQuant); smoothed blocked by quantize-table bug
- **Per-channel int16**: pegasus rejects (only INT8/INT4 for `perchannel_symmetric_affine`)
- **Block-chain 26 NBG**: loads and runs at 6.6 tok/s but produces garbage tokens

### Other:
- **SmolLM2-1.7B**: gen_nbg segfault, 0-byte NBG
- **SmolVLM SigLIP on NPU**: ACUITY Conv shape crash
- **No KV-cache**: static-shape NBG, fixed window only
- **No working LLM int8**: pcq exports but incoherent; per-channel int16 not in ACUITY

## 5. Key hardware facts (verified)
- NPU clock ~1.0 GHz; int16 ~1.45× slower + ~2× working memory vs uint8
- NBG load ~1.35 ms/MB; effective NPU decode BW ~6 GB/s
- ACUITY tensor dim limit 65536 (Qwen vocab 151936 exceeds it, but not the only BF16 blocker)
- Legal dtype boundaries: MATMUL BF16→BF16 only; SLICE no BF16→INT16;
  DATACONVERT BF16→F16→INT16 is legal but ACUITY doesn't emit standalone DataConvert
- NBG files are binary-compatible across A733 boards (same cid)
- Runner must be rebuilt per-board (glibc-matched VIPLite .so)

## 6. Verified building blocks (reuse, don't rebuild)
- `make_real_llm_onnx.py`: fixed-window ONNX from HF weights (SmolLM2, Qwen)
- `convert_onnx_to_nbg.sh`: ONNX → NBG via ACUITY (uint8 | int16 | pcq | bf16 | fp16)
- Host oracle tooling: `compare_onnxruntime_to_oracle.py`, `compare_acuity_host_to_oracle.py`
- `npu_lm_runner.c`: persistent VIPLite runner, load-once, stdio protocol
- `chat_shell.py`: interactive REPL with streaming tokens, window counter
- Vendor blocker reports: t6, t9, t10, t11, t7, t8, q1, q2, v2
- Vendor ticket summary: [docs/vendor-tickets.md](vendor-tickets.md)

## 7. WHY the limits exist
- **Static-shape → no KV-cache → short fixed window (W≤64), O(W²) throughput**
- **Memory-bound decode** on 32-bit LPDDR5; NPU reads ~6 GB/s from NBG
- **Qwen activation outliers** (act_absmax ~1790) exceed int16/FP16 range; BF16 fixes quality but won't export
- **ACUITY quantize-table bug** blocks W8A16/SmoothQuant recovery
- **Per-channel int16 not in ACUITY** — only INT8/INT4

## 8. Recommended production path
**Hybrid NPU-vision + CPU-LLM:**
- MobileCLIP-S0 on NPU (22.6 ms/frame) for vision
- Qwen2.5-0.5B Q8_0 on CPU (18 tok/s, 2×A76, 6×A55 free) for chat with real context
- SmolVLM-256M Q8_0 on CPU (52.6 tok/s, 634 MB RSS) for image chat
- SmolLM2-135M on NPU (21 tok/s) for fast short responses when CPU is busy

## 9. Open items
1. **Vendor tickets**: file the 6 consolidated blocker packets (vendor-tickets.md)
2. **V2 retry**: try direct SigLIP export (bypass Idefics3) — may unlock SmolVLM vision on NPU
3. **If vendor fixes BF16 export**: Qwen full model on NPU
4. **If vendor provides KV-cache runtime**: thousands of tokens of coherent NPU context

## 10. How to work going forward
- One task = one focused session; each commits a reports/<id>.md
- Mark every claim "verified" or "assumption"
- Gate NPU coherence on ACTUAL on-board generation, not pegasus-inference cosine
- Only one NPU consumer on /dev/vipcore at a time
- CPU (llama.cpp) can run alongside NPU workloads

## 11. Suggested next steps
1. File the vendor tickets (they're ready in vendor-tickets.md)
2. Try V2 retry (direct SigLIP export for SmolVLM vision on NPU)
3. For picoclaw + local assistant: deploy the hybrid config (MobileCLIP on NPU + SmolVLM/Qwen on CPU)
