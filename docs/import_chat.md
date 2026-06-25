# A733 NPU LLM/VLM — Project Handoff / Knowledge Base

> Paste this into a new Claude or Claude Code session to continue seamlessly.
> It is the distilled, verified state of the project. Wrong turns are recorded ONLY as
> "tried, verified, didn't work" so they aren't repeated. Repo: github.com/petayyyy/a733_npu_driver

## 0. Your role (read first)
You are advising on running LLM/VLM inference on the NPU of an Allwinner A733 board. The user
runs the actual implementation via Codex (GPT-5.5) in the repo above; you analyze results, catch
mistakes, and write copy-paste task prompts. The user is in Russian; technical prompts for Codex
should be in English. The repo is the shared memory: each task commits a report under reports/
and updates reports/status.md. Always reason from the repo + this doc.

## 1. Goal & honest current verdict
Goal: run a real LLM (and a VLM vision pipeline) with model compute on the A733 NPU (NPU-only),
on the final board. CPU allowed only for orchestration (tokenize, sample/argmax, loop, I/O).

VERDICT: Core goal ACHIEVED as a working proof-of-concept. A real LLM (SmolLM2-135M/360M) runs
NPU-only and coherently on the final Orange Pi Zero 3W, plus an interactive chat shell, a VLM
vision pipeline on NPU, and a CPU fallback. It is a reproducible PoC + toolchain, NOT a polished
product. The bigger/better models (Qwen2.5-0.5B, SmolLM2-1.7B) do NOT compile to NBG — vendor-
gated. For a genuinely usable assistant the recommended direction is HYBRID: NPU for vision,
CPU (llama.cpp) for the LLM.

## 2. Hardware / environment (verified)
- SoC: Allwinner A733. NPU: VeriSilicon Vivante VIP9000, cid=0x1000003b, single core, ~1.0 GHz,
  ~3 TOPS INT8, native INT8/INT16/FP16/BF16. 32-bit LPDDR5 bus. Effective NPU decode bandwidth
  measured ~6 GB/s (decode is memory-bound, NOT compute-bound).
- Dev board: Radxa Cubie A7Z, Debian 11, kernel 5.15.147, ~1 GB RAM. (Used for early dev.)
- FINAL board: Orange Pi Zero 3W (A733, NOT the H618 Zero 3), 6 GB LPDDR5, official image 1.0.4,
  kernel 6.6.98-sun60iw2 (newer than expected; NPU works on it). SSH orangepi@192.168.31.225.
- Toolchain (public, no NDA): VeriSilicon ACUITY 6.30.22 (Docker ubuntu-npu:v2.0.10.1),
  Vivante IDE 5.11.0, runtime VIPLite 2.0.3.2-AW-2024-08-30, device /dev/vipcore, ai-sdk
  (github.com/ZIFENG278/ai-sdk). Flow: ONNX -> pegasus import/quantize/inference/export -> NBG ->
  VIPLite. NOTE: ACUITY "int16" = dynamic fixed point (single power-of-2 scale per tensor), NOT
  IEEE fp16.

## 3. What WORKS (verified on hardware)
### LLM on NPU (int16, fixed-window, NPU-only), measured on Orange Pi:
| Model | W | Coherent | decode tok/s | first-token | RSS | NBG |
|---|---:|---|---:|---:|---:|---:|
| SmolLM2-135M | 32 | yes | 20.7 | 48 ms | 272 MB | 281 MB |
| SmolLM2-135M | 64 | weak | 14.0 | 72 ms | 274 MB | 282 MB |
| SmolLM2-360M | 32 | yes | 8.4 | 114 ms | 646 MB | 673 MB |
| SmolLM2-360M | 64 | yes | 4.9 | 212 ms | 649 MB | 675 MB |
Coherence CLIFF: W>=128 is incoherent for both (no KV-cache; fixed window = prompt+response
combined, ~25-50 words usable). 135M/W32 = fast/basic; 360M/W32 = smarter/slower.

### LLM on CPU (llama.cpp, real KV-cache), measured on Orange Pi — the "ROS2-paused" mode:
| Model | quant | ctx | decode tok/s | first-token |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | Q8_0 | 2k | 18.4 | ~3 s |
| Qwen2.5-0.5B | Q4_K_M | 2k | 19.3 | ~3 s |
| Qwen2.5-0.5B | Q8_0 | 16k | 2.2 | ~18 min |
Q8_0 was faster AND better quality than Q4_K_M here. Long context works but is impractical past
a few thousand tokens, and 0.5B is unreliable at 16k retrieval. CPU decode uses the 2 A76 cores
(thread count / CPU% not yet formally measured — minor open item).

### VLM on NPU, measured on Orange Pi:
- MobileCLIP-S0 vision encoder: input image 1x3x256x256 -> output 1x512 embedding; 22.6 ms/frame;
  RSS 14 MB; NBG 19 MB; on-board vs ACUITY-host cosine 0.99996, top-5 match. REAL working piece.
- Tiny VLM bridge (proof-of-concept ONLY): image embedding + token window -> logits; 0.063 ms;
  cosine 0.99999. Decoder is microscopic (vocab 16) — proves the data path image->embedding->text
  is closed on NPU, NOT a usable captioning/VQA model. There is NO real language head yet.

## 4. Hardware/toolchain facts learned (verified)
- NPU clock ~1.0 GHz (triple-confirmed: cycle counts, i.MX 8M Plus analogy, an independent A733
  repo reporting 1008 MHz). ~1500 MAC/cycle.
- int16 NBG load ~1.35 ms/MB. int16 is ~1.45x slower + ~2x working memory vs uint8 (measured).
- ACUITY per-tensor DIMENSION LIMIT = 65536 (verified externally via ST Edge AI forum, same
  Vivante NBG compiler: "dimension must be in range [0, 65536["). Qwen vocab 151936 > 65536.
- Verified legal dtype boundaries (TIM-VX source): MATMUL allows BF16 only as BF16,BF16->BF16
  (no mixed); SLICE allows BF16->BF16 and INT16->INT16 but NOT BF16->INT16; DATACONVERT supports
  INT16->BF16, BF16->F16/F32, F16->INT16 but NOT direct BF16->INT16. Legal bridge =
  BF16 -> F16 -> INT16 via standalone DataConvert nodes; a BF16<->INT16 boundary must NEVER be
  inside a SLICE/MATMUL/PERMUTE node.
- Porting Radxa->Orange Pi: NBG files transfer AS-IS (same silicon, same cid). MUST be rebuilt:
  the runner C binary (recompile on the OPi, glibc-matched), and the VIPLite .so must come from
  the OPi layout (/home/orangepi/lib) NOT the Radxa ai-sdk path. The kernel NPU module comes from
  the OPi BSP.

## 5. What was TRIED and VERIFIED to NOT work (do NOT repeat these)
### Qwen2.5-0.5B on NPU (the central blocker):
- Plain INT16: exports, INCOHERENT (host cosine 0.236) — int16 dynamic-fixed-point can't hold
  Qwen's large activation outliers (act_absmax ~1790).
- Full FP16: exports, INCOHERENT (cosine 0.541) — FP16 5-bit exponent can't cover the outliers.
- Full BF16: host quality PASSES (cosine 0.991, top-1 match) but NBG export FAILS at
  vnn_VerifyGraph status -3 / error 64768.
- Chunked lm_head (3x <=50646, host cosine 0.99999) with Concat: still VerifyGraph -3 / 64768.
- Chunked lm_head, no Concat (3 separate outputs): still VerifyGraph -3 / 64768.
- Chunk lm_head AND token embedding: failure MOVES to a BF16 DATACONVERT setup failure (65280).
  => the 151936 logits tensor is NOT the only BF16 blocker; the full BF16 body graph is also
     rejected even with all dims < 65536.
- ACUITY hybrid (BF16 only on top outlier layers + lm_head, INT16 elsewhere): host quality FAILS
  (cosine 0.254) AND export fails at an illegal BF16->DFP-INT16 boundary at a PERMUTE node (65280).
- W8A16 + SmoothQuant alpha=0.5: confounded — the aggressive smoothing corrupted even the int16
  control. (A gentler/per-channel W8A16 has NOT been cleanly tested — see open paths.)
### Other dead ends:
- SmolLM2-1.7B on NPU: NBG export FAILS (gen_nbg segfault / 0-byte NBG; 6.85 GB ONNX). Blocked.
- QDQ-ONNX import (pre-quantize externally, import into ACUITY): ACUITY ignores QDQ scales.
- TVM vsi_npu and hand-built TIM-VX: same VIPLite verifier; lack RMSNorm/gather/slice/SwiGLU
  coverage. Not viable for this Qwen path.
- ACUITY pegasus-inference host cosine as a board-run COHERENCE predictor for int16: UNRELIABLE —
  it gave false negatives (failed the known-working 135M/W32). Use the FP-oracle gate for builder
  correctness, and ACTUAL on-board generation for the coherence gate.

## 6. Verified building blocks that exist in the repo (reuse, don't rebuild)
- make_real_llm_onnx.py: builds fixed-window ONNX for SmolLM/Qwen-class models from HF weights;
  reads config correctly (rope_theta, QKV bias, GQA, RMSNorm eps); supports lm_head chunking and
  last-logit slice. Builder verified correct (FP32 ONNX vs HF oracle cosine ~1.0 for all configs).
- convert_onnx_to_nbg.sh: ONNX -> NBG via ACUITY (uint8 | int16 | pcq | bf16 | fp16).
- Host-oracle tooling: dump_real_llm_oracle.py, compare_onnxruntime_to_oracle.py,
  compare_acuity_host_to_oracle.py (FP-oracle gate = reliable; pegasus-inference int16 = not).
- Persistent NPU runner (npu_lm_runner): loads NBG once, stdio protocol, decode loop. Has Orange
  Pi variant (A733_VIP_LEGACY_DEVICE_ID, A733_VIP_NO_CORE_INDEX) built against the OPi vip_lite.h.
- chat_shell.py: interactive REPL on the board (streams tokens, ChatML template, window counter).
- Vendor blocker reports filed: reports/t6, t9, t10, t11 (the Qwen NBG export blockers).

## 7. WHY the limits exist (so you reason correctly, not re-litigate)
- NBG graphs are STATIC-SHAPE -> no dynamic KV-cache -> autoregressive decode uses a fixed window
  and recomputes the whole window every token -> short usable context (W~32-64) and decode that
  goes compute-bound (slower) as W grows. This is fundamental to the public stack, not a bug.
- 3 TOPS + 32-bit LPDDR5 -> decode is memory-bound; raw tok/s is modest. NPU's real value here is
  freeing the 2 A76 CPU cores, not beating CPU on tok/s for these tiny models (CPU with KV-cache
  is actually faster/smarter for <=0.5B).
- Qwen's activation outliers need BF16 for quality, but ACUITY's NBG compiler can't lay out the
  BF16-heavy graph (VerifyGraph -3, illegal BF16<->INT16 boundaries). Genuine toolchain limit.

## 8. OPEN paths (still worth trying — ranked, none yet attempted/finished)
1. HYBRID VLM/LLM (RECOMMENDED for a real assistant, esp. for the user's picoclaw + local LLM
   goal on the 6 GB Orange Pi): vision encoder on NPU (already works) + LLM decoder on CPU via
   llama.cpp (real KV-cache, long context, coherent). Easiest first step: run a small VLM
   (SmolVLM-256M/500M or InternVL3.5-1B) ENTIRELY on CPU via llama.cpp (mmproj) to get a working
   image-chat, then move only the vision encoder to NPU for offload. Expected: ~10-20 tok/s,
   answer to an image question in ~3-8 s, fits RAM easily.
2. Qwen-on-NPU, remaining un-tried angles (research-grade, vendor may still be required):
   (a) force ACUITY to emit the legal BF16->F16->INT16 bridge as standalone DataConvert nodes so
       no PERMUTE/SLICE/MATMUL straddles a mixed boundary;
   (b) PER-CHANNEL INT16 (per-axis affine, not single-scale DFP) — could hold Qwen's outliers AND
       export legally (all int16 boundaries are legal), sidestepping the BF16 wall entirely;
   (c) clean W8A16 (per-channel int8 weights + int16 activations, NO/light smoothing) — check if
       int8<->int16 boundaries are all legal, which would also avoid the BF16 wall;
   (d) block-partitioned multi-NBG (one NBG per decoder block, chained at runtime) — VeriSilicon's
       own public Qwen2.5 example is a single block, hinting this is the intended granularity.
3. File the vendor tickets (t6/t9/t10/t11 reproducers are ready) with Radxa/Allwinner/VeriSilicon.
4. SmolLM2-360M is the smartest model that runs NPU-only today (8.4 tok/s, W32) — a safe upgrade
   over 135M if staying pure-NPU.
5. Open minor item: measure CPU thread count + CPU% for the Qwen CPU baseline (B4 didn't record).

## 9. How to work going forward (process)
- One task = one focused Codex session; each commits a reports/<id>.md and updates status.md.
- Mark every claim "verified" (ran it) or "assumption". For NPU coherence, gate on ACTUAL
  on-board generation, not pegasus-inference cosine.
- For any NPU model: builder check (FP32 ONNX vs HF oracle) -> host export + quality -> board run.
  Only one NPU consumer on /dev/vipcore at a time; a CPU (llama.cpp) job can run alongside.
- The realistic end state to aim for: NPU vision + CPU LLM hybrid for a usable local assistant
  (picoclaw + local LLM), with SmolLM2-on-NPU kept as the NPU-only proof-of-concept.

## 10. Immediate suggested next step
Build the hybrid image-chat (open path #1): stand up a small VLM (start CPU-only via llama.cpp,
e.g. SmolVLM or InternVL3.5-1B) on the Orange Pi to get a working local image+text assistant,
then offload the vision encoder to the NPU. This directly serves the user's goal of running
picoclaw + a local LLM/VLM on the Orange Pi Zero 3W, and builds on everything already verified.