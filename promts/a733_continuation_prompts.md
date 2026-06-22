# A733 NPU LLM/VLM — Continuation Plan & Copy-Paste Prompts

**Goal:** run LLM/VLM model-layer compute on the Allwinner A733 Vivante VIP9000 NPU (NPU-only). Repo: `petayyyy/a733_npu_driver`. Dev board: Radxa Cubie A7Z (Debian 11, kernel 5.15.147-21-a733) @ `192.168.31.76`, user `radxa`. Final target: Orange Pi Zero 3W (Debian 12, kernel 6.1.31).

---

## How to use this file (so you don't get lost)

1. **Run tasks strictly in order: T0 → T1 → T2 → T3 → T4.** T5 (porting) and T6 (vendor) come later / only if needed.
2. **Each task = one fresh Codex session.** Paste the **MASTER PROMPT** first, then paste the task prompt below it. Do not run two tasks in one session.
3. **Gate every task.** Do not start the next task until the current task's *Success gate* is green and committed.
4. T1 and T2 can run in parallel (different sessions) — they don't depend on each other. T3 needs T2. T4 needs T1+T2+T3.

---

## MASTER PROMPT (paste at the TOP of every Codex session)

```
You are continuing the A733 Vivante VIP9000 NPU project in repo a733_npu_driver.
FIRST: read reports/status.md, docs/roadmap.md, and docs/npu-only-requirement.md.

HARD REQUIREMENT (NPU-only): all LLM/VLM model-layer compute (embeddings, attention,
MLP, norms, logits) must run on the A733 NPU. CPU is allowed ONLY for orchestration:
file I/O, tokenization/detokenization, launching the runtime, argmax/sampling, logging.
CPU must never be the primary compute engine for model layers.

WHAT IS ALREADY DONE AND VERIFIED ON HARDWARE — DO NOT REDO OR RE-LITIGATE:
- G0: Debian 11, kernel 5.15.147-21-a733, 8 cores, thermals OK.
- G1: /dev/vipcore present; VIPLite 2.0.3.2-AW-2024-08-30; cid=0x1000003b; YOLOv8n and
  SDK vpm_run run on the NPU.
- G2: ACUITY Docker ubuntu-npu:v2.0.10.1 converts ONNX->NBG; LeNet and Inception v1
  (uint8 + int16) validated on NPU, top-1 preserved.
- G3a vision: MobileCLIP-S0 vision encoder -> int16 NBG -> NPU; 1x512 embedding; 22.6 ms;
  cosine 0.9998 vs ACUITY.
- G3a transformer ON NPU (this is the key proven result): a tiny decoder block
  (causal attention, Softmax, GELU, LayerNorm-style reductions, residuals, logits), a
  tiny LM (int32 token_ids -> ONNX Gather embedding -> decoder -> logits), a tiny VLM
  bridge (image projector + Gather + concat + decoder), and an 8-step fixed-window decode
  loop ALL run on the NPU with cosine ~0.99999 and ZERO op fallbacks. Transformer-op
  support on this NPU is PROVEN; do not re-question it.

MEASURED HARDWARE CONSTANTS (verified from board run.logs):
- NPU clock ~1.0 GHz, ~1500 MAC/cycle (3 TOPS INT8 nominal).
- int16 is ~1.45x slower and uses ~2x working memory vs uint8/int8 (measured on Inception).
- NBG load time ~1.35 ms per MB of NBG (MobileCLIP 19 MB -> 25.6 ms load).
- Working memory pool is tiny (1.5 MB for the real vision model); the binding constraint
  is NBG weight size + per-token weight-read bandwidth, i.e. decode is memory-bound.

KNOWN ISSUES — DO NOT REDISCOVER:
- The current decode loop relaunches vpm_run per token, reloading the NBG every step.
  This is fine at tiny scale but FATAL for a real model (0.2-0.7 s/token just to reload).
  A persistent runtime that loads the NBG once is mandatory (this is task T1).
- sample.txt for vpm_run must be LF-only. Windows CRLF causes "Bad task file. Wrong line @ 0".
- No real LLM has been converted to NBG yet. INT4 is NOT in the public toolkit; int8 per-
  channel (pcq) is the lowest available weight precision.

TOOLCHAIN FACTS:
- ai-sdk built on board (ZIFENG278/ai-sdk). vpm_run at
  /home/radxa/ai-sdk/examples/vpm_run/vpm_run
- VIPLite libs: /home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
- ACUITY target id: VIP9000NANODI_PLUS_PID0X1000003B. Export uses
  pegasus_export_ovx_nbg.sh <name> <int16|uint8|pcq> ... --pack-nbg-unify
- NBG outputs: network_binary.nb + nbg_meta.json
- SSH automation helper: scripts/host/ssh_exec.py (password-based).
- Board model dir: /home/radxa/a733_npu_driver/models/

WORKING RULES:
- After each step: commit to git and update reports/status.md (done / blocked / next).
- Mark every technical claim "verified" (you ran it) or "assumption" (you reasoned it).
- If ACUITY or VIPLite rejects an operation or a graph, STOP, save the full stdout/stderr
  log under logs/, and write a precise blocker note for the vendor (op name + shapes + log).
  Do not silently work around it on CPU.

Now do ONLY the task I paste next. Do not start other tasks.
```

---

## T0 — Script the ACUITY conversion flow (infrastructure, prerequisite)

```
TASK T0: The ACUITY ONNX->NBG conversion steps were run by hand inside the Docker
container and are NOT yet committed as a reusable script. Fix this so every downstream
task converts models reproducibly.

DO:
1. Create scripts/host/convert_onnx_to_nbg.sh that takes: --name, --onnx, --dataset
   (calibration inputs), --quant (uint8|int16|pcq), --inputs, --input-size-list,
   --outputs, and runs the full ACUITY pipeline in ubuntu-npu:v2.0.10.1:
   pegasus import -> pegasus quantize (with the chosen quant) -> pegasus inference
   (host golden) -> pegasus_export_ovx_nbg.sh <name> <quant> ... --pack-nbg-unify.
   Target id VIP9000NANODI_PLUS_PID0X1000003B.
2. The script must emit, into work/model-packages/<name>/<quant>/: network_binary.nb,
   nbg_meta.json, sample.txt (LF-only!), and the host golden output tensor for later diff.
3. Add scripts/host/compare_outputs.py that diffs a board output_0.txt against the ACUITY
   host golden and prints: top-5 index match, max/mean abs diff, RMSE, cosine.
4. Re-run it on the EXISTING tiny_lm_gather model to prove parity with the prior manual run
   (expect cosine ~0.99999). Do not change the model; this only validates the script.

DELIVERABLE: convert_onnx_to_nbg.sh, compare_outputs.py, a short reports/t0-acuity-flow.md
showing the tiny_lm_gather re-conversion matches the previous result.

SUCCESS GATE (T0): one command converts tiny_lm_gather to NBG and compare_outputs.py
reports top-5 match + cosine > 0.9999 vs the committed result. Committed to git.

START FROM: existing scripts/host/make_tiny_lm_onnx.py, the prior manual steps described
in reports/g2-acuity-*.md and reports/g3a-tiny-lm-gather-npu.md.
```

---

## T1 — Persistent NPU runner (THE critical task — do this first among the engineering work)

```
TASK T1: Replace per-token vpm_run relaunches with a persistent C runtime that loads the
NBG ONCE and submits many token windows. The board logs prove per-token NBG reload is
fatal for a real model (~1.35 ms/MB load => 0.2-0.7 s/token for a real LM). This task
removes that overhead and is the prerequisite for any real tok/s measurement.

DO:
1. Study the SDK example at /home/radxa/ai-sdk/examples/vpm_run/ — it is a VIPLite/awnn
   application and is your template. Identify the init / create-network / prepare-network /
   set-input / submit / get-output / destroy calls.
2. Write a new app (e.g. examples/npu_lm_runner) that:
   - inits VIPLite and creates+prepares the NBG exactly ONCE,
   - then runs a loop: write the int32 token window into the input tensor in-place ->
     submit -> read the logits output -> argmax over the LAST position only -> append token
     -> slide the window -> repeat, with NO destroy/re-init between tokens,
   - prints per-step NPU profile time and a final mean tok/s.
3. Validate it on the EXISTING tiny_lm_gather NBG (1x4 int32 -> 1x4x16 logits). Reproduce
   the same generated token sequence as the prior per-token loop (1 5 9 2 -> 1 5 9 2 1 8 4 5
   8 4 8 4) to prove correctness, but now with the NBG loaded once.
4. Report the overhead delta: per-token wall time with persistent runner vs the old
   per-token vpm_run reload (the old loop's create-network ~0.45 ms + init per step).

DELIVERABLE: the runner source + build instructions in scripts/board/, and
reports/t1-persistent-runner.md with the matching token sequence and the per-token timing
comparison.

SUCCESS GATE (T1): persistent runner reproduces the tiny LM token sequence with the NBG
loaded once, and reports lower, stable per-token wall time than the reload loop. Committed.

START FROM: /home/radxa/ai-sdk/examples/vpm_run/ source; scripts/board/run-tiny-lm-decode-loop.sh
(the behavior to replicate, minus the reload).
```

---

## T2 — Architecturally-faithful tiny block (proves the REAL op set compiles)

```
TASK T2: The tiny models proven so far use LayerNorm, learned position embeddings, GELU,
single-head attention. Real small models (Qwen2.5 / Llama / SmolLM2 family) use RMSNorm,
RoPE, SwiGLU, and multi-head attention with GQA. Prove this REAL operator set compiles to
NBG and runs on the NPU, at tiny size, BEFORE scaling parameters. This isolates "does the
real architecture work" from "does it scale in size".

DO:
1. Extend scripts/host/ to generate a fixed-shape ONNX block (dim=64, 2 layers, n_heads=4,
   n_kv_heads=2 for GQA, vocab=256, window W=16) using ONLY these real components, all as
   primitive ONNX ops:
   - RMSNorm: x * Reciprocal(Sqrt(ReduceMean(x*x) + eps)) * gamma
   - RoPE as CONSTANTS (valid because the window is fixed): precompute cos/sin tables per
     position as initializers; rotate_half(x) = Concat(Neg(x[..., d/2:]), x[..., :d/2]);
     x_rope = x*cos + rotate_half(x)*sin. Apply to Q and K.
   - GQA: project K,V with n_kv_heads, then repeat each KV head to match n_heads (Reshape +
     Tile or Gather), then per-head scaled-dot-product attention (MatMul, scale, causal
     mask add, Softmax axis=-1, MatMul with V).
   - SwiGLU MLP: down( silu(gate(x)) * up(x) ), silu(y)=y*Sigmoid(y).
   - final RMSNorm -> logits MatMul.
   Input: int32 token_ids 1xW -> Gather embedding -> + (RoPE handled inside attention) ->
   blocks -> logits 1xWxvocab.
2. Convert with T0's convert_onnx_to_nbg.sh (int16 first), run on the board via vpm_run.
3. Diff board output vs ACUITY golden with compare_outputs.py.
4. If ANY op (RoPE concat/neg, GQA tile/gather, Reciprocal, Sigmoid, etc.) is rejected by
   ACUITY or falls back, STOP and record the exact op + shapes + log for the vendor note.

DELIVERABLE: the generator, the NBG, and reports/t2-faithful-block.md listing every op,
the NBG metadata, runtime, and the cosine vs ACUITY. Explicitly state which real ops
compiled cleanly.

SUCCESS GATE (T2): the RMSNorm+RoPE+SwiGLU+GQA block compiles to NBG, runs on the NPU,
and cosine vs ACUITY > 0.999. (If a specific op fails, the gate is instead a precise,
logged vendor blocker — that is also a valid, useful outcome.)

START FROM: scripts/host/make_tiny_decoder_block_onnx.py and make_tiny_lm_onnx.py as the
ONNX-construction pattern.
```

---

## T3 — Logits slice + int8 (pcq) — two efficiency changes, validated small

```
TASK T3: Apply two changes that are mandatory for a real model, and validate them on the
tiny faithful block from T2 (so that when the real model is built, these are already known-
good and not a source of confusion).

DO:
1. LOGITS SLICE: for autoregressive decode you only need the LAST position's logits.
   Modify the graph so the final logits MatMul runs on the last hidden position only:
   Slice/Gather hidden[:, -1:, :] -> [1,1,dim] -> MatMul -> [1,1,vocab], instead of computing
   logits for all W positions. (At real vocab=150k this avoids a ~W x waste.)
2. INT8 WEIGHTS: re-convert the T2 block with quant=pcq (int8 per-channel) instead of int16.
   Measured expectation: ~1.45x faster and ~half the working memory vs int16.
3. Validate both on the board: confirm the int8 slice graph still produces correct last-
   position argmax (compare argmax/top-5 to the int16 full-logits version), and record the
   speed/memory delta from the run.log (profile time, memory pool size).

DELIVERABLE: updated generator/flags + reports/t3-slice-int8.md with the int16-vs-int8
timing/memory comparison and confirmation the last-position argmax is unchanged.

SUCCESS GATE (T3): int8 (pcq) sliced-logits block runs on NPU; last-position argmax matches
the int16 version; measured speedup recorded. Committed.

START FROM: T2 output; measured int16/uint8 ratio in reports (Inception: 20.85 vs 14.36 ms).
```

---

## T4 — First REAL model end-to-end on NPU (SmolLM2-135M, then Qwen2.5-0.5B)

```
TASK T4: Build the first real small language model as a fixed-window NPU graph and decode
with the persistent runner. Start with the smallest viable real model to clear NBG size and
24-layer compile risk before going bigger.

DO:
1. Start with SmolLM2-135M-Instruct (smallest real target). Build a fixed-window ONNX
   (window W=32 first, then try 64) that mirrors its real architecture using the T2
   components (RMSNorm, RoPE-as-constants, SwiGLU, GQA, correct n_layers/dim/heads/vocab),
   and load the REAL pretrained weights from the HF checkpoint into the initializers. Include
   the T3 logits slice. Use the model's real tokenizer on the CPU side only (allowed).
2. Convert with quant=pcq (int8) via T0's script. Watch for: NBG compile success
   (Error(0)/Warning(0)), NBG size, and ACUITY accepting the full 30-layer graph at real
   vocab. If the converter chokes on size or an op at scale, STOP and log it precisely.
3. Run on the board with the T1 persistent runner. Measure: prefill time, decode tok/s,
   first-token latency, peak RSS, NPU profile per step. Verify outputs are coherent vs a
   CPU reference of the same model+tokenizer (CPU used only as a correctness oracle, not a
   deliverable).
4. Only after SmolLM2-135M passes, repeat the whole flow for Qwen2.5-0.5B-Instruct.

DELIVERABLE: the real-model build + conversion scripts, the NBG(s), and
reports/t4-real-model.md with the full benchmark (prefill/decode tok/s, latency, RAM, NPU
profile) and a coherence check vs CPU reference. State clearly the usable context window.

SUCCESS GATE (T4): SmolLM2-135M produces coherent text where every model-layer forward runs
on the NPU via the persistent runner, with measured tok/s recorded. Bonus gate: Qwen2.5-0.5B
same. (If conversion fails at scale, the gate is a precise logged blocker + the exact op/
size limit hit — escalate via T6.)

START FROM: T1 runner, T2 architecture, T3 slice+int8; HF checkpoints SmolLM2-135M-Instruct
and Qwen2.5-0.5B-Instruct.
```

---

## T5 — Port to Orange Pi Zero 3W (final target — run AFTER T4 works on Radxa)

```
TASK T5: Port the working NPU pipeline from the Radxa Cubie A7Z (kernel 5.15) to the final
target Orange Pi Zero 3W (Debian 12 bookworm, kernel 6.1.31). Same A733 silicon, same
cid=0x1000003b, so NBG files are expected to transfer as-is.

DO:
1. Flash the Orange Pi Zero 3W official image 1.0.4 (from orangepi.org — the Zero 3W on
   A733, NOT the Zero 3 on H618). Confirm boot, kernel 6.1.31, 8 cores, thermals.
2. Verify the NPU stack on this BSP: is /dev/vipcore present? Is a VIPLite runtime/npu-runtime
   package available in the Orange Pi BSP? Locate or obtain the kernel NPU module + VIPLite
   .so for 6.1.31. (The kernel module must come from the Orange Pi BSP; it is not portable
   from the Radxa kernel.)
3. Copy the T4 NBG files unchanged and run them with the persistent runner. The runner C app
   must be RECOMPILED on bookworm (glibc 2.36), not reused from the bullseye binary.
4. Re-run the T4 benchmark on the Orange Pi and compare outputs to the Radxa results (expect
   matching argmax/top-5 within tolerance, since the silicon is identical).

WHAT TRANSFERS AS-IS: NBG model files, the ACUITY conversion workflow, the model architecture.
WHAT MUST BE REBUILT: the kernel NPU module (from OPi BSP), the runner C binary (bookworm
glibc), and any device-tree/overlay needed to expose /dev/vipcore.

DELIVERABLE: reports/t5-orangepi-port.md with the bring-up checklist result, what transferred
vs what was rebuilt, and the Radxa-vs-OrangePi benchmark/output comparison.

SUCCESS GATE (T5): the same NBG runs on the Orange Pi Zero 3W NPU via a bookworm-rebuilt
runner, with outputs matching the Radxa within tolerance and tok/s recorded on the final
target. Committed.

START FROM: T1 runner source, T4 NBGs; orangepi.org Zero 3W downloads; the kernel-portability
analysis in docs/roadmap.md.
```

---

## T6 — Vendor request (CONDITIONAL — only if T2/T4 hits an op or size wall)

```
TASK T6 (only if a previous task logged a hard ACUITY/VIPLite blocker): Turn the logged
blocker into a precise, evidence-backed request to Allwinner / Radxa / VeriSilicon.

DO:
1. Collect the exact failure: op name, tensor shapes, quant mode, the full ACUITY/VIPLite
   stdout+stderr, the ONNX snippet, and the NBG metadata if any was produced.
2. Write a concise request stating: what op/graph fails, the minimal reproducer, what you
   need (e.g. an ACUITY transformer/INT4 export path; VIPLite dynamic-shape or KV-cache
   support; a reference transformer NBG; the MAC/clock spec for A733 VIP9000; confirmation
   that the npu-runtime package ships in the Orange Pi Zero 3W BSP).
3. File it on the appropriate channel (Radxa docs/issue tracker, Allwinner/aw-ol forum,
   VeriSilicon contact) and record the request + any response in reports/t6-vendor.md.

DELIVERABLE: reports/t6-vendor.md with the reproducer, the logs, and the exact ask.

SUCCESS GATE (T6): a filed, reproducible vendor request with attached logs. (This is a valid
project outcome when a hard limit of the public stack is hit.)
```

---

## Quick dependency map

```
T0 (ACUITY flow script) ─┬─> T2 (faithful block) ──> T3 (slice + int8) ──> T4 (real model) ──> T5 (port to Orange Pi)
                         │                                                      │
T1 (persistent runner) ──┴──────────────────────────────────────────────────> ┘
                                                                               └─(if blocked)─> T6 (vendor)
```

- **Do first / in parallel:** T0 and T1.
- **Then:** T2 → T3 → T4.
- **Last:** T5. **Only if blocked:** T6.
- **Highest-information steps:** T2 (does the real op set compile?) and T4 (does it scale to a real model?). If both pass, you have NPU-only LLM inference on the A733.

---

## Notes on running this in Codex (GPT-5.5 Extra High)

- The repo is the shared memory across sessions — always start by reading `reports/status.md`. Codex sessions are stateless; the MASTER PROMPT + status.md are what carry context forward.
- Tasks that touch the board (T1, T4, T5) need the runner/commands to execute on the A733 over SSH (use `scripts/host/ssh_exec.py` or `ssh radxa@192.168.31.76 '<cmd>'`). T0/T2/T3 host-side ACUITY work needs the x86 Docker image `ubuntu-npu:v2.0.10.1`.
- Commit after every gate and keep `reports/status.md` updated with done / blocked / next — that is what prevents the "everything in one chat" confusion from last time.
