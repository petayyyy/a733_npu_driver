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