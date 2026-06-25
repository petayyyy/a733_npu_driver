TASK DOC3-finalize: Bring ALL documentation into a clean, consistent, final state reflecting the
project's true end state. The research/experimentation phase is essentially complete; this is the
"ship it as a good open-source resource" pass. Update existing docs, don't duplicate. Every number
must match the reports/. Keep the honest verified/assumption + works/doesn't-work framing.

=== CURRENT TRUE STATE (use as the single source of truth; reconcile all docs to this) ===
WORKS ON NPU (verified, Orange Pi Zero 3W):
- SmolLM2-135M int16, W=32: 20.7 tok/s, coherent; W=64: 14.0 tok/s (weaker). NPU-only.
- SmolLM2-360M int16, W=32: 8.4 tok/s; W=64: 4.9 tok/s. NPU-only, coherent.
- Coherence cliff: W>=128 incoherent for both (no KV-cache; fixed window = prompt+response).
- MobileCLIP-S0 vision encoder: 22.6 ms/frame, cosine 0.99996.
- Interactive chat shell (B2) on board.
WORKS ON CPU (verified, the practical productivity layer):
- Qwen2.5-0.5B Q8_0, ctx 2k: 18.0 tok/s decode on 2x A76 (taskset -c 6,7) = 25% of the machine,
  6x A55 cores left free for ROS2. Q8_0 beats Q4_K_M here. Long context works but impractical
  (16k = ~18 min first-token; 0.5B unreliable at 16k).
- SmolVLM-256M-Instruct Q8_0 image chat: 52.6 tok/s, 634 MB RSS, accurate (described dog/cat/
  read a 1945 moon-landing newspaper clipping), leaves ~2.3 GB for ROS2. SmolVLM-500M = 22.3
  tok/s, more detail.
VENDOR-GATED / BLOCKED (exhaustively proven on hardware, NOT assumed):
- Qwen2.5-0.5B on NPU: every monolithic config fails (int16 cosine 0.236; FP16 0.541; BF16 won't
  export, vnn_VerifyGraph -3/64768; per-channel int16 does not exist in ACUITY, only INT8/INT4;
  W8A16 cosine 0.079). Block-partitioned int16 (Q2): host sim coherent (0.975) and runtime chaining
  works (Gate 2B, load-once, no reload), but on HARDWARE int16 dynamic-fixed-point collapses over
  24-deep chaining -> degenerate output at 6.6 tok/s (Gate 2C). Genuinely needs vendor support.
- SmolLM2-1.7B on NPU: NBG export fails (gen_nbg segfault / 0-byte, 6.85 GB ONNX).
- SmolVLM SigLIP vision encoder on NPU (V2): ACUITY Conv shape-inference crash (_conv_shape).
  [NOTE: a V2 retry may change this -- see DO below.]
- Real KV-cache: no VIPLite/TIM-VX support (static-shape NBG); short fixed window only.
KEY HARDWARE FACTS (verified): VIP9000, cid 0x1000003b, ~1.0 GHz, ~3 TOPS, native
INT8/INT16/FP16/BF16, 32-bit LPDDR5, effective NPU decode bandwidth ~6 GB/s, int16 NBG load
~1.35 ms/MB, ACUITY per-tensor dimension limit 65536. Final board Orange Pi Zero 3W, 6 GB, kernel
6.6.98, /dev/vipcore, VIPLite 2.0.3.2.

=== DO ===
1. Reconcile EVERY file in docs/ and the root README/RESULTS/blockers/configurations to the state
   above. Specifically:
   - README.md: update the "what works / what doesn't" box to the final state; ensure the
     "start here" path and links are correct; add a one-line honest thesis ("this NPU is a
     vision/CNN accelerator that also runs tiny LLMs; it is NOT an LLM accelerator for Qwen-class
     models -- the productive path is hybrid: NPU vision + small-LLM, CPU for Qwen-class").
   - docs/RESULTS.md: make the master tables exactly match the numbers above (LLM-on-NPU,
     LLM-on-CPU incl. the B4b utilization numbers: 2xA76 = 18 tok/s/25%/6 cores free, the full
     thread-sweep table, VLM-on-NPU, VLM-on-CPU from V1, hardware facts). Add a "coherence cliff"
     note and a "why these limits" section (static-shape -> no KV-cache -> short window; memory-
     bound decode; int16-DFP collapses on Qwen outliers; BF16 won't compile).
   - docs/blockers.md: list every blocker above with its exact error/cosine and a link to the
     report that proved it (Q1, Q2 2C, T8-T11, V2, the 1.7B segfault). Mark all "verified on
     hardware".
   - docs/configurations.md: the decision guide. For each use case give the FINAL recommended
     config: fast NPU chat (SmolLM2-135M/W32, 21 tok/s); smarter NPU chat (360M/W32, 8 tok/s);
     usable chat with real context (Qwen-0.5B on CPU Q8_0, taskset -c 6,7, 18 tok/s, 6 cores free);
     image chat for picoclaw (SmolVLM-256M on CPU, 52 tok/s); NPU vision offload (MobileCLIP).
     Mark the CPU-hybrid (Qwen/SmolVLM on CPU + SmolLM2/MobileCLIP on NPU) as the RECOMMENDED
     production path, with the rationale (NPU frees A76 cores; CPU runs the smarter models).
   - docs/roadmap.md: mark the research phase complete; list the only remaining open items (V2
     vision-encoder retry; vendor tickets) and what each would unlock.
   - Reconcile int8-quantization-strategy.md / npu-only-requirement.md / smollm2-chat.md and any
     other docs to the final state; remove or correct anything now contradicted (e.g. stale
     "Qwen int8 TBD" language -> "Qwen on NPU vendor-gated, proven").
2. Cross-link: README -> RESULTS / configurations / blockers; every blocker -> its proving report.
   Ensure docs/import_chat.md (the handoff) reflects the final state too.
3. Add docs/vendor-tickets.md consolidating the precise vendor-blocker packets (T6 hybrid-table,
   T9 BF16 VerifyGraph -3, T10/T11 mixed boundaries, Q2 2C int16-depth-collapse, V2 Conv crash),
   each with the exact error code, op, shapes, and toolchain version, ready to file with
   Radxa/Allwinner/VeriSilicon.
4. Repo hygiene: verify README directory map, LICENSE present, CONTRIBUTING accurate, .gitignore
   keeps work/ and large artifacts out. Fix any broken doc links. Ensure every command in docs/ is
   copy-pasteable and matches what the reports actually ran.
5. Add a concise top-level "Project Summary / Conclusions" section to README (or docs/SUMMARY.md):
   what was attempted, what works, what's vendor-gated and why, and the honest one-paragraph lesson
   about this NPU's real envelope. This is the "beautiful finale" centerpiece for an outside reader.

DELIVERABLE: a fully reconciled docs/ + README where every number matches reports/, every blocker
links to its proof, configurations.md recommends the hybrid path, vendor-tickets.md is filing-ready,
and a clear Summary/Conclusions section exists. No code behavior changed.

SUCCESS GATE: an outside reader can land on README, understand exactly what works / what doesn't /
why, pick a configuration, and follow a docs/ guide -- with zero contradictions between any doc and
the reports. Committed.

START FROM: all existing reports/ (g/t/b/q/v series) and docs/ (README, RESULTS, blockers,
configurations, roadmap, import_chat, int8-quantization-strategy, etc.). The numbers above are the
reconciliation target.