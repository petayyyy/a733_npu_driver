TASK DOC2-results-tables: Consolidate all measured results from this project into ONE clean,
well-structured results document with comparison tables, so readers see at a glance what runs,
where, and how fast. Use the verified numbers below as the source of truth (they come from the
project's own board runs); pull any missing details from the reports/ files. Tag each number
verified (measured on board) or estimate. Keep it honest — show failures too.

Create docs/RESULTS.md with these tables and short framing text:

=== TABLE 1: LLM on NPU (Orange Pi Zero 3W, 6 GB, int16, fixed-window, NPU-only) ===
| Model | W | Status | Coherent | decode tok/s | first-token | peak RSS | NBG |
SmolLM2-135M  W32  : OK,       coherent      , 20.7 tok/s, 48 ms , 272 MB, 281 MB
SmolLM2-135M  W64  : OK,       weak          , 14.0 tok/s, 72 ms , 274 MB, 282 MB
SmolLM2-135M  W128 : exported, incoherent    , 6.0  tok/s, 166 ms, 282 MB, 287 MB
SmolLM2-135M  W256 : exported, incoherent    , 1.2  tok/s, 860 ms, 375 MB, 337 MB
SmolLM2-360M  W32  : OK,       coherent      , 8.4  tok/s, 114 ms, 646 MB, 673 MB
SmolLM2-360M  W64  : OK,       coherent      , 4.9  tok/s, 212 ms, 649 MB, 675 MB
SmolLM2-360M  W128 : exported, incoherent    , 2.0  tok/s, 502 ms, 681 MB, 693 MB
SmolLM2-360M  W256 : exported, incoherent    , 1.2  tok/s, 834 ms, 711 MB, 709 MB
SmolLM2-1.7B  all  : NBG export FAILS (gen_nbg segfault / 0-byte, 6.85 GB ONNX)
Qwen2.5-0.5B  all  : NBG export FAILS (BF16 needed for outliers; vnn_VerifyGraph -3 / 64768)

=== TABLE 2: LLM on CPU (Orange Pi Zero 3W, llama.cpp, real KV-cache) ===
Qwen2.5-0.5B Q8_0  ctx 2k : decode 18.4 tok/s, prefill 48 tok/s, first-token ~3 s
Qwen2.5-0.5B Q4_K_M ctx 2k: decode 19.3 tok/s, prefill 18 tok/s, first-token ~3 s
Qwen2.5-0.5B Q8_0 ctx 16k : decode 2.2 tok/s (real chat), prefill 13 tok/s, first-token ~18 min
(note: Q8_0 was faster AND higher quality than Q4_K_M on this board; long context impractical and
0.5B unreliable at 16k retrieval)

=== TABLE 3: VLM on NPU (Orange Pi Zero 3W) ===
MobileCLIP-S0 vision encoder: input 1x3x256x256 image -> output 1x512 embedding; 22.6 ms/frame;
  peak RSS 14 MB; NBG 19 MB; on-board vs ACUITY-host cosine 0.99996, top-5 match
Tiny VLM bridge (proof-of-concept): input image embedding + token window -> logits; 0.063 ms;
  NBG 94 KB; cosine 0.99999. NOTE: decoder is tiny (vocab 16) -- proves the data path, NOT a
  usable captioning/VQA model.

=== TABLE 4: Hardware / NPU facts (verified) ===
NPU: Vivante VIP9000, cid 0x1000003b, single core, ~1.0 GHz, ~3 TOPS INT8, native
INT8/INT16/FP16/BF16. int16 NBG load ~1.35 ms/MB. int16 is ~1.45x slower + ~2x working memory vs
uint8. ACUITY tensor dimension limit 65536 (the Qwen vocab 151936 blocker). Effective NPU decode
bandwidth measured ~6 GB/s.

ALSO INCLUDE in RESULTS.md:
- A "Coherence cliff" note: on-board coherence holds at W=32/64 and breaks at W>=128 for both
  models (no KV-cache; fixed window = prompt+response combined ~25-50 words).
- A "Why these limits" short section: static-shape NBG -> no KV-cache -> short window + per-token
  full recompute; memory-bound decode on a 32-bit LPDDR5 bus; int16 dynamic-fixed-point fails on
  Qwen's activation outliers; BF16 fixes quality but won't compile (vendor-blocked).
- A "Recommendations by use case" table: tiny fast NPU chat (135M/W32, 21 tok/s); smarter NPU chat
  (360M/W32, 8 tok/s); usable chat with real context (Qwen-0.5B on CPU, Q8_0, ~18 tok/s); VLM
  vision offload (MobileCLIP on NPU); hybrid (NPU vision + CPU LLM) as the recommended path for a
  real assistant.
- A "Blocked / vendor-gated" subsection summarizing Qwen2.5-0.5B and SmolLM2-1.7B with links to
  the vendor blocker reports (t6, t9, t10, t11) and the 65536 dimension-limit finding.
- A comparison-to-RK3588 note (honest): RK3588 (6 TOPS, RKLLM with KV-cache + int4/int8) runs
  full LLMs/VLMs like InternVL; A733 (3 TOPS, no KV-cache, no working LLM int4/int8 export) cannot
  match this -- the realistic equivalent here is NPU-vision + CPU-LLM hybrid.

DELIVERABLE: docs/RESULTS.md with Tables 1-4 + the framing sections above, all numbers tagged
verified/estimate, cross-linked to the detailed reports/. README.md should link to it prominently.

SUCCESS GATE: one document where a reader sees every measured result, what works/fails, and the
recommended configuration per use case. Numbers match the reports. Committed.

START FROM: reports/b1b-benchmark-matrix.md, b2-chat-shell.md, b3-vlm-orangepi.md,
b4-qwen-cpu-baseline.md, t9/t10/t11 (Qwen blockers), and the hardware facts in t1-t4.