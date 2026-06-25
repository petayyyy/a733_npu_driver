TASK Q2-finish-block-chain: Q2 Gate 1 PASSED -- a single Qwen2.5-0.5B decoder block exports to
int16 NBG with host cosine 0.999965, proving (a) the vnn_VerifyGraph -3 wall is an AGGREGATE-graph
limit beatable by per-block splitting, and (b) int16 per-block quality is near-perfect (the
end-to-end int16 failure, cosine 0.236, is DEPTH-ACCUMULATED, not per-layer-catastrophic). Finish
Gate 2: get a coherent, block-chained int16 Qwen2.5-0.5B running on the Orange Pi NPU. Two
independent risks must BOTH clear; test them cheapest-first. STOP at the first hard failure.

=== GATE 2A: full-chain HOST coherence (CHEAPEST -- do this FIRST, no board, no runtime build) ===
The make-or-break question is whether 24 chained int16 blocks stay coherent end-to-end, or whether
int16 error accumulates across depth into garbage (as the monolithic int16 did at 0.236).
1. Compile all 26 stage NBGs to int16: embedding stage, 24 decoder blocks, final stage
   (RMSNorm + chunked lm_head). Block 0 is already done (22.6 MB, cosine 0.999965); compile the rest.
2. Build a HOST-ONLY chained simulation: run each stage through the ACUITY host simulator (or
   onnxruntime on the int16-quantized per-stage graphs), feeding stage N's output tensor as stage
   N+1's input, for a fixed W=32 token window. Embedding -> block 0..23 -> final -> logits.
3. Compare the final chained logits to the FP32 oracle (dump_real_llm_oracle.py) for the fixed
   prompt "The capital of France is": logits cosine + top-1 + decode the first ~6 tokens.
4. ALSO log per-stage cosine drift: cosine of each block's output vs the FP oracle's hidden state
   at that layer, so you can SEE where/if coherence degrades across the 24 blocks.
   GATE 2A: end-to-end chained host logits cosine > 0.90 AND top-1 match AND first ~6 tokens
   match the oracle ("The capital of France is Paris").
   - If 2A PASSES -> int16 block-chaining is coherent; proceed to Gate 2B.
   - If 2A FAILS (depth accumulation still kills it) -> STOP. Record the per-stage drift curve
     (where it breaks down). This is a real, valuable finding: int16-per-block-good but
     depth-bad. Do NOT build the runtime. Qwen-on-NPU stays vendor-gated; hybrid is the path.

=== GATE 2B: VIPLite Multi-Graph load-once chaining (the RUNTIME make-or-break) ===
Only if 2A passes. The fatal risk: per-token reload of 26 NBGs (~1 GB) is ~350 ms/stage = dead.
The chain is ONLY viable if all 26 NBGs are loaded ONCE and reused.
1. Investigate VIPLite Multi-Graph in the SDK: can the C runtime create+prepare all 26 NBGs once,
   then in the decode loop submit them in sequence per token, passing each stage's output tensor
   to the next stage's input (host-side memcpy or shared tensor) WITHOUT destroy/reload?
   Read /home/orangepi/lib headers and the ai-sdk examples for multi-network / multi-graph API.
2. Build a minimal 2-NBG proof first: load block0 + block1 once, chain block0_out -> block1_in in a
   loop, confirm no per-iteration reload and measure per-iteration time. This de-risks the API
   before wiring all 26.
   GATE 2B: 2-NBG chain runs with both NBGs loaded once (no reload), stable per-iteration time.
   - If VIPLite cannot keep multiple NBGs resident / chain them -> STOP. Record it as the runtime
     blocker (would need vendor Multi-Graph support). Hybrid is the path.

=== GATE 2C: full 26-NBG decode on the Orange Pi (only if 2A and 2B both pass) ===
1. Extend the persistent runner to load all 26 NBGs once and run the chained fixed-window decode
   loop: tokenize (CPU) -> embedding NBG -> blocks 0..23 -> final NBG -> argmax (CPU) -> slide
   window. CPU only for tokenize/memcpy/argmax/detokenize.
2. Watch RAM: ~1 GB of NBGs must be resident simultaneously on the 5.7 GB board (confirm free -h).
3. Validate coherent text vs the FP oracle on the board. Measure: decode tok/s, prefill,
   first-token latency, peak RSS, and the per-token cost of 26 sequential NBG submits (the main
   overhead vs a monolith).
   GATE 2C: coherent Qwen2.5-0.5B text on the Orange Pi NPU, every layer on NPU, tok/s recorded.

DELIVERABLE: reports/q2-qwen-block-nbg.md (extend the existing file) with: Gate 2A end-to-end host
cosine + the per-stage drift curve; Gate 2B Multi-Graph finding + 2-NBG timing; Gate 2C board
coherence + tok/s/RSS/overhead -- OR the exact gate that failed and why. Mark each result
verified. Update the confirmed-dead-ends list if a gate fails.

SUCCESS GATE: block-chained Qwen2.5-0.5B coherent on the Orange Pi NPU with tok/s recorded -- OR a
precise failure at 2A (depth-accumulation, with drift curve) or 2B (no Multi-Graph), which
confirms hybrid CPU-LLM + NPU-vision as the path.

DO NOT: build the full 26-NBG runtime before Gate 2A passes on host; reload NBGs per token; retry
any monolithic-graph, BF16, FP16, W8A16, or per-channel-int16 config (all confirmed dead).

START FROM: the Q2 Gate 1 block-0 NBG + the per-stage ONNX export modes already added to
make_real_llm_onnx.py (--export-block/--export-embedding/--export-final); convert_onnx_to_nbg.sh
(int16); the host-oracle tooling; VIPLite Multi-Graph in /home/orangepi/lib + ai-sdk examples;
the T10b persistent runner as the base; Orange Pi at 192.168.31.225.