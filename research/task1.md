You are a senior embedded-NPU compiler engineer. Investigate a precisely characterized set of
blockers for running Qwen2.5-0.5B and for int8 LLM inference on a VeriSilicon Vivante VIP9000 NPU
via the ACUITY/VIPLite toolchain, and find concrete workarounds. Use web search aggressively;
cite primary sources (GitHub source/issues/commits, vendor docs, forum threads). Tag every claim
"verified" (primary source found) or "assumption". A confirmed dead end is a valid result.
Distinguish "works on THIS exact toolchain/NPU" from "general theory that may not be expressible
here". Build on the findings below; do NOT re-derive them.

=== HARDWARE / TOOLCHAIN (fixed, public, no NDA) ===
SoC Allwinner A733; NPU VeriSilicon Vivante VIP9000, cid=0x1000003b, target
VIP9000NANODI_PLUS_PID0X1000003B, single core, ~1.0 GHz, native INT8/INT16/FP16/BF16.
Toolchain: ACUITY 6.30.22 (Docker ubuntu-npu:v2.0.10.1), Vivante IDE 5.11.0, runtime VIPLite
2.0.3.2, device /dev/vipcore, plus open VeriSilicon TIM-VX and the TVM vsi_npu fork. Flow:
ONNX -> pegasus import/quantize/inference/export -> NBG -> VIPLite. ACUITY "int16" = DYNAMIC
FIXED POINT (single power-of-2 scale per tensor), NOT IEEE fp16.

=== ESTABLISHED FACTS (verified by our runs / by source — treat as given) ===
- A real SmolLM2-135M and 360M (fixed-window W=32/64, RMSNorm/RoPE/SwiGLU/GQA) run coherently on
  this NPU NBG-only in INT16. So transformer ops + fixed-window (no-KV-cache) decode + int16
  export all work for outlier-free models. Our ONNX builder is verified correct (FP32 ONNX vs HF
  oracle cosine ~1.0 for every model/window).
- Qwen2.5-0.5B (24 layers, hidden 896, 14 q / 2 kv heads, vocab 151936, rope_theta 1e6, QKV bias)
  has large activation outliers (act_absmax ~1790; some RMS-squared tensors ~2.65e6). Results,
  all host ACUITY sim vs FP oracle BEFORE board:
  * INT16: exports, incoherent (cosine 0.236).
  * Full FP16: exports, incoherent (cosine 0.541) — FP16 5-bit exponent can't cover outliers.
  * Full BF16: HOST QUALITY PASSES (cosine 0.991, top-1 match) but NBG export FAILS at
    vnn_VerifyGraph status -3 / "Fatal model generation error 64768".
- Verified external: the "status -3 / 64768" signature on the same Vivante NBG compiler is
  documented (ST Edge AI forum) as a PER-TENSOR DIMENSION LIMIT: "dimension must be in range
  [0, 65536[". Qwen vocab 151936 > 65536. Corroborated by vLLM-Ascend "chunked matmul to work
  around the NPU 65536 dimension limit".
- We tried chunking the lm_head into 3 chunks of <=50646 (host ORT cosine 0.99999, perfect):
  * Chunked lm_head, with final Concat: still vnn_VerifyGraph -3 / 64768.
  * Chunked lm_head, no Concat (3 separate logits outputs): still -3 / 64768.
  * Chunk lm_head AND token-embedding table: failure MOVES to a BF16 DATACONVERT setup failure
    (error 65280) at node DATACONVERT.
  => the 151936 logits tensor is NOT the only BF16 export blocker; full BF16 body graph is also
     suspect even with all dims < 65536.
- We tried ACUITY hybrid (BF16 only on top outlier layers + lm_head, INT16 elsewhere): host
  inference runs but host quality FAILS (cosine 0.254), AND export fails at an illegal
  BF16-to-DFP-INT16 boundary at a PERMUTE node (error 65280). Earlier W8A16 + SmoothQuant
  (alpha=0.5) attempt was confounded — the aggressive smoothing corrupted even the int16 control.
- Verified legal dtype boundaries (TIM-VX source): MATRIXMUL allows BF16 only as BF16,BF16->BF16
  (no mixed); SLICE allows BF16->BF16 and INT16->INT16 but NOT BF16->INT16; DATACONVERT supports
  INT16->BF16, BF16->F16/F32, F16->INT16, but NOT direct BF16->INT16. So a legal bridge is
  BF16 -> (DataConvert) F16 -> (DataConvert) INT16, and any BF16<->INT16 boundary must be its own
  DataConvert node, never inside SLICE/MATMUL/PERMUTE.
- Confirmed dead ends: plain INT16 Qwen (incoherent); full FP16 (incoherent); unchunked full BF16
  (export -3); chunked BF16 lm_head alone (export -3); chunked lm_head + embedding (DataConvert
  65280); hybrid BF16-top-layers (host 0.254 + PERMUTE 65280); BF16/INT16 boundaries placed ON
  SLICE/MATMUL/PERMUTE nodes; QDQ-ONNX import (ACUITY ignores QDQ scales); TVM vsi_npu / hand-
  built TIM-VX (same verifier, no RMSNorm/gather/slice/SwiGLU coverage).

=== WHAT TO INVESTIGATE (ranked, build on the above) ===
1. The PERMUTE/DataConvert dtype walls: in ACUITY's generated graph, BF16 transformer output meets
   INT16 at PERMUTE and DATACONVERT nodes and fails. Find HOW to force ACUITY to materialize the
   legal two-hop bridge BF16 -> F16 -> INT16 as explicit standalone DataConvert/Cast nodes around
   PERMUTE, so no PERMUTE/SLICE/MATMUL ever consumes a mixed BF16/INT16 boundary. Look at ACUITY
   pegasus options for forcing cast insertion, per-tensor output dtype control, attach/transform
   passes, or inserting explicit Cast ops in the ONNX so ACUITY maps them to DATACONVERT. Is there
   an ACUITY transform/attach config that controls where dtype conversions are placed?
2. The residual BF16 export -3 with ALL dims < 65536: after lm_head chunking the full BF16 body
   still hits -3. What OTHER VIP9000/NBG resource limit triggers vnn_VerifyGraph -3 / 64768 besides
   the 65536 per-dim cap? Search for: total tensor count, NBG size cap, BF16 buffer/tiling limits,
   number of layers, concat/output count, per-layer SRAM/working-memory limits. Is there a known
   ACUITY/Vivante IDE version where full BF16 multi-layer transformer NBG export works? Where to
   get a newer public ACUITY/IDE (Radxa, VeriSilicon, NXP eIQ which bundles a Vivante converter)?
3. Per-channel INT16 (not BF16, not single-scale DFP): Qwen's outliers are channel-concentrated
   (large QKV-bias terms). Does ACUITY support PER-CHANNEL / per-axis affine INT16 (asymmetric or
   symmetric) rather than single-scale DFP int16? Per-channel int16 could hold the outlier range
   AND export legally (int16 boundaries are all legal), sidestepping the entire BF16 export wall.
   Find the exact ACUITY pegasus flag/quantizer for per-channel int16 and any evidence it works on
   VIP9000 for a transformer.
4. W8A16 done correctly (the earlier attempt was confounded by over-smoothing): what is the
   minimal-damage recipe on ACUITY — per-channel INT8 weights + INT16 activations, with NO or very
   light SmoothQuant (alpha<=0.2) or per-tensor activation clipping — that preserves Qwen coherence?
   Crucially, does W8A16 avoid the BF16 export wall entirely (int8 weights + int16 activations =
   only int8/int16 boundaries, which may all be legal)? Verify the legal INT8<->INT16 boundaries in
   TIM-VX (matmul/permute/slice/dataconvert) the same way we verified BF16.
5. Block-partitioned multi-NBG pipeline: VeriSilicon's own public Qwen2.5 example is a SINGLE
   transformer block, not a monolithic model. Is the intended deployment to compile each decoder
   block as a separate NBG and chain them at runtime (block N output tensor -> block N+1 input)?
   Find any VeriSilicon/NXP/Khadas precedent for chaining per-block NBGs for an LLM, the runtime
   mechanism (VIPLite multi-network, passing tensors between NBGs), and whether per-block BF16 NBGs
   export where the monolithic one fails.
6. KV-cache / dynamic state (the deeper win): does VIPLite/TIM-VX expose any stateful or
   variable-length tensor (TIM-VX has a VARIABLE tensor for recurrent state) that could implement a
   real KV-cache, removing the fixed-window recompute and the per-window NBG? Current maturity and
   any transformer precedent.

=== DELIVERABLE ===
- A section per item (1-6), tagged verified/assumption with source links.
- A single RECOMMENDED NEXT ACTION ranked by promise/effort for getting a coherent, exportable
  Qwen2.5-0.5B NBG on THIS toolchain — naming the most likely of: explicit DataConvert bridge
  placement, per-channel INT16, correct W8A16, block-partitioned NBGs, or a specific newer ACUITY
  version. State if it is genuinely blocked without NDA/vendor.
- An updated "confirmed dead ends" list (add anything you rule out) so we don't re-try them.

=== STARTING RESOURCES ===
VeriSilicon TIM-VX: github.com/VeriSilicon/TIM-VX ; acuity-models: github.com/VeriSilicon/acuity-models
(see models/qwen2.5_7b_decode) ; VIP9000: verisilicon.com/en/IPPortfolio/VivanteVIP9000 ;
TVM vsi_npu: github.com/VeriSilicon/tvm (branch vsi_npu) ; Radxa A733 NPU docs:
docs.radxa.com/en/cubie/a7a/app-dev/npu-dev ; ai-sdk mirror: github.com/ZIFENG278/ai-sdk ;
NXP eIQ / i.MX 8M Plus (same VIP9000 family) converter & forums ; ST Edge AI forum (status -3 /
64768 dimension limit) ; vLLM-Ascend (65536 chunked-matmul workaround) ; RKLLM:
github.com/airockchip/rknn-llm ; SmoothQuant/AWQ/GPTQ arXiv.