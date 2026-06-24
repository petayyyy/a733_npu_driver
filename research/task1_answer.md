# Running Qwen2.5-0.5B / INT8 LLM Inference on the VeriSilicon VIP9000 (Allwinner A733): Source-Backed Workarounds

## TL;DR
- The single highest-promise, lowest-effort path on THIS exact toolchain is **correct W8A16 (per-channel INT8 weights + per-tensor INT16 activations) with light or carefully-tuned SmoothQuant**, because it likely sidesteps the BF16 export wall entirely — all dtype boundaries become INT8↔INT16, which are integer-only conversions the NBG verifier already handles for shipping vision models. **[Assumption — strongly supported by the op table + competing-NPU precedent, not yet board-verified]**
- The BF16 path is a confirmed multi-layer dead end on this toolchain: even with all dims < 65536 the full BF16 body still hits `vnn_VerifyGraph -3 / error 64768`, and every legal BF16↔INT16 bridge attempt (PERMUTE/DataConvert) has failed at export. **[Verified]**
- Block-partitioned multi-NBG (compile each decoder block separately and chain at runtime) is the most credible "make it work at all" fallback, but no public end-to-end per-block-NBG LLM chaining precedent on VIPLite was found, and the acuity-models "Qwen2.5" entries are only HuggingFace links + a JSON viewer, not a runnable per-block recipe. **[Assumption]**

## Key Findings
1. ACUITY exposes a per-channel quantizer (`--quantizer perchannel_symmetric_affine`, called "pcq" in Radxa docs) but per-channel is **gated to certain chip IDs** in the Amlogic SDK ("note only T3(0xBE) can support perchannel quantize"); VIP9000 `0x1000003B` support is undocumented. Per-channel is documented primarily for int8; int16 appears only as a parenthetical option. **[Verified flag; Assumption on chip/int16 support]**
2. **W8A16 is the most promising route** because INT8↔INT16 conversions are integer DataConverts the verifier already accepts; this is exactly how RKLLM ships Qwen on a competing NPU (W8A8 default) and structurally avoids the BF16 wall. **[Assumption — strong]**
3. The `-3 / 64768` signature is confirmed (ST Edge AI) as the VeriSilicon NBG response to exceeding the per-dimension `[0, 65536)` limit; no public source documents a tensor-count or SRAM cap, and the i.MX 8M Plus RM's "network size limit of 2048" is publicly unexplained. **[Verified + gap]**
4. ACUITY auto-inserts DataConvert nodes between mixed-precision ops, but there is **no documented user control** to force a specific two-hop BF16→F16→INT16 placement; inserting explicit ONNX `Cast` ops is the only lever, and ACUITY ignores QDQ scales and may fuse/relocate casts. **[Verified absence of control]**
5. VeriSilicon's acuity-models zoo lists Qwen2.5-0.5B/1.5B/3B as "supported," but ships no NBG and no per-block export recipe — only model references and a JSON viewer. **[Verified]**
6. TIM-VX has a VARIABLE tensor "typically used in recurrent networks to hold recurrent states," but it is marked **InternalOnly** and the recurrent ops are Deprecated; NBG forbids dynamic shapes. No transformer KV-cache path. **[Verified — negative]**

## Details

### Item 1 — PERMUTE/DataConvert dtype walls: forcing a legal BF16→F16→INT16 bridge
**Status: Mostly VERIFIED (negative) + ASSUMPTION on the workaround.**

ACUITY's documented behavior is automatic conversion insertion: in mixed-precision graphs, data-type conversion ops are inserted automatically when precision differs between successive ops. The Radxa/Vivante SDK doc confirms ACUITY "supports UINT8, PCQ (INT8), INT16, BF16 quantization, and mixed quantization" and that "operator fusion is automatically performed."

The failure you hit — BF16 output meeting INT16 at a PERMUTE node, error 65280 — is consistent with the verified TIM-VX dtype-boundary rules: `Transpose`→internal `PERMUTE`, `Slice`→`SLICE`, and `Matmul`→`MATRIXMUL` do not accept mixed BF16/INT16 in/out, and `DataConvert`→`DATACONVERT` does not do a direct BF16→INT16 (legal bridge is BF16→F16→INT16 as two separate nodes).

What I could NOT find in any primary source:
- **No ACUITY pegasus flag** that forces explicit standalone DataConvert/Cast insertion at a chosen tensor boundary, or that controls *where* conversions are placed. The pegasus subcommands are fixed (`import/export/generate/prune/inference/quantize/train/dump/measure/help`); quantize options expose quantizer/qtype/algorithm/hybrid — not per-edge cast placement.
- The only user-facing lever is the ONNX graph. TIM-VX maps ONNX `Cast`→`CAST` and ACUITY maps explicit casts to DATACONVERT, so inserting explicit `Cast` ops (BF16→FP16→INT16 as two separate nodes, positioned so no PERMUTE/SLICE/MATMUL sees a mixed boundary) is the theoretically-correct expression. **[Assumption — plausible from the op-mapping table, but ACUITY's advertised Layer Fusion/Removal/Swapping means a hand-placed cast may not survive to the NBG as placed; conversion placement is owned by ACUITY's optimizer.]**

Conclusion: **Forcing the bridge is not reliably expressible through documented ACUITY controls** — a likely genuine NDA/vendor-support blocker if pursued head-on.

### Item 2 — Residual BF16 export -3 with all dims < 65536
**Status: VERIFIED that -3/64768 = per-dimension limit; OTHER caps UNCONFIRMED.**

The `-3 / "Fatal model generation error: 64768"` signature is confirmed on the same VeriSilicon NBG compiler by the ST Edge AI community: a Dense layer with shape (1500, 384) produced `E [main.c:vnn_VerifyGraph:93]CHECK STATUS(-3:The requested set of parameters produce a configuration that cannot be supported.)` followed by `E ... Fatal model generation error: 64768` and `E010(InvalidModelError): Error during NBG compilation`, fixed by reshaping (1500,384)→(1,1500,384). ST documents the "common constraints": tensors must not be dynamic, ≤ 6D, "dimension must be in the range [0, 65536[", integer types limited to int8/uint8 (int32 bias), no hybrid/un-quantized ops, channel-last output.

Because your chunked-lm_head experiment still hit -3 with all dims < 65536, the dimension cap is necessary but not the only trigger. The candidate other limit is the i.MX 8M Plus Reference Manual's "network size limit of 2048" (p.~5910), which an NXP community thread explicitly questions — "Page 5910 says that there is a network size limit of 2048, but how is the network size defined? Nodes, layers weights, etc.?" — and which received **no public answer**. The 2048 definition is genuinely undocumented in retrievable sources (the RM is gated). No public source documents a tensor-count, NBG-byte, or SRAM working-memory cap for VIP9000.

On newer toolchains: the only public NBG-export escalation thread (Realtek Ameba, ACUITY 6.18.8) was resolved by completing the Vivante IDE install and adding `--optimize 'VIP8000NANONI_PID0XAD' --pack-nbg-unify --viv-sdk ...` flags — environment, not a BF16 fix. I found **no changelog entry** in any ACUITY/Vivante IDE version mentioning a "BF16 multi-layer NBG" fix. Newer public ACUITY exists (6.30.22 in the A733 Docker; 6.21.1 on a Sunplus wiki; NXP eIQ bundles its own VeriSilicon converter), but none advertise a BF16-transformer-NBG fix. **Concluding that a specific newer version fixes the full-BF16 wall is not supportable from public evidence.**

### Item 3 — Per-channel INT16 (not BF16, not single-scale DFP)
**Status: PARTIALLY VERIFIED — flag confirmed, int16 + chip support undocumented.**

ACUITY's per-channel quantizer is real and exposed as `--quantizer perchannel_symmetric_affine`, with `--qtype` accepting `uint8/int8/int16` (Khadas KSNN convert help; pegasus). Radxa docs call this "pcq (int8 per-channel quantized)". The Amlogic `aml_npu_sdk` demo scripts (`1_quantize_model.sh`/`2_export_case_code.sh`) carry the verbatim comment, reproduced across Khadas docs and third-party writeups:
> `#--quantizer dynamic_fixed_point --qtype int8(int16,note s905d3 not support int16 quantize)`
> `# --quantizer perchannel_symmetric_affine --qtype int8(int16, note only T3(0xBE) can support perchannel quantize)`

This is the key risk: **per-channel quantization is gated to specific Vivante chip IDs in the Amlogic toolchain (named: 0xBE / "T3"), and int16 is listed only parenthetically.** No public source confirms per-channel — let alone per-channel *int16* — works on the A733 VIP9000 (`0x1000003B`); the "0xBE" gating is an Amlogic-lineup note, not an Allwinner statement, so applying it to `0x1000003B` is inference. NXP corroborates the VIP9000 hardware bias toward per-tensor: an NXP Community thread citing the i.MX ML Guide states "the NPU provides faster inference for 'per-tensor' quanized models," and the i.MX ML guide notes per-channel models incur a hardware-limitation performance penalty — i.e., per-channel is a software path that may or may not be enabled per chip ID.

Net: per-channel INT16 is the *conceptually ideal* fix (holds channel-concentrated QKV-bias outliers, keeps all boundaries integer/legal, sidesteps BF16 entirely), but **its availability on `0x1000003B` is unverified and possibly chip-ID-gated.** Test empirically: run `--quantizer perchannel_symmetric_affine --qtype int16` and check (a) whether pegasus accepts it for the A733 target and (b) whether NBG export succeeds.

### Item 4 — W8A16 done correctly (the recommended path)
**Status: ASSUMPTION — strongly supported by op-table + competing-NPU precedent.**

The crucial argument: W8A16 = INT8 weights + INT16 activations. Every activation↔activation boundary is INT16↔INT16, and every weight feed is an INT8 weight consumed by a MATMUL/CONV that outputs INT16 — i.e., **only integer dtype boundaries, which the verifier already accepts for shipping vision models** (your YOLO/MobileNet int16 NBGs and the SmolLM2 int16 NBGs prove the integer export path works). This avoids the BF16 export wall by construction. The earlier W8A16 attempt was confounded by SmoothQuant alpha that corrupted even the INT16 control, so the failure was the smoothing, not W8A16.

Minimal-damage recipe (synthesizing SmoothQuant's outlier-migration framing + per-channel-weight best practice):
- **Per-channel symmetric INT8 weights** (weights quantize cleanly; per-channel is standard and low-risk for weights — though note this re-enters the per-channel chip-gating question of Item 3, so have per-tensor INT8 weights as a fallback).
- **Per-tensor INT16 activations** (16-bit holds the ~1790 activation absmax that 8-bit and FP16-5-bit-exponent cannot).
- **SmoothQuant tuned, not default.** Per the SmoothQuant authors (Xiao et al., arXiv:2211.10438), "for models such as OPT and BLOOM, alpha = 0.5 proved to show good results. Whereas if your model has significantly large outliers, a larger alpha value could be used to migrate the quantization difficulty to weight." Qwen2.5-0.5B has *very* large activation outliers (absmax ~1790), so the correct move is **not** to abandon SmoothQuant but to sweep alpha (try 0.6–0.8 to migrate more difficulty into the per-channel INT8 weights, and separately try alpha ≤ 0.2 / none with INT16 activations alone) and pick by host cosine. The prior alpha=0.5 attempt was confounded, not disproven — but because activations are already INT16 (wide), aggressive smoothing is often unnecessary and can hurt; let the host-cosine sweep decide.
- Keep RMSNorm and lm_head at highest integer precision; keep lm_head chunked < 50646 to respect the 65536 dim cap.

Why outliers behave: SmoothQuant (Xiao et al.) states "Outliers appear in a small fraction of the channels. If one channel has an outlier, it persistently appears in all tokens." Dettmers et al. (cited in arXiv:2410.13056) quantify that "retaining a small fraction (less than 1%) of outlier weights in FP16 has been shown to reduce up to 75% of the total quantization error" — confirming the outliers are few, channel-concentrated, and exactly what per-channel weight scaling + 16-bit activations are built to absorb.

Precedent: RKLLM (Rockchip, competing embedded NPU) ships Qwen on integer paths — per airockchip/rknn-llm and Radxa Docs, "Currently supported quantization types include w4a16 and w8a8," producing artifacts like `Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm` (Radxa: "The name indicates this model is W8A8-quantized and targeted for RK3588"). Q-engineering's RK3588 Qwen2.5-VL deployment states verbatim: "All LLM models are quantized to w8a8, while the VLM vision encoders use fp16." Integer weight/activation LLMs are the embedded-NPU industry standard precisely because they avoid float export paths.

TIM-VX boundary verification for INT8↔INT16: the op spec README confirms `MATRIXMUL`, `DATACONVERT`, `CAST`, `SLICE`, `PERMUTE` are all "Mapped," and the SDK lists uint8/int8/int16 as natively-supported quantized types with DATACONVERT as the integer-rescale op. **[Assumption — I confirmed op presence and the integer-type support, not the per-op INT8↔INT16 in/out matrix line-by-line; re-read the TIM-VX op-spec source for the exact INT8↔INT16 legality the same way the BF16 matrix was built, before committing.]**

### Item 5 — Block-partitioned multi-NBG pipeline
**Status: ASSUMPTION — architecturally credible, no public LLM precedent found.**

VeriSilicon's acuity-models zoo lists Qwen1.5/2.5 (0.5B/1.5B/3B), Qwen3-0.6B, Llama, Gemma, DeepSeek-distill, Phi, etc. under "Large Language Model," but the repo ships **only HuggingFace links and a JSON model viewer — no NBG, no per-block export recipe, no chaining code** (verified by fetching the repo tree and README). So the "qwen2.5_7b_decode is a single block" hypothesis cannot be confirmed from the public repo; the zoo is a reference list, not runnable per-block graphs.

The runtime mechanism for chaining exists in principle: NBG is a self-contained binary graph (TIM-VX `NBG` op is "Mapped"), and feeding block N's output tensor into block N+1 is a host-side memcpy between two NBG executions. Whether per-block BF16 (or INT16/W8A16) NBGs export where the monolith fails is **untested in public sources** — but it directly attacks whatever non-dimension/aggregate limit causes the residual -3 (e.g., the unexplained "2048 network size"), because each block is ~1/24th the graph. This is the best "make it work at all" fallback if W8A16 and per-channel INT16 both fail.

### Item 6 — KV-cache / dynamic state
**Status: VERIFIED (negative).**

TIM-VX documents a VARIABLE tensor: "A tensor object which can be used as both input and output for the graph, typically used in recurrent networks to hold recurrent states. Its contents are accessible by the host" (Programming_Guide.md). However, in the op spec README the `VARIABLE` op is marked **InternalOnly**, and the recurrent ops that would consume it (`RNN`, `LSTM`, `LSTMUNIT`, `QUANTIZED_16BIT_LSTM`) are **Deprecated/InternalOnly**. NBG itself requires non-dynamic shapes (ST constraint: "input and output tensors must be not dynamic").

Conclusion: there is no mature, publicly-exposed stateful/variable-length path to implement a true KV-cache on this NBG/VIPLite stack. The fixed-window (W=32/64) recompute approach already working on SmolLM2 remains the only viable decode strategy; a real KV-cache is **effectively blocked without vendor support.**

## Recommendations
Ranked by promise/effort for a coherent, exportable Qwen2.5-0.5B NBG on the A733/`0x1000003B` toolchain.

1. **(Do first — highest promise, modest effort) Correct W8A16.** INT8 weights (try per-channel; fall back to per-tensor if chip-gated) + per-tensor INT16 activations. Sweep SmoothQuant alpha — test {none, 0.2, 0.5, 0.7} — and pick by host cosine, since Qwen's large outliers may need *more* migration to weights, not less. Keep RMSNorm/lm_head at highest int precision; lm_head chunked < 50646. Validate host cosine vs FP oracle BEFORE board, then export.
   - *Proceed-to-board threshold:* host cosine ≥ 0.90 AND NBG export returns status 0.
   - *Abandon threshold:* if pegasus refuses INT8-weight/INT16-activation for this target, OR export still returns -3 with all dims < 65536 → go to #3.

2. **(Do in parallel — cheap to try) Per-channel INT16:** one command, `--quantizer perchannel_symmetric_affine --qtype int16`, to test whether pegasus accepts it for the A733 target and exports. If accepted, this may be the cleanest single-quantizer fix for channel-concentrated outliers.
   - *Kill criterion:* if pegasus rejects per-channel for `0x1000003B` (chip-ID gating per the 0xBE comment), drop immediately — do not invest.

3. **(Fallback — higher effort) Block-partitioned multi-NBG.** Compile each of the 24 decoder blocks as a separate NBG; chain at runtime by passing the hidden-state tensor between executions, in whichever precision passed host validation. This isolates whether the residual -3 is an aggregate-graph/"network size" limit.
   - *Confirming benchmark:* a single decoder block exports cleanly in the precision that failed monolithically.

4. **(Only if 1–3 fail) Escalate to VeriSilicon/Allwinner under NDA.** The two genuine blockers — (a) forcing DataConvert placement around PERMUTE, and (b) the undocumented "network size 2048"/aggregate NBG limit plus per-channel chip-ID gating — are owned by ACUITY's closed optimizer and the gated reference manual, and are not resolvable from public sources.

Do **not** spend further effort on: any full-BF16 body graph; BF16↔INT16 boundaries on PERMUTE/SLICE/MATMUL; or waiting for a newer ACUITY version to fix BF16 (no such changelog exists publicly).

## Caveats
- "W8A16 avoids the BF16 wall" is an inference from the op table + competing-NPU precedent, not a board-verified result on `0x1000003B`. Re-read the per-op INT8↔INT16 in/out legality from TIM-VX op-spec source line-by-line before committing.
- Per-channel support on `0x1000003B` is unverified and may be chip-ID-gated; the only explicit gating statement names chip 0xBE / "T3" in the Amlogic SDK, not Allwinner. Test empirically; keep per-tensor INT8 weights as fallback.
- SmoothQuant alpha=0.5 is the paper's standard balance point, not inherently "too aggressive"; the prior failure was confounded. The right answer is a host-cosine alpha sweep, and for very-large-outlier models the authors suggest a *larger* alpha to migrate difficulty into the (16-bit-protected) weights.
- The "network size limit of 2048" is from a gated NXP reference manual; its definition (layers/nodes/tensors) is unconfirmed, so block-partitioning's effectiveness against it is a hypothesis.
- acuity-models' LLM "support" is a reference list of HF models plus a JSON viewer, not runnable per-block NBG recipes; treat any "VeriSilicon officially supports Qwen2.5 on NPU" reading with caution.

## Updated "confirmed dead ends" list (do not re-try)
Previously established (unchanged): plain INT16 Qwen (incoherent, cosine 0.236); full FP16 (incoherent, 0.541); unchunked full BF16 (export -3/64768); chunked BF16 lm_head alone (export -3); chunked lm_head + embedding (DataConvert error 65280); hybrid BF16-top-layers (host 0.254 + PERMUTE error 65280); BF16/INT16 boundaries on SLICE/MATMUL/PERMUTE; QDQ-ONNX import (ACUITY ignores QDQ scales); TVM vsi_npu / hand-built TIM-VX (same verifier, missing RMSNorm/gather/slice/SwiGLU coverage).

Newly added from this investigation:
- **Forcing ACUITY to materialize a chosen BF16→F16→INT16 two-hop bridge via a documented pegasus flag** — no such control exists; conversion placement is owned by ACUITY's optimizer (no flag in import/export/generate/quantize). Not expressible without vendor support.
- **Expecting a newer public ACUITY/Vivante IDE version to fix the multi-layer BF16 NBG wall** — no public changelog mentions such a fix; the only NBG-export escalation thread (Ameba, 6.18.8) was an environment/IDE-install issue.
- **Relying on acuity-models qwen2.5 zoo entries as a runnable per-block recipe** — the repo ships only HF links + a JSON viewer, no NBG/recipe.
- **A real KV-cache via TIM-VX VARIABLE tensor / recurrent ops** — VARIABLE is InternalOnly and recurrent ops Deprecated; NBG forbids dynamic shapes. Blocked without vendor support.
- **(Caution, not yet a hard dead end) Per-channel quantization on `0x1000003B`** — possibly chip-ID-gated to 0xBE per the Amlogic SDK comment; test once cheaply (Rec #2) and, if rejected, record as a confirmed dead end for this chip.