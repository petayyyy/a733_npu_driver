# Getting a Coherent, Exportable Qwen2.5-0.5B NBG on ACUITY 6.30.22 / VIPLite 2.0.3.2 / VIP9000 (Allwinner A733)

## TL;DR
- **The most likely single fix: the `vnn_VerifyGraph` status -3 (error 64768) is a per-tensor dimension limit, not a BF16 bug.** A VeriSilicon NBG-verifier constraint documented as "dimension must be in the range [0, 65536[" is almost certainly tripped by Qwen's **151936-wide vocab projection in BF16**. Split/tile the lm_head (and the embedding gather) into chunks ≤ 65535 wide so the BF16-heavy graph passes verify — this is the highest-promise, lowest-effort next action, and chunking projections under a 65536 cap is an independently established NPU workaround.
- **Mixed BF16-body + INT16-logits is the right architecture, but the dtype boundary must be an explicit `DataConvert` node.** VeriSilicon's own runtime inserts `DataConvert` at any dtype mismatch; the illegal-boundary errors at SLICE/MATMUL mean those ops were asked to consume mixed dtypes directly. Put a DataConvert/Cast between the BF16 and INT16 regions rather than letting a SLICE or MATMUL straddle them.
- **Confirmed dead ends:** full-BF16 export (verify -3), full-FP16 quality (cos 0.541), plain INT16 Qwen (cos 0.236), and the three mixed splits as configured. RKLLM succeeds on Qwen2.5 because it uses grouped/per-channel W8A8/W4A16 + GPTQ + fp16 lm_head — a recipe ACUITY's single-scale DFP cannot replicate, which points to per-channel (`pcq`) or hybrid as the quantization fix.

## Key Findings

1. **The BF16 export failure is a graph-verifier shape constraint, not "BF16 is unsupported."** The exact error string — `E [main.c:vnn_VerifyGraph:93] CHECK STATUS(-3 ...) ; Fatal model generation error: 64768` — appears verbatim in the STMicroelectronics Community thread "Issue with TensorFlow Dense Layer Conversion to .nb Format" (community.st.com/t5/edge-ai, td-p/777751, marked Solved), where it was caused purely by a tensor dimension and resolved by reshaping. ST documents the relevant Vivante NBG constraint as "dimension must be in the range [0, 65536[." Qwen's vocab is 151936 > 65536. *(verified error + constraint; inference that 151936 is the trigger)*
2. **FP16 exporting fine at 151936 while BF16 fails is the one piece the pure-dimension theory does not fully explain** — so there is likely a secondary BF16-specific size/tiling/buffer interaction on top of the dimension limit. This remains an assumption; no primary source confirms a BF16-only byte/2 GB limit.
3. **`DataConvert` is the canonical, hardware-supported dtype-conversion op** and is exactly how VeriSilicon's stack bridges precision boundaries. The SLICE and MATMUL op_check failures occur because those nodes were handed two different dtypes directly; the fix is an explicit converter node at the boundary. *(verified)*
4. **VIP9000 natively supports hybrid quantization** ("mixing data formats between neural network operations"), and ACUITY exposes it via `--hybrid` + `customized_quantize_layers` in the `.quantize` file, producing a new `quantize.json` graph structure. *(verified)*
5. **No coherent low-bit Qwen2.5 on any VIP9000 NPU was found in public sources;** the working precedent (RKLLM on Rockchip) relies on grouped W8A8/W4A16 + GPTQ + per-channel scaling + fp16 embedding/lm_head — capabilities ACUITY's single-scale INT16 DFP lacks but its `pcq` (per-channel int8) and hybrid modes partly provide. *(verified for RKLLM; assumption for portability)*
6. **The TVM `vsi_npu` BYOC path is immature for this use case** — it targets quantized CNNs (qnn.conv2d/requantize), has open build/runtime issues, and shows no transformer/large-vocab LLM precedent. Because it emits NBG through the same Vivante compiler, it is unlikely to bypass the same `vnn_VerifyGraph` limits. *(verified maturity; assumption about shared verifier)*

## Details

### Item 1 — Is `vnn_VerifyGraph` status -3 on BF16-heavy graphs a known limitation?

**VERIFIED (primary source).** The identical error signature was reported on the STMicroelectronics Community thread "Issue with TensorFlow Dense Layer Conversion to .nb Format" (community.st.com/t5/edge-ai, td-p/777751), which wraps the same Vivante NBG compiler as ACUITY/VIPLite:

> `E [main.c:vnn_VerifyGraph:93]CHECK STATUS(-3:The requested set of parameters produce a configuration that cannot be supported.)` … `E 21:37:15 Fatal model generation error: 64768` … `E010(InvalidModelError): Error during NBG compilation, model is not supported`

The user's Dense layer with input shape (1500, 384) failed; (1, 1500, 384) worked. The ST technical moderator cited the Vivante NBG "Common constraints," including:

> "input and output tensors must be not dynamic … must not be greater than 6D … **dimension must be in the range [0, 65536[** … mixed data operations (i.e hybrid operator) are not supported, activations and weights should be quantized"

**Application to Qwen (assumption, strongly supported):** Qwen2.5-0.5B's output projection / logits tensor is **151936 wide**, which **exceeds the documented 65536 per-dimension limit**. A 151936-extent tensor would be expected to produce exactly this `vnn_VerifyGraph` -3 / error-64768 failure. The absence of an emitted node name is consistent with a global graph-configuration check rather than a per-op datatype check.

**Independent corroboration that 65536 is a real NPU per-dimension cap and that chunking is the standard fix (verified):** The vLLM-Ascend project documents the same class of limit and remedy in its release notes — "Chunked wq_b matmul to work around the NPU 65536 dimension limit (#9780)." This establishes that a 65536 per-dimension cap on matmul/projection tensors is a known NPU constraint and that vocab/projection chunking under that cap is an established engineering fix, not speculation.

**Open puzzle (assumption):** The project reports full-**FP16** exports fine at 151936 (991 MB .nb) while full-**BF16** fails verify. A pure per-dimension limit would block both equally, so there is likely an additional BF16-specific code path or tiling/buffer-size interaction. No primary source confirms a BF16-only or per-byte/2 GB limit; the NXP i.MX 8M Plus reference manual is noted by community users (community.nxp.com, "NPU specifications i.MX 8M Plus") to contain a "network size limit of 2048" (units undefined, likely max nodes/layers — a different constraint). This secondary BF16 effect is unverified.

**Where to get a newer toolchain (verified availability, unverified that it fixes -3):** ACUITY 6.30.22 is the Allwinner/Radxa-bundled version (Docker `ubuntu-npu:v2.0.10.1`). Newer public Vivante toolchains exist via: NXP eIQ (bundles a VeriSilicon converter; i.MX Machine Learning User's Guide rev LF6.18.2), Khadas/Amlogic `aml_npu_sdk`, and the Radxa/ZIFENG278 `ai-sdk` mirror. There is no public changelog entry confirming a VerifyGraph-3 large-vocab fix in any newer release.

### Item 2 — Legal BF16↔INT16(DFP) boundaries at SLICE and MATMUL; how to insert a converter

**VERIFIED:** `DataConvert` is the TIM-VX op that changes tensor format ("Change the format from input tensor to output tensor"), and `Cast` is the variant that ignores scale/zeroPoint. VeriSilicon's own tflite-vx-delegate inserts `DataConvert` automatically whenever a tensor's datatype differs from the consuming op's target datatype (op_map.cc creates a `tim::vx::ops::DataConvert` when `datatype_ != target_dtype`). EVIS/CL kernels are generated per explicit dtype-pair (e.g. `hswish_BF16toBF16`, `Cast.shape…_fp32_to_int32`), confirming dtype conversion is a first-class, kernel-backed operation.

**Interpretation of the project's two boundary errors:**
- `vsi_nn_op_strided_slice.c:op_check:507 … BFLOAT16, DFP INT16 … Check node SLICE fail` — the SLICE node was given a BF16 input and a DFP-INT16 output (or vice-versa). SLICE/StridedSlice is a data-movement op and does not requantize; it requires input and output to be the same dtype.
- `vsi_nn_op_matrixmul.c:op_check:130 … DFP INT16, DFP INT16, BFLOAT16 … Check node MATRIXMUL fail` — the MATMUL was asked to take INT16 inputs and emit BF16 directly. The matmul op_check rejects mixed in/out dtype combinations.

**The fix (verified mechanism; placement is engineering):** Do **not** let a SLICE or MATMUL straddle the precision change. Insert an explicit `DataConvert` (quant-aware, respects DFP scale) or `Cast` node so the boundary is its own op: …BF16 → DataConvert → INT16(DFP) → SLICE(INT16→INT16)…; and for the matmul, run an all-INT16 matmul and convert the INT16 result to BF16 with a dedicated DataConvert node *after* it, rather than asking the matmul to output BF16.

**Important caveat (verified):** ST's mirror of the constraint explicitly states "mixed data operations (i.e hybrid operator) are not supported" — a single op cannot mix dtypes; the conversion MUST be a separate node. This is consistent with the op_check failures.

**Auto-insertion in ACUITY (assumption, strongly supported):** ACUITY's `--hybrid` step "changes the model structure" and emits a new `quantize.json`, and VIP9000 "supports hybrid quantization natively." Combined with the delegate auto-creating DataConvert at mismatches, the strong inference is that ACUITY's hybrid path *does* auto-insert convert nodes at precision boundaries — and that the project's manual mixed-splits failed because the split points were placed *on* a SLICE/MATMUL rather than letting ACUITY's hybrid flow choose boundaries. No verbatim VeriSilicon sentence confirming auto-insertion was found.

### Item 3 — Making the 151936-wide projection exportable

Given Item 1, the single most promising structural change is to **eliminate any tensor with an extent > 65536**:
- **Tile/split the lm_head matmul** into N chunks of vocab ≤ 65535 (e.g. 3 × ~50646, or 4 × 37984), each a separate MATMUL producing a ≤65535-wide logits slice, then concatenate (or argmax per chunk and combine). This keeps every tensor dimension under the documented limit while allowing the body to stay BF16. This mirrors the vLLM-Ascend "chunked matmul to work around the NPU 65536 dimension limit" pattern. *(assumption for ACUITY — directly derived from the verified 65536 constraint plus the Ascend precedent)*
- **Likewise split the token-embedding gather** if it materializes a 151936-extent tensor.
- Keeping the lm_head in **INT16 DFP** while the body is BF16 is viable *only if* a DataConvert node sits at the body→head boundary (Item 2). The logits saturation observed in plain INT16 (fl=10, range ±32 vs max_abs_diff 34) is a *separate* dynamic-range problem; per-tensor DFP cannot hold both the outlier body and the logits. Splitting the vocab does not fix dynamic range, so the head likely still needs BF16 or per-channel treatment.

### Item 4 — Quantization that handles outliers AND exports

- **ACUITY supports `pcq` = `perchannel_symmetric_affine` (per-channel INT8).** Per-channel scaling is the standard outlier remedy — it is what per-channel KV/weight methods use against Qwen's large QKV-bias-driven outlier channels. KVQuant (Hooper et al., arXiv:2401.18079) reports that "By leveraging per-channel quantization for Keys and per-token quantization for Values, we observe a 3.82 perplexity improvement on Wikitext-2 for 3-bit LLaMA-7B quantization." ACUITY does **not** offer per-channel INT16; INT16 is single-scale `dynamic_fixed_point` only. *(verified)*
- **ACUITY KLD + `--MLE` and `--hybrid`** are the documented in-toolchain accuracy levers: compute per-layer entropy (`--compute-entropy`, writes `entropy.txt`, range [0,1], higher = worse), then promote the highest-entropy layers to higher precision via `customized_quantize_layers`. This is the *intended* mechanism to keep outlier-heavy layers in BF16 while the rest stay INT16 — and crucially, it lets ACUITY (not you) place the convert nodes. *(verified)*
- **Qwen's outliers are caused by large QKV bias terms** — a documented Qwen-family property. Tu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models" (arXiv:2504.04823), states: "the bias terms of key projection layers can be extremely large in the pre-trained Qwen-1.5B and 7B models, e.g. the maximum absolute value in key projection bias terms reaches 402 in Qwen-1.5B." This concentrates outliers in a few channels, making **per-channel** (not per-tensor) quantization the correct tool. *(verified for the mechanism)*
- No public example of a coherent low-bit Qwen2.5 on any VIP9000 NPU was found. *(verified absence)*

### Item 5 — RKLLM cross-check (why Rockchip succeeds)

Per the airockchip/rknn-llm Releases changelog and DeepWiki, RKLLM ships coherent Qwen2.5-0.5B/1.5B/3B with: "Group-wise quantization: Various group sizes (32/64/128 for w4a16, 128/256/512 for w8a8); GRQ Int4: Enhanced 4-bit quantization algorithm; GPTQ-Int8: Support for GPTQ quantized models; Mixed quantization: Combination of grouped and non-grouped quantization," plus a "gdq algorithm to improve 4-bit quantization accuracy" and "support for converting HuggingFace GPTQ-int4 models (requires groupsize to be 32, 64, or 128, and desc_act set to false)." *(verified)* Conceptual portability to ACUITY:
- **Grouped/per-channel scaling** is the key outlier handler RKLLM has and ACUITY's single-scale INT16 lacks — ACUITY's nearest equivalent is `pcq` (per-channel int8) or hybrid promotion.
- RKLLM keeps numerically sensitive pieces (embedding, lm_head, norms) at higher precision — mirroring the recommendation to keep Qwen's lm_head/outlier layers in BF16 via hybrid.
- RKLLM's runtime/compiler is purpose-built for decoder LLMs with KV-cache; ACUITY/VIPLite is a general CNN/transformer offline compiler, which is why the fixed-window (no-KV-cache) builder is being used here.
- **Not portable as tooling:** RKLLM's toolkit and runtime are Rockchip-specific and cannot run against VIP9000; only the *recipe* (grouped/per-channel + GPTQ + fp16 head) transfers conceptually.

### Item 6 — TVM `vsi_npu` / TIM-VX direct path

**VERIFIED maturity assessment:** The `vsi_npu` TVM fork registers a limited set of quantized operators/patterns (qnn.conv2d, qnn.requantize, bias_add) and generates NBG via the same TIM-VX backend. Public issues show build/cross-compile breakage (TIM-VX #195) and runtime "PLS isn't existed" failures (#189). It is CNN-oriented; **no transformer or large-vocab LLM precedent exists.** Because it emits NBG through the same Vivante compiler, it would hit the **same `vnn_VerifyGraph` constraints**, so it is unlikely to bypass the -3 error. Hand-built TIM-VX gives finer control over inserting DataConvert nodes but inherits the identical verifier and the 65536 limit. *(maturity verified; verifier-shared assumption)*

## Recommendations

**Ranked by promise / effort:**

1. **(Highest promise, moderate effort) Split the 151936-wide lm_head (and embedding) into ≤65535-wide chunks, then re-attempt the BF16-body export.** This directly targets the verified `[0, 65536[` dimension constraint that is the most likely cause of error 64768, and matches the documented vLLM-Ascend "chunked matmul to work around the NPU 65536 dimension limit" precedent. **Benchmark/threshold:** if full-BF16 (or BF16-body + chunked head) then passes `vnn_VerifyGraph`, the dimension limit was the blocker. If it still fails -3 with all dims < 65536, the secondary BF16 hypothesis (Item 2 puzzle) is confirmed and you escalate to step 5.
2. **(High promise, moderate effort) Use ACUITY's native `--hybrid` flow instead of hand-placed splits.** Run `--compute-entropy`, inspect `entropy.txt`, and promote the highest-entropy (outlier) layers — and the lm_head — to BF16 in `customized_quantize_layers`, leaving the rest INT16. Let ACUITY emit the new `quantize.json` and insert its own convert nodes. This avoids the manual SLICE/MATMUL boundary errors entirely. Combine with step 1 (chunked head) if the head is promoted to BF16.
3. **(Promising, low effort) Where you must hand-place a BF16↔INT16 boundary, insert an explicit `DataConvert` (quant-aware) or `Cast` node so no SLICE/MATMUL straddles the dtype change.** Convert *before* SLICE and *after* an all-INT16 MATMUL. This is the verified mechanism behind the op_check failures.
4. **(Outlier fix, moderate effort) Try `pcq` (per-channel INT8) on the linear layers**, optionally as W8 weights with INT16 activations via hybrid, to handle Qwen's channel-concentrated QKV-bias outliers — the same lever RKLLM uses. Combine with SmoothQuant at a *gentle* alpha (the earlier over-smoothing failure suggests α≈0.5 or lower with per-channel weight scaling, not aggressive activation migration).
5. **(Fallback, high effort) If steps 1–4 all fail verify, obtain a newer Vivante toolchain via NXP eIQ or the Amlogic/Radxa SDKs and re-test**, but treat a fix as unconfirmed — no public changelog documents a large-vocab/BF16 VerifyGraph fix. This is also the point to open a VeriSilicon support ticket with the exact (chunked, all-dims-<65536) BF16 graph that still fails.
6. **(Do last / research) TVM `vsi_npu` or hand-built TIM-VX** only if you need graph-level control to place DataConvert nodes that ACUITY won't; expect the same verifier limits and no LLM precedent.

**Threshold that changes the plan:** If chunking the head below 65536 makes a BF16 graph export, prioritize a fully-BF16-body + chunked-head model and accept the ~1 GB .nb size. If it does not, the blocker is BF16-specific and the per-channel/hybrid INT16 path (steps 2–4) becomes the primary route to a coherent *and* exportable model.

## Caveats
- The claim that Qwen's 151936 vocab triggers error 64768 is an **inference** from a verified constraint observed on the same Vivante NBG verifier via ST Edge AI and corroborated by the vLLM-Ascend 65536 chunking precedent; it has not been directly reproduced on ACUITY 6.30.22 in a primary source.
- The exact legal BF16↔INT16(DFP) `DataConvert` dtype-pair table could not be retrieved from TIM-VX source; verify by grepping `vsi_nn_op_dataconvert.c` and the kernel map locally. DataConvert is confirmed to exist and to be auto-used at dtype mismatches, but the specific BF16→DFP single-kernel path is unconfirmed (conversions may route via FP32/FP16).
- Whether ACUITY auto-inserts convert nodes at hybrid boundaries is strongly implied (the `--hybrid` step "changes the model structure"; the delegate auto-creates DataConvert) but not stated verbatim in any VeriSilicon doc.
- RKLLM portability is conceptual; RKLLM's toolkit and runtime are Rockchip-specific and cannot be run against VIP9000.
- "Full FP16 exports but full BF16 fails" is an unexplained asymmetry; resolving it definitively may require VeriSilicon support (potentially an NDA toolchain) if steps 1–4 fail.
- The NXP i.MX 8M Plus "network size limit of 2048" is a separate, differently-defined constraint (likely node/layer count) and was not confirmable from the reference manual itself.