# INT8/INT4 Quantization on the Allwinner A733 VIP9000 (ACUITY/VIPLite): Root-Cause Analysis and Recommended Path

*Senior research-engineer technical report. Every claim is tagged **verified** (a primary source was found) or **assumption** (researcher inference). Throughout, "this actually works on this exact NPU/toolchain" is kept distinct from "general LLM-quant theory that may not be expressible here."*

## TL;DR
- **The int8 blocker is a quantization-quality + tooling problem, not a hardware problem.** The VIP9000 already runs the full SmolLM2-135M transformer in int16, and ACUITY's own *host* int8 simulation diverges from FP before the NPU is even involved — so naive per-channel int8 PTQ destroying coherence of a small LLM is expected behavior, and the ACUITY hybrid-quantize zero-byte bug is almost certainly an ACUITY serializer/scaling defect (large-vocab / many-`dtype_converter`), not a documented known issue (assumption; no public report exists at all).
- **The highest-promise route is neither ACUITY's broken hybrid path nor QDQ-ONNX import.** It is to reproduce the proven RKLLM idea inside what ACUITY can actually express: **per-channel int8 weights + int16 activations (W8A16)**, with SmoothQuant-style activation smoothing done **offline in PyTorch before ONNX export**, and attention/softmax/RMSNorm/embedding/lm_head held at int16.
- **The pre-quantized-ONNX (QDQ) import route (B.3) is a confirmed dead end for ONNX.** The only documented "import-preset-scales" channel is a quantized **TFLite** model (acuitylite `TFLiteLoader`, "no need to quantize"), which is per-tensor activation / per-channel weight int8 only and does **not** carry GPTQ/AWQ int4 schemes.

## Key Findings
- **VIP9000 hardware natively supports INT8/INT16/FP16/BF16 and native hybrid quantization** (verified, verbatim from VeriSilicon's Aug 7, 2019 launch release, verisilicon.com/PressRelease/VIP9000): *"VIP9000 enables neural network inference with different data formats based on design choice (INT8, INT16, Float16, Bfloat16). VIP9000 also supports hybrid quantization (mixing data formats between neural network operations) natively."* It *"adopts Vivante's latest VIP V8 NPU architecture."*
- **ACUITY exposes `int4`/`uint4` as quantize-target qtypes** in its CLI, but only as PTQ output targets; there is no public evidence the A733 VIPLite 2.0.3 runtime executes a w4a16 LLM (verified the qtype strings exist; verified dead end on a working A733 int4 LLM).
- **TIM-VX has a VARIABLE tensor type** explicitly (verified, verbatim from VeriSilicon/TIM-VX `docs/Programming_Guide.md`): *"VARIABLE Tensor: A tensor object which can be used as both input and output for the graph, typically used in recurrent networks to hold recurrent states. It's contents are accessible by the host."* — the one stack mechanism relevant to a KV-cache, though NBG graphs remain static-shape.
- **The ONNX Runtime VIPLite EP (issue #28244) is only a feature request, now bot-marked stale**; no usable EP exists (verified). The issue (opened by user evgen-pervenenko, Apr 27, 2026) specifies the target as *"VeriSilicon VIP9000, 3 TOPS, supports INT8/INT16/FP16/BF16; … device at /dev/vipcore; Userspace library: libVIPhal.so (VIPLite API)."*
- **etnaviv/Teflon supports only conv/add/ReLU on the Vivante NPU; no attention/transformer ops** (verified). Maintainer Tomeu Vizoso (tomeuvizoso.net) confirms the backend supports only addition and convolution (ReLU only as a fused convolution activation), and that the freed Allwinner NPU docs *"are about how to use the downstream drivers"* — not the VeriSilicon hardware needed for new ops.

## Details

### PRIMARY A — ACUITY hybrid-quantize zero-byte table bug

**What was observed (project-reported, treated as ground truth):** ACUITY 6.30.22's hybrid/w8a16 path (`pegasus_quantize_hybird.sh`, `--hybrid --compute-entropy --quantizer perchannel_symmetric_affine --qtype int8`) reaches "End quantization… Dump net quantize tensor table", writes the intermediate `<name>_pcq.quantize.JSON` (589 `dtype_converter` ops, int16↔int8 and int16↔fp32), then **truncates the final YAML `<name>_pcq.quantize` to 0 bytes and hangs CPU-active forever** — reproducible with both a fresh seed pass and an existing calibrated seed table.

**Is this a known/documented ACUITY bug? No (verified dead end).** I searched VeriSilicon/acuity-models, VeriSilicon/TIM-VX issues, NXP eIQ / i.MX community, linux-sunxi, aw-ol, whycan, and the Khadas forum and found **no public report** of the hybrid-quantize zero-byte/truncated `.quantize` table or the JSON→YAML dump hang. This is consistent with hybrid quantization being a rarely exercised path that essentially no community user has run on a large-vocab transformer.

**Most probable root cause (assumption, reasoned from evidence):** The failure signature — a complete JSON dump (589 `dtype_converter` ops) followed by a 0-byte YAML and an infinite CPU-active loop — is characteristic of a serializer choking on graph scale. SmolLM2-135M's 49,152-row vocab embedding and lm_head, plus hundreds of inserted int16↔int8/fp32 converters, produce an unusually large hybrid tensor table. A hang *after* JSON but *before* YAML is the signature of either (a) an O(n²) consistency/entropy pass over the converter list, or (b) a YAML emitter that fails silently (truncates to 0 bytes) on a structure the JSON emitter handled. This is an ACUITY tooling defect, not an NPU or model-correctness issue.

**Workarounds / exits (ranked):**
1. **Bypass the YAML dump by reconstructing it from the JSON.** The intermediate `_pcq.quantize.JSON` is already complete. There is **no documented ACUITY flag** to "export from `.quantize.json`" (assumption — not found), but because the YAML `.quantize` and the JSON encode the same tensor table, the lowest-effort potential unblock is to hand-author/repair the YAML `.quantize` (the documented artifact consumed by `pegasus_export_ovx.sh`) from the JSON.
2. **Reduce the `dtype_converter` count.** Each precision boundary inserts a converter; the suspected trigger is the converter count the serializer must emit. Constraining hybrid to fewer precision transitions — keep the whole transformer body at one precision and convert only at the embedding/lm_head edges — shrinks the table dramatically and directly attacks the suspected cause.
3. **Get a newer ACUITY.** The public pegasus CLI help (mirrored for acuity-toolkit 6.6.1) already lists `int4`/`uint4` and a richer quantize surface, and Radxa's quant-accuracy page references `acuity-toolkit-whl-6.30.22`. Newer ACUITY ships inside **NXP eIQ** (which bundles a VeriSilicon converter for the i.MX 8M Plus VIP9000) and via **Realtek Ameba AIoT's offline toolkit** (gated by email request to AmebaAIoT@realtek.com). These are the realistic public/semi-public channels for a newer ACUITY than Allwinner's `ubuntu-npu:v2.0.10.1` Docker (verified these channels exist; not verified that they fix the bug).

### PRIMARY B.1 — Has anyone run a coherent int8/int4 LLM on a VIP9000-class NPU?

**No public, verified case of a coherent low-bit LLM on any VIP9000/VIPNano NPU (verified dead end).** The only documented on-NPU transformer-adjacent work on this exact A733/VIP9000 is the Frigate object-detection effort (CNN heads; the author found *"uint8 dropped a real-scene car from ~0.80 → ~0.50… int16 keeps ~float accuracy"*), plus the project's own SmolLM2-135M int16 result. NXP's community states transformers *"are supported, however, the performance really depends upon the model architecture"* on the i.MX 8M Plus VIP9000 — but provides no coherent-LLM int8 demo. All *working* low-bit small-LLM references live on **other** NPUs (Rockchip RKLLM, Qualcomm QNN/Hexagon, Intel NPU via OpenVINO, Hailo). The project is therefore at or near the frontier for LLMs on this exact silicon.

### PRIMARY B.2 — Mapping standard PTQ techniques to ACUITY/TIM-VX

| Technique | Expressible on ACUITY/TIM-VX? | Notes |
|---|---|---|
| Per-channel symmetric int8 **weights** | **Yes (verified)** — ACUITY "pcq" is per-channel int8 | Native sweet spot. |
| Per-tensor static int8 **activations** | **Yes (verified)** | Standard ACUITY PTQ activation path. |
| **int16 activations (W8A16)** | **Yes (verified)** — int16 runs on hardware today | The single most important lever; sidesteps the activation-outlier problem that kills int8. |
| **Per-token / dynamic** activation quant | **No (assumption)** — NBG is static-shape/static-quant; NPUs are "architected for static quantization" | Cannot compute per-token scales on the fly in an NBG. |
| **SmoothQuant** (offline activation→weight migration) | **Partial — only if done OFFLINE** before ONNX export | Smoothing is a weight re-parameterization + inserted scale; result is standard ops ACUITY can quantize. Key portable idea. |
| **AWQ** (per-channel weight scaling by activation salience) | **Partial — offline** | Scaling folds into weights pre-export; the int4 grouping does not map. |
| **GPTQ** (Hessian-based weight rounding) | **Weights only, if re-imported as plain int** | Better int weights, but ACUITY must own the final scale representation; QDQ re-import unsupported (B.3). |
| **LLM.int8() outlier split / mixed FP16+INT8** | **No (assumption)** | Requires runtime dynamic decomposition; not expressible in a static NBG. The CPU-offload "shadow outlier" trick (llm.npu, arXiv 2407.05858) needs a CPU/NPU split this flow doesn't provide. |
| **Per-group (block) weight quant (gs 32–128)** | **No public evidence** | ACUITY pcq is per-channel, not per-group. |

### PRIMARY B.3 — Can ACUITY import a pre-quantized (QDQ) ONNX and preserve its params? **VERDICT: NO — confirmed dead end for ONNX.**

This was the highest-priority strategic question, and the answer is decisive (verified across the ACUITY/acuitylite docs and CLI reference):
- **The pegasus importer subcommands are `caffe/tensorflow/tflite/darknet/onnx`; all quantization flags** (`--qtype`, `--quantizer`, `--hybrid`, qtypes including `int4`/`uint4`) **live under the `quantize` subcommand, not `import`.** `--qtype` selects ACUITY's *own* PTQ output target — it is not a mechanism to ingest externally computed scales.
- **Every documented ACUITY/acuitylite ONNX workflow** (Radxa A7A resnet50/yolov5/yolo26 examples; the acuitylite ONNX demo) **feeds a float ONNX + calibration dataset, then runs ACUITY's own PTQ.** The acuitylite ONNX demo explicitly calls `Quantization(model).quantize(...)` on a float model.
- There is **no public evidence** that `pegasus_import` parses QuantizeLinear/DequantizeLinear nodes or preserves their scales/zero-points, and **no flag** to do so. (Whether it errors *specifically* on QDQ nodes is genuinely unknown — not documented either way. The Radxa tip *"If the source model is already quantized, no additional quantization is needed here, otherwise it will cause an error"* is, in full context, a workflow note that appears next to a float Keras example, not a spec that the ONNX importer ingests QDQ scales.)
- **The one supported "preset-scales" route is TFLite, not ONNX.** acuitylite's TFLite demo states verbatim: *"No need to quantize using acuity lite for quantized model"* — it imports a pre-quantized int8/uint8 TFLite (per-tensor activation scales, per-channel weight scales, weight zero-point forced to 0) and **skips ACUITY PTQ entirely**, then exports to TIM-VX/NBG. This is per-tensor/per-channel int8/uint8 only; it does **not** carry GPTQ/AWQ int4 QDQ schemes.

**Implication:** Do not build a "GPTQ/AWQ → QDQ ONNX → ACUITY import" pipeline; the toolchain will not honor those parameters. If you want to set scales yourself, the only honored channel is a **quantized TFLite** (int8 per-tensor activation / per-channel weights). That is a viable but narrow lever — worth a spike specifically because it lets you carry SmoothQuant-smoothed, externally calibrated int8 scales into the NPU without trusting ACUITY's calibrator.

### PRIMARY B.4 — RKLLM as the reference working recipe, and what's portable

RKLLM (working on RK3588/RK3576) ships **w8a8** and **w4a16** for coherent small-LLM output via `rkllm-toolkit`. Its essence (verified from Rockchip/DeepWiki/ArmSoM docs and the airockchip/rknn-llm release notes):
- **Per-channel weights + grouped activation/weight quant**, with selectable group sizes — verbatim from the release notes: *"Added support for grouped quantization (w4a16 group sizes of 32/64/128, w8a8 group sizes of 128/256/512). Added gdq algorithm to improve 4-bit quantization accuracy."*
- A **GRQ/gdq int4** algorithm and **GPTQ-int8** support (GPTQ-int4 import requires group size 32/64/128 and `desc_act=false`), plus **hybrid (mixed grouped/non-grouped) quantization** by ratio.
- Calibration data generated by `generate_data_quant.py`; the broader-ecosystem W8A8 recipe is SmoothQuant + per-channel weights + (on GPUs) per-token dynamic activations.

**Portable to ACUITY/TIM-VX:**
- ✅ Per-channel int8 weights (ACUITY pcq) — direct match.
- ✅ Offline SmoothQuant-style activation smoothing (re-parameterize before ONNX export) — directly portable; it's just weight scaling.
- ✅ Keeping activations at int16 (the conservative analogue of RKLLM's "a16" in w4a16) — the single most important transfer.
- ❌ Grouped (block) weight quantization and on-device int4 — not expressible in public ACUITY.
- ❌ Per-token dynamic activation scales — not expressible (static NBG).

**Strategic lesson:** RKLLM gets coherence at low bit-width mainly from **grouping + keeping activations wide (a16) + good calibration**, not from per-channel int8 alone. The portable subset for this toolchain is "**per-channel int8 weights + int16 activations + offline smoothing**" = **W8A16**. This matches the project's own observation that pure pcq int8 yields garbage while int16 is coherent.

## Secondary Questions

**1) INT4 / w4a16 on this exact toolchain.** ACUITY's CLI lists `int4`/`uint4` as quantize qtypes, and VeriSilicon markets INT4 for VIP9000. But there is **no public evidence** that the A733's VIPLite 2.0.3 runtime or the `ubuntu-npu:v2.0.10.1` ACUITY compiles and runs a w4a16 LLM end-to-end (verified the qtype string exists; verified dead end on a working A733 int4 LLM). Treat int4 as **"marketed, not demonstrated"** on this SoC; w4a16 is demonstrated only on Rockchip and larger/other VeriSilicon classes.

**2) Dynamic shapes / KV-cache.** NBG graphs are static-shape, forcing full-window recompute (the throughput ceiling). The one relevant mechanism is TIM-VX's **VARIABLE tensor** — host-accessible, usable as both input and output, documented as *"typically used in recurrent networks to hold recurrent states"* (verified). In principle this could hold a fixed-size KV buffer updated between fixed-shape executions — but it gives no dynamic sequence length, and there is **no public example** of a KV-cache transformer built this way on VIP9000. TVM's vsi_npu/Relay path also has "extremely poor" dynamic-shape support (verified). Realistic near-term: keep the fixed-window decode; treat a VARIABLE-tensor KV buffer as a research spike, not a quick win.

**3) ONNX Runtime VIPLite EP (#28244).** A **feature request only**, opened Apr 27, 2026 (by user evgen-pervenenko), now bot-marked **stale**, with no implementation (verified). No usable ORT VIPLite EP exists; it offers no alternative int8/LLM path today. (ORT now generally requires new EPs to be plugin EPs, raising the bar further.)

**4) etnaviv / Teflon.** Upstream in Mesa since 24.1; on the Vivante NPU path the maintainer states only **convolution, tensor add, and ReLU** (ReLU only fused into convolution) are implemented — *"Nothing else is implemented at the moment"* (verified). No attention/softmax/transformer ops; not viable for LLMs now. A community member explicitly flagged the A733/T527 Vivante NPU and the freed Allwinner docs, but Vizoso noted those docs only describe downstream driver usage and that Allwinner likely lacks the VeriSilicon hardware documentation needed for new ops.

**5) Exact VIP9000 configuration in the A733.** Verified: NPU is the Vivante VIP9000 (VIP V8 architecture), chip id `0x1000003B`, target `VIP9000NANODI_PLUS_PID0X1000003B`, ~3 TOPS INT8, on-die, exposed at `/dev/vipcore` via VIPLite 2.0. Radxa's docs reference an "NPU Version Comparison Table" and instruct **NPU_VERSION v3 for A733** (v2 for T527) (verified). The exact **MAC-array width / number of MACs for this NANODI_PLUS sub-variant is NOT public** — the A733 datasheet (V0.93) and Radxa docs do not disclose it (verified dead end on MAC count). The ~1.0 GHz clock is corroborated only by analogy to NXP's eIQ table listing the i.MX 8M Plus *"VIP9000 (1000 MHz)"* (verified as an analogy, not an A733 datasheet figure). For reference only, the **different** Amlogic A311D VIPNano is documented at *"5.0 TOPS INT8 inference up to 1536 MAC"* at 800 MHz (verified for A311D) — do not assume this for the A733.

**6) Other public LLM/VLM attempts on A733/A523/T527 or VIP9000-class NPUs.** Beyond the project's own SmolLM2-135M and the Frigate CNN work, **no other public LLM/VLM-on-NPU attempt** was found for Allwinner A733/A523/T527 (verified dead end). All neighboring LLM-on-NPU activity is on other silicon. This reinforces that the project is at the frontier for LLMs on this exact NPU.

## Recommendations (the int8 path, ranked by promise/effort)

**Stage 1 — Highest promise, lowest effort: lock in W8A16 with offline smoothing.**
- Quantize transformer linear/MLP **weights to per-channel int8 (pcq)** but force **all activations, attention, softmax, RMSNorm, RoPE, embedding and lm_head to int16** — not just the edges. The project already proved "edges-only int16" is insufficient and that full int16 is coherent; the hypothesis to test is that the **activations inside the transformer body** are what break int8. W8A16 is fully expressible in ACUITY today.
- Before ONNX export, apply **SmoothQuant-style activation smoothing in PyTorch** (migrate activation outliers into weights via a per-channel scale). This is the most portable trick from RKLLM/SmoothQuant and needs no toolchain feature.
- **Go/no-go gate:** compare ACUITY's **host** int8/int16 simulation cosine vs the FP reference, per layer. If host W8A16 cosine stays >0.99 through the stack, export to NBG; if a specific layer collapses, mark it int16 and re-run. This per-layer cosine is the decision metric. (The project already knows the host int8 output diverges, so use the host simulator as the fast iteration loop before ever touching the board.)

**Stage 2 — If W8A16 (int8 weights) still degrades: widen the most sensitive layers.**
- Use ACUITY's per-layer precision control to keep the most sensitive linears (typically the first and last transformer blocks, and any layer with host cosine <0.98) in int16, int8 elsewhere. This is the **expressible, static** analogue of outlier handling. Do **not** pursue LLM.int8()-style runtime decomposition (not expressible).

**Stage 3 — Parallel spike: the quantized-TFLite preset-scale route.**
- Because ONNX-QDQ import is dead, test the **quantized-TFLite import** path (acuitylite `TFLiteLoader`, "no need to quantize"). Export your SmoothQuant-smoothed, externally calibrated int8 model as a per-tensor-activation / per-channel-weight int8 TFLite, import it, export to NBG. This lets you control scales without trusting ACUITY's calibrator. Scope as a spike: int8-only (no int4), per-tensor activations, so it may not beat Stage 1.

**Stage 4 — Only if a newer toolchain is obtained: revisit hybrid and int4.**
- Pursue a newer ACUITY via NXP eIQ or Realtek Ameba AIoT, then re-test the hybrid path (the zero-byte bug may be fixed) and the `int4` qtype. Until then, treat int4/w4a16 on the A733 as unproven.

**Benchmarks/thresholds that would change the plan:**
- If Stage-1 host W8A16 per-layer cosine >0.99 end-to-end → ship to NBG; expected coherence comparable to the int16 baseline at lower memory.
- If a newer ACUITY completes the hybrid YAML dump on the full 49,152-vocab graph without hanging → re-open the w8a16 hybrid path as the preferred production flow.
- If a quantized-TFLite import round-trips with preserved scales and beats Stage-1 cosine → make TFLite the scale-control channel.

**Do NOT pursue (confirmed dead ends — don't re-try):**
- GPTQ/AWQ → **QDQ ONNX → ACUITY import** preserving scales: unsupported on the ONNX path.
- **Per-token/dynamic activation** quant or **LLM.int8() runtime outlier split**: not expressible in a static NBG.
- **etnaviv/Teflon** for any transformer op: only conv/add/ReLU exist.
- **ONNX Runtime VIPLite EP**: does not exist (stale feature request).
- Expecting the **ACUITY hybrid YAML dump** to "just finish" on the full large-vocab graph: it hangs — reduce `dtype_converter` count or reconstruct the YAML from the JSON instead.

## Caveats
- The hybrid zero-byte bug root cause (serializer choking on large converter count) is a reasoned **assumption**; no VeriSilicon source confirms it, because no public source discusses the bug at all. A confirmed dead end on documentation is itself a useful result: do not expect a vendor knowledge-base article to exist.
- The ~1.0 GHz clock is corroborated only by analogy to NXP's i.MX 8M Plus VIP9000; the A733-specific MAC count remains unpublished. The 1536-MAC figure belongs to the **different** Amlogic A311D and must not be assumed for the A733.
- **"W8A16 + offline smoothing will be coherent" is a hypothesis**, grounded in (a) the project's int16-coherent / int8-garbage result and (b) RKLLM's reliance on wide activations + grouping. It must be validated by the per-layer host-cosine gate before being trusted.
- ACUITY `int4`/`uint4` qtypes are present in the CLI but unproven on this exact runtime; do not plan around them.
- The B.3 verdict rests on the public pegasus help output, Radxa vendor docs, and acuitylite docs; the gated "Vivante Programming ACUITY Toolkit User Guide" PDF could in principle contain an undocumented QDQ-import capability, but all available primary sources point the other way.