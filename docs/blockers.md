# A733 / Vivante VIP9000 NPU — LLM/VLM blockers (asking for ideas)

## Setup
- SoC: Allwinner A733. NPU: VeriSilicon Vivante VIP9000, cid=0x1000003b, single core, ~1.0 GHz,
  ~3 TOPS INT8, native INT8/INT16/FP16/BF16, 32-bit LPDDR5.
- Board: Orange Pi Zero 3W (A733), 6 GB, kernel 6.6.98, /dev/vipcore, VIPLite 2.0.3.2-AW-2024-08-30.
- Toolchain (public, no NDA): VeriSilicon ACUITY 6.30.22 (Docker ubuntu-npu:v2.0.10.1),
  Vivante IDE 5.11.0. Flow: ONNX -> pegasus import/quantize/inference/export -> NBG -> VIPLite.
  NOTE: ACUITY "int16" = dynamic fixed point (one power-of-2 scale per tensor), not IEEE fp16.
- What works: SmolLM2-135M/360M run NPU-only, coherent, fixed-window (W=32/64), int16, on the
  board (135M/W32 = 20.7 tok/s). Builder verified correct (FP32 ONNX vs HF oracle cosine ~1.0).

## Problem 1 — Qwen2.5-0.5B will not produce a coherent, exportable NBG
Qwen2.5-0.5B (24 layers, hidden 896, 14 q / 2 kv heads, vocab 151936, rope_theta 1e6, QKV bias)
has large activation outliers (act_absmax ~1790; some RMS-squared tensors ~2.65e6).
All numbers = ACUITY host sim vs FP oracle, before board:
- INT16 (dynamic fixed point): exports, INCOHERENT, logits cosine 0.236.
- Full FP16: exports, INCOHERENT, cosine 0.541 (5-bit exponent can't cover the outliers).
- Full BF16: host quality PASSES (cosine 0.991, top-1 match) but NBG export FAILS:
  `E [main.c:vnn_VerifyGraph:93] CHECK STATUS(-3 ...) ; Fatal model generation error: 64768`.
- Chunked lm_head (3 chunks <=50646, host cosine 0.99999), with or without final Concat:
  still `vnn_VerifyGraph -3 / 64768`.
- Chunk lm_head AND token embedding: failure MOVES to a BF16 `DATACONVERT` setup failure
  (`Fatal model generation error: 65280`).  => the 151936 logits tensor is not the only blocker;
  the full BF16 body graph is rejected even with all dims < 65536.
- ACUITY hybrid (BF16 on top outlier layers + lm_head, INT16 elsewhere): host quality FAILS
  (cosine 0.254) AND export fails at an illegal BF16->DFP-INT16 boundary at a `PERMUTE` node (65280).

Verified context:
- The `vnn_VerifyGraph -3 / error 64768` signature on the same Vivante NBG compiler is documented
  as a per-tensor DIMENSION LIMIT 65536 ("dimension must be in range [0, 65536[") — ST Edge AI
  Cloud forum. Qwen vocab 151936 > 65536. Corroborated by vLLM-Ascend ("chunked matmul to work
  around the NPU 65536 dimension limit").
- Legal dtype boundaries (verified in TIM-VX source, github.com/VeriSilicon/TIM-VX,
  src/tim/vx/internal/src/ops/): MATMUL allows BF16 only as BF16,BF16->BF16 (no mixed);
  SLICE allows BF16->BF16 and INT16->INT16 but NOT BF16->INT16; DATACONVERT supports INT16->BF16,
  BF16->F16/F32, F16->INT16 but NOT direct BF16->INT16. So a BF16<->INT16 boundary must be a
  standalone DataConvert (legal bridge BF16->F16->INT16) and must never sit inside SLICE/MATMUL/PERMUTE.
- VeriSilicon's own public Qwen2.5 example (github.com/VeriSilicon/acuity-models,
  models/qwen2.5_7b_decode) is a SINGLE transformer block, not a monolithic model with a vocab head.

Core question: how to get a coherent + exportable Qwen2.5-0.5B on this NPU — i.e. either
(a) hold the outliers without BF16 (per-channel/per-axis INT16? clean W8A16 with int8 weights +
int16 activations?), (b) force ACUITY to place the legal BF16->F16->INT16 DataConvert nodes so no
PERMUTE/SLICE/MATMUL straddles a mixed boundary, (c) block-partitioned per-decoder-block NBGs, or
(d) is this genuinely a toolchain limit needing a newer ACUITY / vendor support?

Tried and ruled out: QDQ-ONNX import (ACUITY ignores QDQ scales); TVM vsi_npu and hand-built
TIM-VX (same verifier, no RMSNorm/gather/slice/SwiGLU coverage); W8A16 + SmoothQuant alpha=0.5
(over-smoothing corrupted even the int16 control).

## Problem 2 — SmolLM2-1.7B will not export
All windows: NBG export FAILS (`gen_nbg` segfault / 0-byte NBG) from the 6.85 GB external-data
ONNX graph. Builder passes FP gate. Question: graph-size / external-data limit, or a known fix?

## Problem 3 — No KV-cache -> short usable context
NBG graphs are static-shape, so autoregressive decode uses a fixed window and recomputes the whole
window every token. On-board coherence holds at W=32/64 and breaks at W>=128 (usable context =
prompt+response ~25-50 words). TIM-VX has a VARIABLE tensor for recurrent state — but we found no
working KV-cache / dynamic-length transformer precedent on VIP9000. Question: any stateful/
variable-length path in VIPLite/TIM-VX to implement a real KV-cache on this NPU?

## Problem 4 (minor) — int16 host sim is not a coherence predictor
ACUITY `pegasus inference` int16 host cosine gave false negatives (failed the known-good
SmolLM2-135M/W32). We now gate coherence on actual on-board generation + an FP-oracle check.
Flagging in case it's a known ACUITY quirk.

## Reference (for comparison)
Rockchip RK3588 (6 TOPS) runs full small LLMs/VLMs (incl. Qwen2.5, InternVL3.5-1B) via RKLLM with
a real KV-cache + int4/int8. We're trying to approach that on VIP9000/ACUITY without that runtime.