# A733 VIP9000 Execution Roadmap

This is the execution version of the roadmap in `task.md`. Gates are written so
they can be checked from logs.

## Active Constraint - NPU-Only LLM/VLM

The active user requirement is that LLM/VLM model-layer compute must run on the
A733 NPU. CPU decoder paths are diagnostic baselines only and do not satisfy any
LLM/VLM gate. See `docs/npu-only-requirement.md`.

## Phase 0 - Environment

Goal: prove the board is usable before touching the NPU stack.

Gate G0:

- Board boots and is reachable over SSH.
- `/proc/cpuinfo` or `nproc` reports 8 cores.
- Kernel and OS image are recorded.
- Thermal zones are readable and stable at idle.
- CPU governor/frequency information is recorded when available.

Primary script:

```bash
scripts/board/a733-g0-g1-smoke.sh
```

## Phase 1 - NPU Bring-up / CNN Proof Of Principle

Goal: prove the Vivante VIP9000 path through VIPLite and `/dev/vipcore`.

Gate G1:

- `/dev/vipcore` exists.
- VIPLite/runtime libraries are present.
- `vpm_run` or an equivalent awnn/VIPLite sample is present or built.
- A CNN/NBG inference runs successfully.
- Logs contain the expected A733 VIP9000 identity, especially
  `cid=0x1000003b`.

Primary scripts:

```bash
scripts/board/a733-g0-g1-smoke.sh
scripts/board/build-ai-sdk.sh --sdk-dir <ai-sdk>
scripts/board/run-vpm.sh <vpm_run args>
```

## Phase 2 - ACUITY Toolchain

Goal: produce a custom NBG from a known ONNX model.

Gate G2:

- Host has access to ACUITY Docker image `ubuntu-npu:v2.0.10`.
- ONNX model converts to NBG with int16 quantization.
- Converted NBG runs on board.
- Accuracy is compared against ONNX baseline and recorded.

Current status: passed for SDK LeNet and ONNX Inception v1 in both uint8 and
int16 using `ubuntu-npu:v2.0.10.1`. Inception v1 validates the non-toy ONNX CNN
path on the Radxa board:

- uint8: `profile inference time` about `14.36ms`, `vpm run ret=0`.
- int16: `profile inference time` about `20.85ms`, ONNX/non-quantized top-5
  preserved, `vpm run ret=0`.

The G2-adjacent static vision encoder path is also proven through the
MobileCLIP-S0 Phase 3a report.

Primary script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1 `
  -AcuityImage ubuntu-npu:v2.0.10.1
```

## Phase 3a - NPU-Only Transformer Path

Goal: prove that transformer decoder compute can run on the A733 VIP9000 NPU,
then extend the result into a tiny NPU language model and NPU VLM path.

Gate G3a:

- Fixed-shape transformer decoder block exports to NBG.
- Decoder block runs on NPU and produces expected output/logits.
- Tiny fixed-shape language model runs on NPU.
- VLM path runs model-layer compute on NPU: vision encoder, projector/adapter,
  and language decoder graph.
- Per-stage timing is captured.

Current status: vision encoder NPU subgate passed; CPU decoder result is
historical baseline only and is no longer a target gate.

The first compatibility probe passed. The tiny random CLIP vision ONNX from
`hf-internal-testing/tiny-random-CLIPModel` was fixed to `1x3x30x30`, quantized
to int16, exported to NBG, and run on the A733:

- NBG size: `720,824` bytes.
- Output: `1x64` int16 embedding tensor.
- Runtime: `profile inference time` about `2.17ms`, `vpm run ret=0`.
- Covered transformer-style ops include MatMul, Softmax, LayerNorm pattern,
  Gather, Conv patch embedding, and MLP blocks.

The real encoder pass also succeeded. `Xenova/mobileclip_s0`
`onnx/vision_model.onnx` was fixed to `1x3x256x256`, quantized to int16,
exported to NBG, and run on the A733:

- NBG size: `19,376,840` bytes.
- Output: `1x512` int16 image embedding tensor.
- Runtime: `profile inference time` about `22.6ms`, `vpm run ret=0`.
- ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
  diff `0.002471924`, mean abs diff `0.000398278`, cosine `0.999884700`.

This completed the static vision-encoder NPU proof and established the encoder
side of the NPU-only VLM path.

The first decoder-block NPU subgate also succeeded. A deterministic tiny
fixed-shape transformer decoder block was generated as ONNX, quantized/exported
through ACUITY to int16 NBG, and run on the A733:

- NBG size: `85,144` bytes.
- Input: `1x4x8` float16 embedding tensor.
- Output: `1x4x16` logits tensor, int16 dynamic fixed point `dfp=14`.
- Runtime: `profile inference time` between `59us` and `68us`,
  `vpm run ret=0`.
- ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
  diff `0.000549316`, mean abs diff `0.000133514`, cosine `0.999999919`.
- Covered decoder ops include MatMul, Softmax, GELU, LayerNorm-style
  reductions, causal attention, residuals, and logits projection.

This proves a static transformer decoder block can execute on the VIP9000 NPU.

The tiny language-model NPU subgate also succeeded. A deterministic fixed-shape
LM graph was generated with int32 token IDs, ONNX `Gather` token embeddings,
position embeddings, decoder compute, and logits in one NBG:

- NBG size: `87,016` bytes.
- Input: `1x4` int32 token IDs (`1 5 9 2`).
- Output: `1x4x16` logits tensor, int16 dynamic fixed point `dfp=14`.
- Runtime: `profile inference time` between `62us` and `71us`,
  `vpm run ret=0`.
- ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
  diff `0.000610352`, mean abs diff `0.000153542`, cosine `0.999999929`.
- Covered language-model path includes token embedding `Gather`, position
  embedding add, causal attention, MLP, LayerNorm-style reductions, and logits
  projection.

This proves that the public ACUITY/VIPLite path can run a complete tiny
fixed-shape language-model graph on the A733 NPU. The next G3a gate is NPU VLM
integration: MobileCLIP-S0 encoder output, projector/adapter, and NPU language
decoder graph.

Historical CPU baseline: llama.cpp built on the Radxa board at
commit `f449e0553708b895adbd94a301431cef691f632d`; the separate
`llama-simple`, `llama-simple-chat`, and `llama-bench` targets were used because
the current upstream unified `llama-app` target did not link in this
configuration. `SmolLM2-135M-Instruct-Q4_K_M.gguf` ran CPU-only:

- Model: `134.52M` params, `98.87 MiB` in llama-bench.
- Best llama-bench decode for this tiny model: `56.74 tok/s` at 2 threads.
- Best llama-bench prompt throughput: `122.57 tok/s` at 8 threads.
- Generation smoke via `llama-simple`: prompt eval `46.93 tok/s`, decode eval
  `29.92 tok/s`, total `2515.07 ms / 64 tokens`.

This CPU result does not complete G3a and is not a deliverable.

## Phase 3b - LLM-on-NPU R&D

Goal: time-boxed investigation only.

Gate G3b:

- Either one transformer-relevant op/subgraph is proven on NPU through
  TVM-BYOC/TIM-VX/VIPLite, or a concrete blocker is documented for vendor
  escalation.

## Phase 4 - Optimization

Gate G4:

- At least one measured 20 percent improvement versus the naive baseline, or a
  documented reason why the current bottleneck cannot be improved with available
  controls.

## Phase 5 - Orange Pi Zero 3W Port

Gate G5:

- Orange Pi image provides a compatible `/dev/vipcore`.
- NBG files from Radxa run unchanged.
- Userspace apps are rebuilt against the target image.
- Outputs match Radxa within tolerance.

## Phase 6 - Benchmark And Documentation

Gate G6:

- Benchmark matrix includes latency, throughput, RSS, power if available, and
  accuracy.
- Logs and methodology are sufficient to reproduce the numbers.
