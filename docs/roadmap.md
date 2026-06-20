# A733 VIP9000 Execution Roadmap

This is the execution version of the roadmap in `task.md`. Gates are written so
they can be checked from logs.

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

The next G2-adjacent model is a vision encoder candidate for Phase 3a.

Primary script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1 `
  -AcuityImage ubuntu-npu:v2.0.10.1
```

## Phase 3a - Hybrid VLM Path

Goal: put the static vision encoder on NPU and keep autoregressive decode on
CPU.

Gate G3a:

- Small ViT/SigLIP-like encoder exported to NBG.
- Encoder inference runs on NPU.
- llama.cpp CPU decoder runs on A76 cores.
- End-to-end image-to-text response works.
- Per-stage timing is captured.

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
