# Status

## 2026-06-20

- Read `task.md` and selected the practical first target: G0/G1 hardware
  bring-up and CNN proof of principle.
- Added project structure for docs, board scripts, host scripts, and reports.
- Added SSH launch helper for copying board scripts and starting G0/G1 remotely.
- Verified host preparation script locally. Docker is installed, but the daemon
  is not running and `ubuntu-npu:v2.0.10` is not present locally.
- Verified bash syntax for board scripts with Git Bash.
- Connected to Radxa Cubie A7Z at `192.168.31.76`.
- Gate G0 passed: Debian 11, kernel `5.15.147-21-a733`, 8 cores, thermals
  readable.
- Gate G1 passed: `/dev/vipcore` present; VIPLite 2.0.3.2 loaded
  `yolov8n_6_uint8_a733.nb`; single-image YOLO inference on `dog.jpg` produced
  bicycle/dog/car detections.
- Built standard SDK `examples/vpm_run` on the board from `ZIFENG278/ai-sdk`;
  `operator/v3/network_binary.nb` runs with `cid=0x1000003b` and
  `profile inference time=2807us`.
- Gate G2 passed for SDK LeNet: ACUITY Docker `ubuntu-npu:v2.0.10.1` generated
  uint8 and int16 NBG files, both validated on the A733 through `vpm_run`.
- Gate G2 extension passed for ONNX Inception v1: ACUITY generated uint8 and
  int16 NBG files, both validated on the A733 through `vpm_run`.
  - uint8: `1x3x224x224`, `profile inference time` about `14.36ms`, top-1
    class index `885`, `vpm run ret=0`.
  - int16: `1x3x224x224`, `profile inference time` about `20.85ms`, top-1
    class index `885`, ONNX/non-quantized top-5 preserved, `vpm run ret=0`.
- Phase 3a probe started: `hf-internal-testing/tiny-random-CLIPModel`
  `onnx/vision_model.onnx` was fixed to `1x3x30x30`, converted to int16 NBG,
  and validated on the A733 through `vpm_run`.
  - NBG size: `720,824` bytes.
  - Operators covered include MatMul, Softmax, LayerNorm pattern, Gather, Conv,
    and MLP blocks.
  - Runtime: `profile inference time` about `2.17ms`, output shape `1x64`,
    `vpm run ret=0`.
- Phase 3a real encoder subgate passed: `Xenova/mobileclip_s0`
  `onnx/vision_model.onnx` was fixed to `1x3x256x256`, converted to int16 NBG,
  and validated on the A733 through `vpm_run`.
  - NBG size: `19,376,840` bytes.
  - Output: `1x512` int16 image embedding.
  - Runtime: `profile inference time` about `22.6ms`, `vpm run ret=0`.
  - ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
    diff `0.002471924`, mean abs diff `0.000398278`, cosine `0.999884700`.
- Phase 3a CPU decoder subgate passed: llama.cpp built on the Radxa board at
  commit `f449e0553708b895adbd94a301431cef691f632d`, and
  `SmolLM2-135M-Instruct-Q4_K_M.gguf` ran through CPU-only GGUF inference.
  - Model: `134.52M` params, `98.87 MiB` in llama-bench, Q4_K_M.
  - llama-bench, CPU-only: best decode for this model was `56.74 tok/s` at
    2 threads; best prompt throughput was `122.57 tok/s` at 8 threads.
  - llama-simple chat prompt smoke: prompt eval `46.93 tok/s`, decode eval
    `29.92 tok/s`, total `2515.07 ms / 64 tokens`.

## Next Gate

Phase 3a / hybrid VLM path:

1. Select the first image-to-text target pairing for the MobileCLIP-S0
   embedding path.
2. Wire encoder output transfer and decoder input plumbing.
3. Add or source the projector/adapter needed between `1x512` image embeddings
   and the selected decoder.
4. Capture end-to-end timing for image preprocess, NPU encoder, projector, and
   CPU decode.
