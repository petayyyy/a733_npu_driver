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

## Next Gate

Phase 2 extension / custom model:

1. Convert a custom ONNX CNN to NBG with int16 quantization.
2. Validate converted NBG on the Radxa board.
3. Start Phase 3a vision-encoder candidate selection.
