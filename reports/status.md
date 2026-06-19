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

## Next Gate

Phase 2 / ACUITY toolchain:

1. Start Docker daemon on host or provide ACUITY image `ubuntu-npu:v2.0.10`.
2. Convert a known ONNX CNN to NBG with int16 quantization.
3. Validate converted NBG on the Radxa board.
