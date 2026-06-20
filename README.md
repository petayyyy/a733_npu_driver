# A733 NPU Driver / VIP9000 Bring-up Workspace

This repository tracks the practical work from `task.md`: bring up the
Allwinner A733 Vivante VIP9000 NPU, prove CNN inference first, then move toward
a hybrid VLM pipeline where the vision encoder runs on the NPU and the LLM
decoder runs on CPU.

## Current Status

- Phase 0 / G0: passed on Radxa Cubie A7Z over SSH.
- Phase 1 / G1: passed. `/dev/vipcore`, VIPLite 2.0.3.2, YOLOv8n, and SDK
  `vpm_run` all confirm the A733 VIP9000 path.
- Phase 2 / G2: passed for SDK LeNet and ONNX Inception v1 in both uint8 and
  int16 through ACUITY Docker `ubuntu-npu:v2.0.10.1`.
- Phase 3a: vision encoder NPU subgate passed. A tiny CLIP-like probe and a
  real MobileCLIP-S0 static vision encoder were exported to int16 NBG and
  validated on the A733 NPU.

The next milestone is pairing the MobileCLIP-S0 embedding output with a
CPU-side llama.cpp decoder for the hybrid VLM path.

## Repository Layout

```text
docs/
  hardware-access.md       SSH handoff checklist and board requirements
  roadmap.md               Gate-by-gate execution plan
scripts/
  board/
    a733-g0-g1-smoke.sh    Board diagnostics and NPU smoke-test collector
    build-ai-sdk.sh        Build helper for an already cloned ai-sdk tree
    run-vpm.sh             Logged wrapper around vpm_run
  host/
    prepare-workspace.ps1  Create local host workspace and check Docker image
    run-board-smoke.ps1    Copy board scripts over SSH and run G0/G1 smoke test
    ssh_exec.py            Password-based SSH/SFTP helper for automation
reports/
  status.md                Living status log
  g1-radxa-cubie-a7z.md    Hardware bring-up report
  g2-acuity-lenet.md       ACUITY LeNet validation report
  g2-acuity-inception-v1.md ACUITY ONNX Inception validation report
  g3a-clip-tiny-vision.md  Tiny CLIP vision-encoder NPU probe
  g3a-mobileclip-s0-vision.md MobileCLIP-S0 vision-encoder NPU validation
```

Generated board logs, host logs, models, and temporary workspaces are ignored by
git.

## Host Preparation

From PowerShell on the x86 host:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1
```

This creates the local working directories and checks whether Docker and the
expected ACUITY image (`ubuntu-npu:v2.0.10.1`) are available. It does not
download anything by default.

When SSH access is available, the host can launch the board smoke test with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\run-board-smoke.ps1 `
  -Host <board-host-or-ip> -User <ssh-user>
```

## Board Bring-up

Once SSH access is available, copy the board scripts to the board and run:

```bash
chmod +x scripts/board/*.sh
scripts/board/a733-g0-g1-smoke.sh
```

If a known `vpm_run` model command is available, pass it through the environment:

```bash
A733_VPM_RUN_ARGS="--help" scripts/board/a733-g0-g1-smoke.sh
```

For a real model run, use the exact arguments required by the board's `vpm_run`
package:

```bash
A733_VPM_RUN_ARGS="<vendor vpm_run arguments>" scripts/board/a733-g0-g1-smoke.sh
```

For boards that have a custom sample instead of `vpm_run`, use
`A733_NPU_RUN_CMD`:

```bash
A733_NPU_RUN_CMD='cd /home/radxa/yolo_shm && export LD_LIBRARY_PATH=/home/radxa/lib:$LD_LIBRARY_PATH && ./build/test_npu_a733 -nb yolov8n_6_uint8_a733.nb -i dog.jpg' \
  scripts/board/a733-g0-g1-smoke.sh
```

The script writes logs under `logs/board/<host>-<timestamp>/` and summarizes
whether G0/G1 checks are satisfied.

## SSH Stage Inputs

For a new board or image, send:

- SSH endpoint and username.
- Board type: Radxa Cubie A7Z or Orange Pi Zero 3W.
- OS image name/version and whether sudo is available.
- Any existing NPU demo/model directory on the board.
- Whether network access from the board is allowed for cloning/building
  `ai-sdk`.

With that, the first hardware pass is:

1. Confirm kernel, CPU cores, thermals, and image details.
2. Confirm `/dev/vipcore` and VIPLite userspace libraries.
3. Locate or build `vpm_run`.
4. Run the first CNN/NBG inference and capture the VIPLite banner containing
   `cid=0x1000003b`.
