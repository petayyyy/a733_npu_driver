# A733 NPU Driver / VIP9000 Bring-up Workspace

This repository tracks the practical work from `task.md`: bring up the
Allwinner A733 Vivante VIP9000 NPU, prove CNN inference first, then move toward
a hybrid VLM pipeline where the vision encoder runs on the NPU and the LLM
decoder runs on CPU.

## Current Status

- Phase 0 / G0: waiting for board access.
- Phase 1 / G1: scripts prepared for `/dev/vipcore`, VIPLite, and `vpm_run`
  smoke testing.
- Phase 2+: host-side workspace scaffolding prepared; ACUITY conversion still
  depends on the vendor Docker/toolkit being available locally.

The next real milestone is an SSH session to the Radxa Cubie A7Z or target
Orange Pi board.

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
```

Generated board logs, host logs, models, and temporary workspaces are ignored by
git.

## Host Preparation

From PowerShell on the x86 host:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1
```

This creates the local working directories and checks whether Docker and the
expected ACUITY image (`ubuntu-npu:v2.0.10`) are available. It does not download
anything by default.

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

## What I Need For The SSH Stage

Send:

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
