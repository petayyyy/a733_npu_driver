# A733 NPU Driver / VIP9000 Bring-up Workspace

This repository tracks the practical work from `task.md`: bring up the
Allwinner A733 Vivante VIP9000 NPU, prove CNN inference first, then move toward
LLM/VLM execution where model-layer compute runs on the NPU. The active project
constraint is NPU-only LLM/VLM inference; CPU execution is allowed only for
orchestration, validation, and other non-inference support work.

## Current Status

- Phase 0 / G0: passed on Radxa Cubie A7Z over SSH.
- Phase 1 / G1: passed. `/dev/vipcore`, VIPLite 2.0.3.2, YOLOv8n, and SDK
  `vpm_run` all confirm the A733 VIP9000 path.
- Phase 2 / G2: passed for SDK LeNet and ONNX Inception v1 in both uint8 and
  int16 through ACUITY Docker `ubuntu-npu:v2.0.10.1`.
- Phase 3a: NPU-only path is active. A tiny CLIP-like probe and a real
  MobileCLIP-S0 static vision encoder were exported to int16 NBG and validated
  on the A733 NPU. A tiny fixed-shape transformer decoder block was also
  exported to int16 NBG and validated on the A733 NPU, including attention,
  Softmax, GELU, LayerNorm-style reductions, residuals, and logits output. The
  follow-up tiny language-model probe with int32 token IDs, ONNX `Gather` token
  embeddings, decoder compute, and logits also runs as an A733 NBG on the NPU.
  A tiny VLM bridge now also runs on the A733 NPU: MobileCLIP-S0-style
  `1x512` image embedding input, NPU projector/adapter, token embedding
  `Gather`, image/text concat, decoder compute, and `1x5x16` logits in one NBG.
  A fixed-window tiny LM decode loop has also been validated: CPU updates the
  `1x4` token window and postprocesses logits, while every LM forward pass runs
  as an NBG on the NPU. The per-token `vpm_run` relaunch path has now been
  replaced for the tiny LM by a persistent VIPLite C runner that loads the NBG
  once and reproduces the same generated token sequence with lower stable
  per-token wall time. On the Orange Pi Zero 3W, the same persistent-runner
  pattern now drives `SmolLM2-135M-Instruct` W=32 int16 through an interactive
  chat shell that streams tokens at about 21 tok/s while keeping model-layer
  compute on the NPU.
  The previous CPU llama.cpp decoder result is retained only as a diagnostic
  baseline and is not a project deliverable.

The next milestone is extending the persistent fixed-window loop pattern to the
VLM bridge path.

## Results

All measured numbers, comparison tables, coherence analysis, and use-case
recommendations are consolidated in **[docs/RESULTS.md](docs/RESULTS.md)**.

## Repository Layout

```text
docs/
  RESULTS.md               Consolidated results, tables, and recommendations
  hardware-access.md       SSH handoff checklist and board requirements
  npu-only-requirement.md  Active hard requirement for NPU-only LLM/VLM work
  roadmap.md               Gate-by-gate execution plan
scripts/
  board/
    a733-g0-g1-smoke.sh    Board diagnostics and NPU smoke-test collector
    build-ai-sdk.sh        Build helper for an already cloned ai-sdk tree
    build-npu-lm-runner.sh Build the persistent tiny LM VIPLite runner
    chat_shell.py          Orange Pi interactive SmolLM2 NPU chat shell
    npu_lm_runner.c        Persistent tiny LM VIPLite runner source
    build-llama-cpp.sh     Historical CPU baseline helper, not a deliverable
    run-llama-decode.sh    Historical CPU baseline helper, not a deliverable
    run-npu-lm-runner.sh   Logged persistent tiny LM runner wrapper
    run-tiny-lm-decode-loop.sh NPU-only tiny LM fixed-window decode loop
    run-vpm.sh             Logged wrapper around vpm_run
  host/
    make_tiny_decoder_block_onnx.py Generate fixed-shape decoder-block ONNX probe
    make_tiny_lm_onnx.py Generate fixed-shape tiny LM ONNX probes
    make_tiny_vlm_bridge_onnx.py Generate fixed-shape VLM bridge ONNX probe
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
  g3a-tiny-decoder-block-npu.md Tiny transformer decoder block NPU validation
  g3a-tiny-lm-decode-loop-npu.md Tiny LM NPU fixed-window decode loop
  g3a-tiny-lm-gather-npu.md Tiny token-id LM NPU validation
  g3a-tiny-vlm-bridge-npu.md Tiny VLM bridge NPU validation
  t1-persistent-runner.md  Persistent tiny LM VIPLite runner validation
  g3a-llama-cpp-decoder.md Historical CPU baseline, not a deliverable
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

## Orange Pi NPU Chat Shell

The current human-facing SmolLM2 chat entry point on the Orange Pi Zero 3W is:

```bash
cd /home/orangepi/a733_npu_driver

python3 scripts/board/chat_shell.py \
  --model /home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb \
  --tokenizer /home/orangepi/a733_npu_driver/work/models/smollm2-135m-instruct \
  --runner /home/orangepi/a733_npu_driver/build/npu_lm_runner \
  --vip-lib /home/orangepi/lib \
  --window 32 \
  --max-new-tokens 32 \
  --greedy
```

Build or rebuild the runner on that image with:

```bash
cd /home/orangepi/a733_npu_driver

bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
```

The shell prints the NBG size, `/dev/vipcore` status, `cid=0x1000003b`,
`nbg_loaded_once=1`, the live fixed-window counter, and tok/s after each reply.
Use `/reset` to clear the conversation window and `/exit` to quit. The fixed
window is honest: once the rendered chat exceeds `--window`, only the last
`W` tokens are submitted to the NPU graph.

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
