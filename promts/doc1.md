TASK DOC1-repo-cleanup: Turn this repo into a clean, well-documented open-source resource so
someone reproducing LLM/VLM-on-NPU work for the Allwinner A733 / Vivante VIP9000 can follow the
path. Do NOT change any working code behavior; this is documentation + structure only. Keep all
existing reports/ (they are the research log) but make them navigable.

DO:
1. Rewrite README.md as the project front door, with these sections:
   - One-paragraph "What this is": running real LLMs and a VLM vision pipeline NPU-only on the
     Allwinner A733 (Vivante VIP9000, ~3 TOPS), validated on Radxa Cubie A7Z (dev) and Orange Pi
     Zero 3W (final). Honest framing: this is a working proof-of-concept + a full reproducible
     toolchain, not a polished product.
   - "What works / what doesn't" summary box (link to the results doc from DOC2).
   - Hardware/OS table (both boards: SoC, NPU cid 0x1000003b, kernels, RAM, image versions).
   - A clear "Start here" path and a link to docs/ guides below.
2. Create a docs/ folder with task-oriented run guides (each a standalone how-to with exact
   commands, prerequisites, and expected output):
   - docs/01-setup-host.md: x86 host + ACUITY Docker (ubuntu-npu:v2.0.10.1), the conversion flow
     (convert_onnx_to_nbg.sh), the host-oracle gate tooling.
   - docs/02-board-bringup.md: bringing up the NPU on each board — /dev/vipcore, VIPLite
     2.0.3.2, building/locating vpm_run and the persistent runner, the glibc-matched VIPLite .so
     difference between Radxa and Orange Pi (the /home/orangepi/lib vs ai-sdk path issue).
   - docs/03-run-llm-npu.md: how to convert + run a SmolLM2 LLM NPU-only, with the fixed-window
     concept explained (W, no KV-cache), and which (model, W) configs are coherent.
   - docs/04-chat-shell.md: how to use the interactive chat shell (B2) on the board.
   - docs/05-run-vlm-npu.md: the VLM vision-encoder + bridge pipeline (B3): inputs/outputs, how
     to run it.
   - docs/06-cpu-baseline.md: running Qwen2.5-0.5B on CPU via llama.cpp (B4) for the "ROS2 paused"
     / long-context mode, with the recommended quant/context.
   - docs/07-porting-radxa-to-orangepi.md: what transfers as-is (NBG files, same silicon) vs what
     must be rebuilt (kernel NPU module from the OPi BSP, runner recompiled on bookworm,
     glibc-matched .so).
   - docs/08-known-limits-and-blockers.md: the honest limits — fixed-window (no KV-cache),
     3 TOPS memory-bound decode, int16-dynamic-fixed-point breaks on outlier models, Qwen2.5-0.5B
     and SmolLM2-1.7B do NOT export to NBG (link the vendor blocker reports t6/t9/t10/t11).
3. Add a "Configurations" reference: docs/configurations.md — a decision guide. For each goal
   (fast tiny chat / smarter chat / VLM vision / long-context CPU chat / hybrid), state the exact
   model, backend (NPU vs CPU), window or context, expected tok/s, and which guide to follow.
4. Repo hygiene: a clear directory map in README (scripts/host, scripts/board, reports/, docs/,
   work/ is gitignored), a CONTRIBUTING note (how reports are structured, verified/assumption
   convention), a LICENSE if missing (suggest Apache-2.0 or MIT, ask the user which), and a
   .gitignore review so large artifacts stay out.
5. Add an "Acknowledgements / references" section linking the external resources that helped:
   Radxa A733 NPU docs, VeriSilicon TIM-VX/ACUITY, ZIFENG278/ai-sdk, the Rabs9 A733 kernel repo,
   RKLLM as the reference architecture, and the research findings (the 65536 dim limit source).

DELIVERABLE: a rewritten README.md, a docs/ folder with the 8 guides + configurations.md,
CONTRIBUTING.md, LICENSE (pending user choice), updated .gitignore. Every command must be
copy-pasteable and match what the reports actually did. Mark anything not yet board-verified as
such.

SUCCESS GATE: a newcomer can read README -> pick a configuration -> follow one docs/ guide ->
reproduce a working run. No code behavior changed. Committed.

START FROM: the existing reports/ (t0-t11, b1b, b2, b3, b4, g-series) as the source of truth for
every command and result; the scripts/host and scripts/board directories.