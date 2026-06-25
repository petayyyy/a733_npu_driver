# Contributing

This repo is a research log and reproducible toolchain for running LLM/VLM
models on the Allwinner A733 Vivante VIP9000 NPU. Contributions that extend
the tooling, fix issues, or add new model ports are welcome.

## Report conventions

Reports live under `reports/`. Each report follows this structure:

- **Date**: when the work was done
- **Status**: `complete`, `blocked`, or `in-progress`
- **Verified / Assumption**: every technical claim is tagged
  - `[verified]` — measured on the board or confirmed from run logs
  - `[assumption]` — reasoned but not yet measured
- **Commands**: exact commands that were run, with their outputs
- **Logs**: paths to preserved board/host logs

### Blocker format

When ACUITY or VIPLite rejects an operation or a graph:

1. Save the full stdout and stderr under `logs/host/` or `logs/board/`
2. Write a report with:
   - Model name, config (window size, input/output shapes)
   - ACUITY version and target string
   - Exact error message and exit code
   - Node name if available (e.g., `SLICE`, `MATRIXMUL`, `RESHAPE2`)
   - Toolchain version: ACUITY `6.30.22`, VIPLite `2.0.3.2`
3. Reference the logs by path

### Adding new results

New benchmark results should:
- List the board, model, quantization, window size
- Report RSS, NBG size, tok/s, and whether text was coherent
- Include the exact prompt used
- Tag each number as `[verified]` or `[estimate]`

## Code conventions

- Shell scripts: POSIX (`set -eu`, no bashisms except when bash is needed)
- Python scripts: `from __future__ import annotations`, type hints encouraged
- C code: C11, no external dependencies beyond VIPLite SDK
- All scripts must pass `bash -n` / `python -m py_compile` before commit
- No hardcoded board IPs, usernames, or passwords (use environment variables)

## Repository structure

```
reports/            Research log (never delete entries, append only)
scripts/host/       x86 host tools
scripts/board/      Board-side scripts and runner source
docs/               User-facing guides and reference
logs/               Ignored; board and host logs
work/               Ignored; generated artifacts (ONNX, NBG, SDK builds)
```

## Adding a new board

If you have another A733-based board:

1. Confirm `cid=0x1000003b` from a `vpm_run` dump
2. Locate VIPLite 2.0.3.2 libraries for your OS image
3. Rebuild `npu_lm_runner` against your board's VIPLite headers/libs
4. Run a known-good NBG to verify binary compatibility
5. Add a report under `reports/` and a board reference in `docs/02-board-bringup.md`

NBG files are silicon-compatible across all A733 boards. Only the
runner and kernel module are board-specific.
