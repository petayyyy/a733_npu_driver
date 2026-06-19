# Status

## 2026-06-20

- Read `task.md` and selected the practical first target: G0/G1 hardware
  bring-up and CNN proof of principle.
- Added project structure for docs, board scripts, host scripts, and reports.
- Added SSH launch helper for copying board scripts and starting G0/G1 remotely.
- Verified host preparation script locally. Docker is installed, but the daemon
  is not running and `ubuntu-npu:v2.0.10` is not present locally.
- Verified bash syntax for board scripts with Git Bash.
- Hardware execution is pending SSH access to the board.

## Next Gate

G0/G1 on real hardware:

1. Run `scripts/board/a733-g0-g1-smoke.sh`.
2. Locate or build `vpm_run`.
3. Run a vendor CNN/NBG demo.
4. Capture VIPLite banner and result logs.
