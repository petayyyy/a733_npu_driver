# Hardware Access Handoff

When the board is ready, send the SSH endpoint and I will run the bring-up
sequence directly.

## Minimum Access

- SSH user and host.
- Sudo access if `/dev/vipcore`, package logs, or kernel logs require it.
- Board type and image name/version.
- Network policy: whether the board may clone from GitHub or install packages.

## Useful Board Paths

If known, include paths for:

- `ai-sdk` checkout.
- VIPLite runtime libraries.
- `vpm_run`.
- Existing NBG demo models.
- Calibration or test images.

## First Commands I Will Run

```bash
uname -a
cat /etc/os-release
nproc
ls -l /dev/vipcore
find /usr /opt -name 'vpm_run' -o -name 'libVIPhal.so' -o -name 'libNBGlinker.so'
```

Then I will run:

```bash
scripts/board/a733-g0-g1-smoke.sh
```

From the host side, this can also be launched with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\run-board-smoke.ps1 `
  -Host <board-host-or-ip> -User <ssh-user>
```

If a valid demo command is known:

```bash
A733_VPM_RUN_ARGS="<vendor vpm_run arguments>" scripts/board/a733-g0-g1-smoke.sh
```

## Success Evidence For G1

The first successful hardware milestone needs:

- A complete smoke-test log directory.
- `/dev/vipcore` present.
- VIPLite runtime libraries found.
- `vpm_run` or equivalent sample executed successfully.
- Log text showing `cid=0x1000003b` or another exact A733 VIP9000 identifier.
