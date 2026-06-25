# 02 — Board Bring-Up

Bringing up the NPU on Radxa Cubie A7Z and Orange Pi Zero 3W.

## Prerequisites

- Physical or SSH access to the board
- Board boots Debian and is reachable
- For Orange Pi: the kernel must include the NPU module (`sunxi_npu`)

## Quick diagnostic

```bash
uname -a
cat /etc/os-release
nproc
free -h
ls -l /dev/vipcore
```

Expected output for a working NPU:
- `/dev/vipcore` present (character device, major:minor 199,0)
- Kernel: `5.15.147-21-a733` (Radxa) or `6.6.98-sun60iw2` (Orange Pi)
- 8 cores (Radxa) or 8 cores (Orange Pi: 2×A76 + 6×A55)

## Automated smoke test

```bash
cd ~/a733_npu_driver
chmod +x scripts/board/*.sh
bash scripts/board/a733-g0-g1-smoke.sh
```

This collects:
- Kernel and CPU info
- Thermal zones
- `/dev/vipcore` presence
- VIPLite runtime library locations
- Available `vpm_run` or equivalent binaries

## Gate G0: board is usable

- Board boots and is reachable over SSH
- `/proc/cpuinfo` or `nproc` reports 8 cores
- Thermal zones are readable and stable at idle
- Free RAM sufficient for target workload (>1 GiB for LLM NBG)

## Gate G1: NPU is accessible

### Locate VIPLite libraries

```bash
find /usr /opt /home -name 'libVIPhal.so' -o -name 'libNBGlinker.so' 2>/dev/null
```

Common locations:
- **Radxa**: `/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/`
- **Orange Pi**: `/home/orangepi/lib/`

### Locate or build vpm_run

```bash
find /usr /opt /home -name 'vpm_run' 2>/dev/null
```

Common locations:
- **Radxa**: `/home/radxa/ai-sdk/examples/vpm_run/vpm_run`
- **Orange Pi**: `/opt/vpm_run/vpm_run`

If `vpm_run` doesn't exist, build the SDK:

```bash
# Clone the A733-compatible SDK
git clone https://github.com/ZIFENG278/ai-sdk ~/ai-sdk

# Build
cd ~/a733_npu_driver
bash scripts/board/build-ai-sdk.sh --sdk-dir ~/ai-sdk
```

### Run a CNN smoke test

```bash
cd /path/to/yolo_shm  # or wherever the demo NBG is
export LD_LIBRARY_PATH=/path/to/viplite-libs:$LD_LIBRARY_PATH
./vpm_run -s sample.txt -l 1 -d 0
```

Or with the helper:

```bash
A733_VPM_RUN_ARGS="-s sample.txt -l 1 -d 0" \
  bash scripts/board/a733-g0-g1-smoke.sh
```

Success evidence:
- `vpm run ret=0`
- Output contains `cid=0x1000003b`
- VIPLite version line: `VIPLite driver software version 2.0.3.2-AW-2024-08-30`

## Build the persistent runner

The persistent runner (`npu_lm_runner`) is required for LLM inference. It
loads an NBG once and runs repeated forward passes without reloading.

### Radxa Cubie A7Z

```bash
cd ~/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh
```

This auto-detects the VIPLite paths under `/home/radxa/ai-sdk/`.

### Orange Pi Zero 3W

The Orange Pi VIPLite headers/libs are in different locations. Use explicit
paths:

```bash
cd /home/orangepi/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
```

The Orange Pi uses an older VIPLite header variant (no
`VIP_NETWORK_PROP_SET_CORE_INDEX` property). The build script auto-detects
this and applies the correct preprocessor defines
(`-DA733_VIP_LEGACY_DEVICE_ID -DA733_VIP_NO_CORE_INDEX`).

### Verify the runner

```bash
cd ~/a733_npu_driver
bash scripts/board/run-npu-lm-runner.sh --label test
```

Expected output: runner loads NBG, reports `cid=0x1000003b`,
`nbg_loaded_once=1`, generates tokens, exits with status 0.

## Orange Pi: `/dev/vipcore` notes

On the Orange Pi Zero 3W, the NPU module may need to be loaded manually:

```bash
# Check if the module is loaded
lsmod | grep sunxi_npu

# If not, load it
sudo modprobe sunxi_npu

# Verify
ls -l /dev/vipcore
```

The kernel module is part of the Orange Pi BSP; it may not be in the
upstream kernel. If `/dev/vipcore` is absent after boot, check:
- Is the correct device tree overlay loaded?
- Does `dmesg | grep -i npu` show errors?

## Board memory requirements

| Workload | Min free RAM | Notes |
|---|---|---|
| SmolLM2-135M W=32 int16 | ~400 MB | NBG is 281 MB, RSS peaks ~272 MB |
| SmolLM2-360M W=32 int16 | ~900 MB | NBG is 673 MB, RSS peaks ~646 MB |
| SmolLM2-1.7B int16 (if it exported) | >3 GB | Not runnable on 1 GiB boards |
| Qwen2.5-0.5B Q8_0 CPU | ~1.3 GB | For llama.cpp |
| MobileCLIP-S0 vision | ~30 MB | Lightweight |

Check free memory before running:

```bash
free -h
df -h / /home
```

The Radxa Cubie A7Z (1 GiB RAM) is too small for Qwen int16 NBG (>1 GB) or
8-layer Qwen pcq NBG (~370 MB). The Orange Pi Zero 3W (5.7 GiB RAM) can run
all SmolLM2 models and the CPU llama.cpp path.

## Cleanup after NPU runs

The persistent runner and `vpm_run` hold `/dev/vipcore`. Check and release:

```bash
# Check if anyone has /dev/vipcore open
fuser -v /dev/vipcore

# Kill any remaining runner processes
pkill -f npu_lm_runner || true
pkill -f vpm_run || true

# Verify no users remain
fuser /dev/vipcore
```

## Next

- [03-run-llm-npu.md](03-run-llm-npu.md) — Convert and run SmolLM2 on the NPU
- [04-chat-shell.md](04-chat-shell.md) — Interactive chat
- [05-run-vlm-npu.md](05-run-vlm-npu.md) — VLM pipeline on NPU
