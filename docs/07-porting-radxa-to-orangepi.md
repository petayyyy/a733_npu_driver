# 07 — Porting from Radxa Cubie A7Z to Orange Pi Zero 3W

What transfers as-is between the two A733 boards and what must be rebuilt.

## Same silicon, same VIPLite

Both boards use the Allwinner A733 SoC with the same Vivante VIP9000 NPU
(`cid=0x1000003b`, single core). Kernel versions and userspace differ:

| | Radxa Cubie A7Z | Orange Pi Zero 3W |
|---|---|---|
| SoC | Allwinner A733 | Allwinner A733 |
| NPU | VIP9000, cid `0x1000003b` | VIP9000, cid `0x1000003b` |
| Kernel | 5.15.147-21-a733 | 6.6.98-sun60iw2 |
| OS | Debian 11 | Debian bookworm |
| RAM | ~1 GiB | 5.7 GiB |
| glibc | 2.31 | 2.36 |

## What transfers as-is

- **NBG files** (`network_binary.nb`): Binary-compatible — same NPU
  microarchitecture, same VIPLite runtime version (2.0.3.2). No
  re-conversion needed.
- **Tokenizer files**: JSON files, portable.
- **Python scripts** (`chat_shell.py`, benchmark scripts): Python 3 only.

## What must be rebuilt on the Orange Pi

### 1. The persistent runner (`npu_lm_runner`)

The C runner links against VIPLite `.so` files. These are glibc-matched
and may differ in ABI between Debian 11 and bookworm.

```bash
cd /home/orangepi/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
```

### 2. VIPLite header variant detection

The Orange Pi image uses a slightly older VIPLite header variant. The
build script auto-detects this and applies:
- `-DA733_VIP_LEGACY_DEVICE_ID`: uses `VIP_NETWORK_PROP_SET_DEVICE_ID`
  instead of `VIP_NETWORK_PROP_SET_DEVICE_INDEX`
- `-DA733_VIP_NO_CORE_INDEX`: `VIP_NETWORK_PROP_SET_CORE_INDEX` is
  missing from the header

The Radxa SDK has both properties. The Orange Pi BSP header has only
`VIP_NETWORK_PROP_SET_DEVICE_ID`.

### 3. Kernel NPU module

The Orange Pi kernel is from the Orange Pi BSP, not the Radxa BSP. The
NPU device tree node and kernel module (`sunxi_npu`) come from the BSP.
`/dev/vipcore` should appear after boot if the module is loaded:

```bash
sudo modprobe sunxi_npu
ls -l /dev/vipcore
```

### 4. llama.cpp (if using CPU baseline)

Must be rebuilt natively on the Orange Pi for the A76 CPU flags:

```bash
cd ~/llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build . --config Release -j $(nproc)
```

## Path differences

| Component | Radxa path | Orange Pi path |
|---|---|---|
| Runner source | `scripts/board/npu_lm_runner.c` (same file) | same |
| VIPLite headers | `~/ai-sdk/viplite-tina/.../v2.0/inc` | `/home/orangepi/yolo_shm` |
| VIPLite libs | `~/ai-sdk/viplite-tina/.../v2.0/` | `/home/orangepi/lib` |
| `vpm_run` | `~/ai-sdk/examples/vpm_run/vpm_run` | `/opt/vpm_run/vpm_run` |
| Models | `~/a733_npu_driver/models/` | `/home/orangepi/a733_npu_driver/models/` |
| LD_LIBRARY_PATH | `~/ai-sdk/viplite-tina/.../v2.0` | `/home/orangepi/lib` |

## Board memory: the big difference

The Radxa has ~1 GiB RAM. The Orange Pi has 5.7 GiB. This matters:

| Workload | Radxa (1 GiB) | Orange Pi (5.7 GiB) |
|---|---|---|
| SmolLM2-135M W=32 (NBG 281 MB) | Runs | Runs |
| SmolLM2-360M W=32 (NBG 673 MB) | Runs | Runs |
| Qwen int16 (NBG 1,065 MB) | Killed (OOM) | Still too large (>1 GiB NBG + runner RSS) |
| Qwen 7-layer pcq (NBG 357 MB) | Runs | Runs |
| Qwen 8-layer pcq (NBG 371 MB) | Killed | Runs |
| Qwen CPU Q8_0 (RSS ~1.2 GB) | Exceeds RAM | Runs comfortably |

## Verification checklist

After porting, verify:

1. `/dev/vipcore` present
2. `cid=0x1000003b` from any `vpm_run` call
3. NBG from Radxa runs unchanged: upload an existing NBG and run it
4. Runner rebuilds and links: `build/npu_lm_runner` exists and is executable
5. Runner produces coherent text: run with a known-good SmolLM2 W=32 int16 NBG
6. `/dev/vipcore` has no users after each run

## Next

- [03-run-llm-npu.md](03-run-llm-npu.md) — Run SmolLM2 on the Orange Pi
- [04-chat-shell.md](04-chat-shell.md) — Interactive chat on Orange Pi
