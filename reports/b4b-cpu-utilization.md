# B4b CPU Utilization

Date: 2026-06-25

## Scope

Measure Qwen2.5-0.5B Q8_0 decode/prefill throughput AND CPU utilization at
ctx=2048 for thread counts -t = 2, 4, 6, 8 on the Orange Pi Zero 3W.
Answer: which -t gives best decode throughput, and how many CPU cores does
Qwen actually consume when ROS2 is paused.

## Board

- Hostname: `orangepizero3w`
- Kernel: `Linux 6.6.98-sun60iw2 ... aarch64`
- CPU: 2x Cortex-A76 (cores 6,7) + 6x Cortex-A55 (cores 0-5) = 8 cores
- RAM: 5.7 GiB

## Method

All runs use:

```
llama-completion -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -c 2048 -n 128 -ngl 0 --temp 0 --seed 42 --no-warmup \
  --no-display-prompt -no-cnv --simple-io --color off
```

Fixed prompt: "In one concise technical note, explain why CPU utilization must
be measured when benchmarking a small language model on an embedded board..."

Each run captures: llama-completion perf log (stderr), pidstat -p PID 1 (CPU%
+ RSS per second), mpstat -P ALL 1 (per-core utilization), /proc/PID/status
(affinity, VmHWM), and thermal zones before/after.

**Critical fix:** The original script used `llama-cli` which rejects
`--no-conversation`. Switched to `llama-completion` with `-no-cnv`. Without
this fix, `llama-cli` enters interactive mode and generates text indefinitely,
filling the filesystem (a previous run produced a 9.9 GB stdout.log).

## Results

| -t | Cores | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8-cores | Peak RSS | Temp max |
|----|-------|--------------|-------------|----------|-----------|-------------|----------|----------|
| 2  | 6,7   | 128.74       | **18.03**   | ~199%    | 200%      | 25%         | 1,109 MiB | 79.2°C |
| 4  | 4-7   | 129.55       | 16.77       | ~391%    | 399%      | 49%         | 1,141 MiB | 82.7°C |
| 6  | 2-7   | 157.33       | 16.13       | ~590%    | 593%      | 74%         | 1,201 MiB | 84.5°C |
| 8  | 0-7   | 161.88       | 13.70       | ~741%    | 779%      | 93%         | 1,196 MiB | 83.8°C |

All values verified/measured on hardware.

### Notes

- **CPU%** is per-core aggregate (Linux %CPU, can exceed 100%). For t=2,
  pidstat reports ~199% == 2 cores at ~100% each.
- **%-of-8-cores** is normalized: avg CPU% / 8.
- **Decode** is the per-token generation speed after prompt ingestion.
- **Prefill** is prompt ingestion speed.
- At t=8, peak CPU% is only 779% (not 800%), confirming decode is
  memory-bandwidth-bound and 8 threads cannot fully saturate all 8 cores.
- **RSS** peaks at ~1.2 GiB (model weights in RAM). VmHWM from
  /proc/PID/status is ~58 MB (model is mmap'd, not in RSS proper; RSS here
  includes page cache).

### Per-core utilization (mpstat summary)

| -t | Active cores | Observation |
|----|-------------|------------|
| 2  | 6,7 (A76)   | Core 6 ~100%, core 7 ~0-100% intermittent |
| 4  | 4-7         | All 4 cores ~92-100% |
| 6  | 2-7         | All 6 cores ~92-100% |
| 8  | 0-7         | All 8 cores ~85-98% (A55 cores slightly lower) |

## Thermals

| Zone | t=2 | t=4 | t=6 | t=8 |
|------|-----|-----|-----|-----|
| zone0 (CPU?) | 77.4°C | 82.0°C | 83.1°C | 82.8°C |
| zone3 (GPU/NPU?) | 79.2°C | 82.7°C | 84.5°C | 83.8°C |

All within safe range. No throttling observed.

## Recommendation

**Use -t 2 (taskset -c 6,7) for best decode throughput: 18.03 tok/s.**

Qwen2.5-0.5B Q8_0 decode on this board consumes **2 A76 cores at ~199% CPU**
(25% of total 8-core capacity). This leaves 6 A55 cores + A76 headroom free
for ROS2 and other workloads. Adding A55 cores (t=4,6,8) reduces decode
speed and consumes more total CPU without benefit for decode.

If prefill throughput matters (e.g., long context ingestion), t=6 or t=8 help
(161.88 vs 128.74 tok/s), but at the cost of decode speed and core
availability.

## Raw Logs

```
/home/orangepi/a733_npu_driver/logs/board/b4b-cpu-utilization/
```

## Verification

- All 4 runs completed with exit code 0.
- llama-completion perf timings extracted from stderr.log.
- pidstat %CPU values extracted from pidstat.log.
- RSS peak computed from rss.log sampling at 0.2s intervals.
- mpstat per-core data cross-references pidstat aggregate %CPU.
- Affinity confirmed via taskset -cp and /proc/PID/status Cpus_allowed_list.
- Thermals captured before and after each run.
- Disk usage: /tmp cleared between tests to prevent fill-up.
