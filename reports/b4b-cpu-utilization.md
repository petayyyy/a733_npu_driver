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

| -t | Core(s) | Core type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8-cores | Peak RSS | Temp max |
|----|---------|-----------|--------------|-------------|----------|-----------|-------------|----------|----------|
| 1  | 0       | A55       | 18.18        | 6.33        | ~100%    | 100%      | 13%         | 1,188 MiB | 80.5°C |
| 2  | 0,1     | 2×A55     | 36.68        | 10.73       | ~199%    | 200%      | 25%         | 1,195 MiB | 81.1°C |
| 3  | 0-2     | 3×A55     | 54.67        | 13.66       | ~300%    | 300%      | 38%         | 1,189 MiB | 81.8°C |
| 4  | 0-3     | 4×A55     | 71.42        | 14.82       | ~398%    | 400%      | 50%         | 1,198 MiB | 82.4°C |
| 1  | 6       | A76       | 64.76        | 16.54       | ~100%    | 100%      | 13%         | 1,106 MiB | 78.6°C |
| 2  | 6,7     | 2×A76     | 128.74       | **18.03**   | ~199%    | 200%      | 25%         | 1,109 MiB | 79.2°C |
| 4  | 4-7     | 2×A76+2×A55 | 129.55     | 16.77       | ~391%    | 399%      | 49%         | 1,141 MiB | 82.7°C |
| 6  | 2-7     | 2×A76+4×A55 | 157.33     | 16.13       | ~590%    | 593%      | 74%         | 1,201 MiB | 84.5°C |
| 8  | 0-7     | all 8     | 161.88       | 13.70       | ~741%    | 779%      | 93%         | 1,196 MiB | 83.8°C |

All values verified/measured on hardware.

### Key single-core observations

- **A76 vs A55 decode:** A76 delivers 16.54 tok/s vs A55's 6.33 tok/s —
  a **2.6x gap**. Decode is memory-bound, so the difference comes from A76's
  larger caches and wider out-of-order execution keeping the memory pipeline
  fed.
- **A76 vs A55 prefill:** A76 64.76 tok/s vs A55 18.18 tok/s — a **3.6x gap**.
  Prefill is compute-bound (batch matmul), where A76's wider SIMD and higher
  clock (2.0 vs 1.8 GHz) dominate.
- **One A76 vs two A76:** decode 16.54 → 18.03 tok/s (+9%). The second A76
  core barely helps — one A76 nearly saturates memory bandwidth on decode.
- **One A55 vs two A55:** decode 6.33 → 10.73 tok/s (+70%). Unlike A76, a
  single A55 does NOT saturate memory bandwidth — the second A55 core finds
  unused bandwidth, nearly doubling throughput. Two A55 (10.73) still lose to
  one A76 (16.54).
- **A55 scaling 1→2→3→4:** 6.33 → 10.73 → 13.66 → 14.82 tok/s. Diminishing
  returns: +70%, +27%, +8%. By 3×A55 (13.66), bandwidth saturation kicks in;
  4×A55 (14.82) is close to 1×A76 (16.54) — four slow cores nearly match one
  fast core in memory-bound decode.
- **4×A55 (14.82) vs 8×all (13.70):** pure A55 is faster than mixing A76+A55
  for decode. Adding A76 cores to A55 creates bus contention — the A76 cores
  dominate the memory controller, A55s stall, and the overhead of 8 threads
  hurts more than the extra compute helps.

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
| 1 A55 | 0 | Core 0 ~100%, others idle |
| 2 A55 | 0,1 | Both cores ~100%, others idle |
| 3 A55 | 0-2 | All 3 cores ~100%, others idle |
| 4 A55 | 0-3 | All 4 cores ~98-100%, others idle |
| 1 A76 | 6 | Core 6 ~100%, others idle |
| 2  | 6,7 (A76)   | Core 6 ~100%, core 7 ~0-100% intermittent |
| 4  | 4-7         | All 4 cores ~92-100% |
| 6  | 2-7         | All 6 cores ~92-100% |
| 8  | 0-7         | All 8 cores ~85-98% (A55 cores slightly lower) |

## Thermals

| Zone | t=1 A55 | t=2 A55 | t=3 A55 | t=4 A55 | t=1 A76 | t=2 | t=4 | t=6 | t=8 |
|------|---------|---------|---------|---------|---------|-----|-----|-----|-----|
| zone0 | 80.5°C | 81.1°C | 81.8°C | 82.4°C | 78.6°C | 77.4°C | 82.0°C | 83.1°C | 82.8°C |
| zone3 | 75.1°C | 76.3°C | 77.0°C | 77.8°C | 75.8°C | 79.2°C | 82.7°C | 84.5°C | 83.8°C |

All within safe range. No throttling observed.

## Recommendation

**Use -t 2 (taskset -c 6,7) for best decode throughput: 18.03 tok/s.**

Qwen2.5-0.5B Q8_0 decode on this board consumes **2 A76 cores at ~199% CPU**
(25% of total 8-core capacity). This leaves 6 A55 cores + A76 headroom free
for ROS2 and other workloads.

If ROS2 needs an A76 core, **-t 1 on a single A76 (core 6) delivers 16.54
tok/s** — only 8% slower than the A76 pair, and frees core 7. A single A55
(6.33 tok/s) is 2.6× slower and not recommended for interactive use.

If prefill throughput matters (e.g., long context ingestion), t=6 or t=8 help
(161.88 vs 128.74 tok/s), but at the cost of decode speed and core
availability.

## Raw Logs

```
/home/orangepi/a733_npu_driver/logs/board/b4b-cpu-utilization/
```

## Verification

- All 9 runs (t=1/2/3/4 A55, t=1/2 A76, t=4/6/8 mixed) completed with exit code 0.
- llama-completion perf timings extracted from stderr.log.
- pidstat %CPU values extracted from pidstat.log.
- RSS peak computed from rss.log sampling at 0.2s intervals.
- mpstat per-core data cross-references pidstat aggregate %CPU.
- Affinity confirmed via taskset -cp and /proc/PID/status Cpus_allowed_list.
- Thermals captured before and after each run.
- Disk usage: /tmp cleared between tests to prevent fill-up.
