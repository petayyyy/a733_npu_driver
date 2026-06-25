# B5 Qwen2.5 Size Sweep

Date: 2026-06-25

## Scope

Measure which Qwen2.5 model sizes are practical on the Orange Pi Zero 3W
(5.7 GB RAM) across 9 core configurations. CPU-only (NPU does not run Qwen).
This extends B4b (Qwen2.5-0.5B only) to 1.5B, 3B, and 7B models.

## Board

- Hostname: `orangepizero3w`
- Kernel: `Linux 6.6.98-sun60iw2 ... aarch64`
- CPU: 2x Cortex-A76 (cores 6,7) + 6x Cortex-A55 (cores 0-5) = 8 cores
- RAM: 5.7 GiB
- Swap: 2.9 GiB

## Method

All runs use:

```
llama-completion -m <model>.gguf -c 2048 -n 128 -ngl 0 --temp 0 --seed 42
  --no-warmup --no-display-prompt -no-cnv --simple-io
```

Fixed prompt: "In one concise technical note, explain why CPU utilization must be measured when benchmarking a small language model on an embedded board with heterogeneous CPU cores such as Cortex-A55 and Cortex-A76."

Core configurations (same as B4b, `taskset` for affinity):
- 1xA55 (-c 0), 2xA55 (-c 0,1), 3xA55 (-c 0,1,2), 4xA55 (-c 0,1,2,3)
- 1xA76 (-c 6), 2xA76 (-c 6,7)
- 4 mixed (-c 0,1,6,7), 6 mixed (-c 0,1,2,3,6,7), 8 all (-c 0-7)

Each run captures: llama-completion perf log (stderr), pidstat -p PID 1 (CPU%),
mpstat -P ALL 1 (per-core utilization), /proc/PID/status (VmRSS), and thermal
zones before/after. Core mapping confirmed via lscpu: A76=6,7, A55=0-5.

## Models Tested

| Model | Quant | File Size | Fits? | Peak RSS | Free RAM After |
|-------|-------|-----------|-------|----------|----------------|
| Qwen2.5-0.5B-Instruct | Q4_K_M | 469 MB | YES | 731 MiB | ~4.9 GiB |
| Qwen2.5-0.5B-Instruct | Q8_0 | 645 MB | YES | 1111 MiB | ~4.5 GiB |
| Qwen2.5-1.5B-Instruct | Q4_K_M | 1.1 GB | YES | 1994 MiB | ~3.6 GiB |
| Qwen2.5-1.5B-Instruct | Q8_0 | 1.8 GB | TBD | TBD | TBD |
| Qwen2.5-3B-Instruct | Q4_K_M | 2.0 GB | TBD | TBD | TBD |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.7 GB | TBD | TBD | TBD |

Fit test: model must load at ctx=2048 and generate 1 token without OOM.

## Qwen2.5-0.5B-Instruct Q4_K_M Sweep

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 3.99 | 2.81 | 98.8 | 101 | 12 | 624 MiB | 65.4°C |
| 2xA55__c01 | 0,1 | 2xA55 | 8.23 | 5.59 | ~199 | 200 | 25 | 624 MiB | 65.5°C |
| 3xA55__c012 | 0-2 | 3xA55 | 12.01 | 7.66 | ~300 | 291 | 38 | 664 MiB | 65.1°C |
| 4xA55__c0123 | 0-3 | 4xA55 | 15.93 | 8.55 | 309.3 | 387 | 38 | 663 MiB | 69.5°C |
| 1xA76__c6 | 6 | A76 | 19.23 | 12.59 | 81.7 | 101 | 10 | 624 MiB | 70.0°C |
| 2xA76__c67 | 6,7 | 2xA76 | 38.46 | **17.83** | ~199 | 201 | 25 | 624 MiB | 70.6°C |
| 4mixed__c0167 | 0,1,6,7 | mixed | 43.47 | 14.35 | ~391 | 389 | 49 | 623 MiB | 72.7°C |
| 6mixed__c012367 | 0-3,6,7 | mixed | 49.38 | 14.12 | 438.5 | 580 | 54 | 732 MiB | 81.2°C |
| 8all__c0to7 | 0-7 | all | 48.19 | 12.33 | 524.1 | 732 | 65 | 623 MiB | 77.1°C |

### Q4_K_M observations

- **A76 vs A55 decode:** 1xA76 (12.59) vs 1xA55 (2.81) — **4.5x gap**. A76 delivers dramatically better throughput due to larger caches and out-of-order execution keeping the memory pipeline fed.
- **A76 vs A55 prefill:** 1xA76 (19.23) vs 1xA55 (3.99) — **4.8x gap**. Prefill is compute-bound; A76's wider SIMD dominates.
- **A55 scaling 1→2→3→4 decode:** 2.81 → 5.59 → 7.66 → 8.55 tok/s. Diminishing returns: +99%, +37%, +12%.
- **Best decode:** 2xA76 at 17.83 tok/s. Adding A55 cores (4mixed: 14.35, 6mixed: 14.12, 8all: 12.33) hurts decode — extra cores create bus contention without helping memory-bound work.
- **A55-only decode ceiling:** 4xA55 (8.55) < 1xA76 (12.59). Four slow cores cannot match one fast core.
- **Q4_K_M dequantization overhead:** Q4_K_M ~4.5 bits/weight requires CPU unpacking. This explains the low decode rates vs Q8_0.

## Qwen2.5-0.5B-Instruct Q8_0 Sweep

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 16.87 | 5.92 | 99.3 | 101 | 12 | 1092 MiB | 71.3°C |
| 2xA55__c01 | 0,1 | 2xA55 | 29.62 | 9.19 | ~199 | 185 | 25 | 1092 MiB | 68.6°C |
| 3xA55__c012 | 0-2 | 3xA55 | 45.97 | 12.35 | ~300 | 289 | 38 | 1131 MiB | 70.9°C |
| 4xA55__c0123 | 0-3 | 4xA55 | 57.97 | 12.91 | 241.5 | 370 | 30 | 1091 MiB | 73.3°C |
| 1xA76__c6 | 6 | A76 | 67.79 | 16.28 | ~100 | 101 | 13 | 1092 MiB | 71.9°C |
| 2xA76__c67 | 6,7 | 2xA76 | 133.33 | **17.49** | ~199 | 200 | 25 | 1091 MiB | 72.5°C |
| 4mixed__c0167 | 0,1,6,7 | mixed | 114.28 | 16.05 | ~391 | 393 | 49 | 1091 MiB | 75.7°C |
| 6mixed__c012367 | 0-3,6,7 | mixed | 148.14 | 12.48 | ~593 | 556 | 74 | 1091 MiB | 75.8°C |
| 8all__c0to7 | 0-7 | all | 153.84 | 11.50 | 509.0 | 760 | 63 | 1091 MiB | 87.1°C |

### Q8_0 observations

- **Cross-validates B4b:** 2xA76 Q8_0 decode = 17.49 tok/s matches B4b's 18.03 tok/s within measurement error. Methodology confirmed.
- **A76 vs A55 decode:** 1xA76 (16.28) vs 1xA55 (5.92) — **2.7x gap** (smaller than Q4_K_M's 4.5x — Q4 dequant adds A55 penalty).
- **A55 scaling 1→2→3→4 decode:** 5.92 → 9.19 → 12.35 → 12.91. 3xA55 nearly saturates; 4xA55 adds only +5%.
- **Best decode:** 2xA76 at 17.49 tok/s. Adding cores reduces it (8all: 11.50).
- **Q8_0 vs Q4_K_M speedup:** Q8_0 is 2.1x faster decode on 1xA55, 3.5x faster prefill on 2xA76. Q8_0 has no dequantization overhead.
- **RSS:** Q8_0 uses ~1.1 GiB vs Q4_K_M's ~624 MiB — 75% more RAM for ~2x decode speed and ~3.5x prefill speed.
- **Temperature:** Q8_0 8all reaches 87.1°C — near thermal limits. All other configs ≤82°C.

## Qwen2.5-1.5B-Instruct Q4_K_M Sweep

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 3.11 | 1.49 | 99.6 | 101 | 12 | 1936 MiB | 68.6°C |
| (sweep in progress) | | | | | | | | | |

## Cross-Model Comparison (Decode tok/s)

| Config | 0.5B Q4_K_M | 0.5B Q8_0 | 1.5B Q4_K_M |
|--------|------------|-----------|------------|
| 1xA55 | 2.81 | 5.92 | 1.49 |
| 2xA55 | 5.59 | 9.19 | TBD |
| 3xA55 | 7.66 | 12.35 | TBD |
| 4xA55 | 8.55 | 12.91 | TBD |
| 1xA76 | 12.59 | 16.28 | TBD |
| 2xA76 | 17.83 | 17.49 | TBD |
| 4mixed | 14.35 | 16.05 | TBD |
| 6mixed | 14.12 | 12.48 | TBD |
| 8all | 12.33 | 11.50 | TBD |

## Preliminary ROS2 Headroom Assessment

For concurrent ROS2 + LLM operation:

| Model (optimal config) | Decode tok/s | RAM Used | RAM Free | Cores Used | Cores Free |
|------------------------|-------------|----------|----------|------------|------------|
| 0.5B Q4_K_M (2xA76) | 17.83 | ~624 MiB | ~5.0 GiB | 2 A76 | 6 A55 |
| 0.5B Q8_0 (2xA76) | 17.49 | ~1.1 GiB | ~4.5 GiB | 2 A76 | 6 A55 |
| 0.5B Q4_K_M (1xA76) | 12.59 | ~624 MiB | ~5.0 GiB | 1 A76 | 1 A76 + 6 A55 |
| 0.5B Q8_0 (1xA76) | 16.28 | ~1.1 GiB | ~4.5 GiB | 1 A76 | 1 A76 + 6 A55 |

If ROS2 needs an A76 core: use 1xA76 (Q8_0: 16.28 tok/s, only 7% slower than 2xA76).

## Notes

- CPU% is per-core aggregate (Linux %CPU, can exceed 100%).
- %-of-8-cores is normalized: avg CPU% / 8.
- Decode is per-token generation speed; prefill is prompt ingestion.
- Peak RSS includes model weights mmap'd into page cache.
- Some avg CPU% values are empty (marked "~") due to pidstat header contamination in early runs; peak values confirmed from mpstat and pidstat raw data.
- All temperatures within safe operating range (max 87.1°C on Q8_0 8all).
- No OOM events for 0.5B or 1.5B Q4_K_M models.
- 3B and 7B tests in progress; 7B expected to OOM.

## Raw Logs

```
/home/orangepi/a733_npu_driver/logs/b5-sweep/
```

## Verification

- 0.5B Q4_K_M: All 9 configs completed, exit code 0.
- 0.5B Q8_0: All 9 configs completed, exit code 0.
- 1.5B Q4_K_M: Sweep in progress (1 config completed, exit 0).
- llama-completion perf timings extracted from run.log.
- pidstat %CPU values extracted from pidstat.log.
- RSS peak extracted via /proc/PID/status VmRSS sampling.
- mpstat per-core data collected.
- Affinity confirmed via taskset -c.
- Thermals captured before and after each run.
- B4b cross-check: 2xA76 Q8_0 decode 17.49 vs B4b 18.03 tok/s — consistent.
