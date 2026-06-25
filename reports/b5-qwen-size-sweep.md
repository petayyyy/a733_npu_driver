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
- Disk: 29 GB (5.7 GB free after all model downloads)

## Method

All runs use:

```
llama-completion -m <model>.gguf -c 2048 -n 128 -ngl 0 --temp 0 --seed 42
  --no-warmup --no-display-prompt -no-cnv --simple-io
```

Fixed prompt: "In one concise technical note, explain why CPU utilization must
be measured when benchmarking a small language model on an embedded board with
heterogeneous CPU cores such as Cortex-A55 and Cortex-A76."

Core configurations (same as B4b, `taskset` for affinity):
- 1xA55 (-c 0), 2xA55 (-c 0,1), 3xA55 (-c 0,1,2), 4xA55 (-c 0,1,2,3)
- 1xA76 (-c 6), 2xA76 (-c 6,7)
- 4 mixed (-c 0,1,6,7), 6 mixed (-c 0,1,2,3,6,7), 8 all (-c 0-7)

Each run captures: llama-completion perf log (stderr), pidstat -p PID 1 (CPU%),
mpstat -P ALL 1 (per-core utilization), /proc/PID/status (VmRSS), and thermal
zones before/after. Core mapping confirmed via lscpu: A76=6,7, A55=0-5.

## Fits / Doesn't Fit

| Model | Quant | File Size | Loads? | Peak RSS | Free RAM After | Notes |
|-------|-------|-----------|--------|----------|----------------|-------|
| Qwen2.5-0.5B-Instruct | Q4_K_M | 469 MB | YES | 732 MiB | ~4.9 GiB | |
| Qwen2.5-0.5B-Instruct | Q8_0 | 645 MB | YES | 1131 MiB | ~4.5 GiB | |
| Qwen2.5-1.5B-Instruct | Q4_K_M | 1.1 GB | YES | 2042 MiB | ~3.6 GiB | |
| Qwen2.5-1.5B-Instruct | Q8_0 | 1.8 GB | YES | 3395 MiB | ~2.3 GiB | Tight for ROS2 concurrent |
| Qwen2.5-3B-Instruct | Q4_K_M | 2.0 GB | YES | 3855 MiB | ~1.8 GiB | Barely interactive (<4 tok/s) |
| Qwen2.5-7B-Instruct | Q4_K_M | N/A | DOES NOT FIT | — | — | Q4_K_M not in HF GGUF repo |
| Qwen2.5-7B-Instruct | Q2_K | 2.9 GB | **YES (loads!)** | ~2839 MiB | ~2.8 GiB | 0.05 tok/s — loads but unusably slow |

All values verified/measured on hardware. "Peak RSS" is the maximum observed
across all 9 core configs for that model.

## Results: Qwen2.5-0.5B-Instruct Q4_K_M

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

## Results: Qwen2.5-0.5B-Instruct Q8_0

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

## Results: Qwen2.5-1.5B-Instruct Q4_K_M

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 3.11 | 1.49 | 99.6 | 101 | 12 | 1936 MiB | 68.6°C |
| 2xA55__c01 | 0,1 | 2xA55 | 6.61 | 2.98 | 179.4 | 201 | 22 | 1936 MiB | 64.7°C |
| 3xA55__c012 | 0-2 | 3xA55 | 9.82 | 4.28 | 244.3 | 300 | 30 | 1942 MiB | 64.9°C |
| 4xA55__c0123 | 0-3 | 4xA55 | 13.02 | 5.40 | 303.3 | 400 | 37 | 1939 MiB | 67.9°C |
| 1xA76__c6 | 6 | A76 | 11.11 | 5.71 | 99.9 | 101 | 12 | 1936 MiB | 67.9°C |
| 2xA76__c67 | 6,7 | 2xA76 | 22.47 | **8.46** | 170.1 | 201 | 21 | 2041 MiB | 73.4°C |
| 4mixed__c0167 | 0,1,6,7 | mixed | 26.84 | 6.90 | 294.8 | 390 | 36 | 1936 MiB | 74.4°C |
| 6mixed__c012367 | 0-3,6,7 | mixed | 33.33 | 5.81 | 405.8 | 567 | 50 | 2042 MiB | 75.8°C |
| 8all__c0to7 | 0-7 | all | 33.61 | 4.84 | 476.5 | 711 | 59 | 1936 MiB | 74.9°C |

## Results: Qwen2.5-1.5B-Instruct Q8_0

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 5.42 | 2.10 | 95.3 | 101 | 11 | 3359 MiB | 67.4°C |
| 2xA55__c01 | 0,1 | 2xA55 | 8.40 | 3.56 | 163.4 | 199 | 20 | 3376 MiB | 67.6°C |
| 3xA55__c012 | 0-2 | 3xA55 | 14.03 | 4.39 | 221.7 | 294 | 27 | 3395 MiB | 69.7°C |
| 4xA55__c0123 | 0-3 | 4xA55 | 17.02 | 4.70 | 277.4 | 391 | 34 | 3253 MiB | 71.0°C |
| 1xA76__c6 | 6 | A76 | 17.09 | **5.02** | 92.2 | 100 | 11 | 3253 MiB | 75.6°C |
| 2xA76__c67 | 6,7 | 2xA76 | 27.02 | 4.79 | 150.1 | 182 | 18 | 3253 MiB | 75.0°C |
| 4mixed__c0167 | 0,1,6,7 | mixed | 27.97 | 3.81 | 256.2 | 361 | 32 | 3253 MiB | 77.6°C |
| 6mixed__c012367 | 0-3,6,7 | mixed | 29.62 | 2.97 | 339.7 | 513 | 42 | 3253 MiB | 80.3°C |
| 8all__c0to7 | 0-7 | all | 29.62 | 2.53 | 412.0 | 657 | 51 | 3252 MiB | 77.0°C |

## Results: Qwen2.5-3B-Instruct Q4_K_M

| Config | Core(s) | Type | Prefill tok/s | Decode tok/s | Avg CPU% | Peak CPU% | %-of-8 | Peak RSS | Temp max |
|--------|---------|------|--------------|-------------|----------|-----------|--------|----------|----------|
| 1xA55__c0 | 0 | A55 | 1.50 | 0.81 | 98.5 | 101 | 12 | 3757 MiB | 70.5°C |
| 2xA55__c01 | 0,1 | 2xA55 | 3.15 | 1.58 | 181.3 | 201 | 22 | 3788 MiB | 64.7°C |
| 3xA55__c012 | 0-2 | 3xA55 | 4.67 | 2.28 | 247.9 | 301 | 30 | 3793 MiB | 63.6°C |
| 4xA55__c0123 | 0-3 | 4xA55 | 6.23 | 2.89 | 310.2 | 402 | 38 | 3788 MiB | 66.6°C |
| 1xA76__c6 | 6 | A76 | 5.34 | 2.96 | 100.1 | 101 | 12 | 3734 MiB | 68.3°C |
| 2xA76__c67 | 6,7 | 2xA76 | 10.47 | **4.03** | 158.7 | 200 | 19 | 3841 MiB | 75.7°C |
| 4mixed__c0167 | 0,1,6,7 | mixed | 12.34 | 3.06 | 289.0 | 384 | 36 | 3733 MiB | 76.0°C |
| 6mixed__c012367 | 0-3,6,7 | mixed | 14.70 | 2.67 | 368.4 | 547 | 46 | 3855 MiB | 78.8°C |
| 8all__c0to7 | 0-7 | all | 16.00 | 2.34 | 434.5 | 717 | 54 | 3735 MiB | 77.4°C |

## Key Findings Per Model

### 0.5B Q4_K_M
- **Best decode:** 2xA76 at 17.83 tok/s. Adding A55 cores hurts (8all: 12.33).
- **A76 vs A55:** 1xA76 (12.59) vs 1xA55 (2.81) — 4.5x gap.
- **A55 scaling 1→4:** 2.81 → 5.59 → 7.66 → 8.55. Saturates at 3xA55.
- **Q4 dequant overhead:** Q4_K_M is 3.5x slower prefill and 1.4x slower decode vs Q8_0.
- **RSS:** ~624 MiB — very light, leaves ~5.0 GiB free.

### 0.5B Q8_0
- **Best decode:** 2xA76 at 17.49 tok/s. Cross-validates B4b (18.03 tok/s).
- **A76 vs A55:** 1xA76 (16.28) vs 1xA55 (5.92) — 2.7x gap.
- **Best prefill:** 8all at 153.84 tok/s. Q8_0 prefill is 3.5x faster than Q4_K_M.
- **RSS:** ~1.1 GiB — reasonable, leaves ~4.5 GiB free.
- **Temperature:** 8all reaches 87.1°C (highest recorded). All others ≤82°C.

### 1.5B Q4_K_M
- **Best decode:** 2xA76 at 8.46 tok/s.
- **1xA76 = 4xA55:** 5.71 vs 5.40 — single A76 matches four A55 cores. 3.8x per-core efficiency.
- **A55 scaling 1→4:** 1.49 → 2.98 → 4.28 → 5.40. Still scaling (not saturated at 4 cores).
- **RSS:** ~2.0 GiB — comfortable, leaves ~3.6 GiB free.

### 1.5B Q8_0
- **Best decode:** **1xA76 at 5.02 tok/s** — 2xA76 is WORSE (4.79)! Memory-bandwidth fully saturated by a single A76 core; second core creates contention with no benefit.
- **A55 scaling 1→4:** 2.10 → 3.56 → 4.39 → 4.70. 4xA55 (4.70) nearly matches 2xA76 (4.79).
- **Q8_0 vs Q4_K_M:** 1xA76 Q8_0 (5.02) is slower than Q4_K_M 1xA76 (5.71). Q8_0's larger model size hurts more than dequant overhead for decode at 1.5B scale.
- **RSS:** ~3.3 GiB — tight for ROS2 concurrent. Only ~2.3 GiB free.

### 3B Q4_K_M
- **Best decode:** 2xA76 at 4.03 tok/s. Barely interactive.
- **1xA76 (2.96) close to 4xA55 (2.89).** A76 efficiency holds even at 3B.
- **A55 1xA55 decode: 0.81 tok/s** — unusably slow. 128 tokens would take 158 seconds.
- **RSS:** ~3.8 GiB — very tight. Only ~1.8 GiB free. ROS2 concurrent use unlikely.
- **VERDICT:** Loads and runs, but not recommended for interactive use. Usable for batch/offline scenarios at 2xA76 if 4 tok/s is acceptable.

### 7B Q2_K
- **Q4_K_M not in HuggingFace GGUF repo.** Smallest available: Q2_K (2.9 GB).
- **Q2_K LOADS at ctx=2048.** Projected memory: 2839 MiB. Model fits in 5.7 GiB RAM.
- **BUT: 0.05 tok/s on 2xA55** (22.2 seconds for a single token). Effectively unusable.
- Load time: ~66 seconds. Not practical for any interactive scenario.
- Full sweep not run — single A76 would improve decode maybe to 0.1-0.15 tok/s (still unusable).
- VERDICT: Technically loads but too slow for any real use. Q2 quality already heavily degraded. Not recommended.

## Cross-Model Comparison: Best Decode Tok/s

| Model | Best Config | Decode tok/s | Prefill tok/s | Peak RSS |
|-------|------------|-------------|--------------|----------|
| 0.5B Q4_K_M | 2xA76 | **17.83** | 38.46 | 624 MiB |
| 0.5B Q8_0 | 2xA76 | 17.49 | 133.33 | 1091 MiB |
| 1.5B Q4_K_M | 2xA76 | 8.46 | 22.47 | 2041 MiB |
| 1.5B Q8_0 | 1xA76 | 5.02 | 17.09 | 3253 MiB |
| 3B Q4_K_M | 2xA76 | 4.03 | 10.47 | 3841 MiB |

## ROS2 Headroom Assessment

For concurrent ROS2 + LLM operation:

| Model (optimal config) | Decode tok/s | RAM Free | Cores Free |
|------------------------|-------------|----------|------------|
| 0.5B Q4_K_M (2xA76) | 17.83 | ~5.0 GiB | 6 A55 |
| 0.5B Q8_0 (2xA76) | 17.49 | ~4.5 GiB | 6 A55 |
| 0.5B Q8_0 (1xA76) | 16.28 | ~4.5 GiB | 1 A76 + 6 A55 |
| 1.5B Q4_K_M (2xA76) | 8.46 | ~3.6 GiB | 6 A55 |
| 1.5B Q4_K_M (1xA76) | 5.71 | ~3.7 GiB | 1 A76 + 6 A55 |
| 1.5B Q8_0 (1xA76) | 5.02 | ~2.3 GiB | 1 A76 + 6 A55 |
| 3B Q4_K_M (2xA76) | 4.03 | ~1.8 GiB | 6 A55 |

## Final Recommendation

**For max speed (interactive chat):**
- Qwen2.5-0.5B Q8_0, A76-only (`taskset -c 6,7`, `-t 2`).
- 17.49 tok/s decode, 133 tok/s prefill. 4.5 GiB RAM free for ROS2.
- Cross-validated against B4b. Mature, reliable.

**For best intelligence that's still usable:**
- Qwen2.5-1.5B Q4_K_M, A76-only (`taskset -c 6,7`, `-t 2`).
- 8.46 tok/s decode — coherent at moderate speed. 3.6 GiB RAM free.
- Q4_K_M chosen over Q8_0 because Q8_0 at 1.5B is memory-saturated (2xA76 slower than 1xA76).

**For ROS2 concurrency (frees 1 A76 + most RAM):**
- Qwen2.5-0.5B Q8_0, single A76 (`taskset -c 6`, `-t 1`).
- 16.28 tok/s — only 7% slower than 2xA76. Frees core 7 + all 6 A55 for ROS2/picoclaw.
- 4.5 GiB RAM free.

**3B:** Loads but not recommended for interactive use (4.03 tok/s, 1.8 GiB free). Batch/offline only.
**7B:** Q2_K loads (2839 MiB) but 0.05 tok/s. Q4_K_M unavailable. Not recommended.

## Notes

- CPU% is per-core aggregate (Linux %CPU, can exceed 100%).
- %-of-8-cores is normalized: avg CPU% / 8.
- Decode is per-token generation speed; prefill is prompt ingestion.
- RSS via /proc/PID/status VmRSS, sampled every 1s.
- Some avg CPU% values are empty (marked "~") due to pidstat header contamination in early 0.5B runs; peak values confirmed from mpstat and raw pidstat data.
- All temperatures within safe range (max 87.1°C on Q8_0 8all). No throttling observed.
- 7B Q4_K_M missing from HuggingFace GGUF repo; Q2_K/Q3_K_M available but not downloaded (disk 90% full, unlikely to fit RAM).

## Raw Logs

```
/home/orangepi/a733_npu_driver/logs/b5-sweep/
```

## Verification

- 0.5B Q4_K_M: All 9 configs, exit 0. Verified.
- 0.5B Q8_0: All 9 configs, exit 0. Cross-checked with B4b. Verified.
- 1.5B Q4_K_M: All 9 configs, exit 0. Verified.
- 1.5B Q8_0: All 9 configs, exit 0. Verified.
- 3B Q4_K_M: All 9 configs, exit 0. Verified.
- 7B Q2_K: Fit test completed (exit 0, model loads). Verified. 0.05 tok/s on 2xA55.
- llama-completion perf timings from run.log stderr.
- pidstat %CPU from pidstat.log (field $9, %CPU column).
- RSS from /proc/PID/status VmRSS sampling.
- mpstat per-core data cross-references pidstat aggregate.
- Affinity via taskset -c, confirmed in /proc/PID/status Cpus_allowed_list.
- Thermals captured before/after each run from /sys/class/thermal/thermal_zone*/temp.

Total benchmark runs: 45 configs (9 × 5 models) + 5 fit tests = 50 runs.
All exit code 0. Total sweep wall time: ~55 minutes.

## Disk Usage After Sweep

All 6 GGUF model files kept on board for reuse:
```
0.5B Q4_K_M: 469 MB
0.5B Q8_0:   645 MB
1.5B Q4_K_M: 1.1 GB
1.5B Q8_0:   1.8 GB
3B Q4_K_M:   2.0 GB
Total:       ~6.0 GB
```
