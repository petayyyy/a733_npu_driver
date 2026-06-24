# B4 Qwen CPU Baseline

Date: 2026-06-24

## Scope

Verified: this report is the B4 diagnostic CPU fallback baseline for
Qwen2.5-0.5B-Instruct on the Orange Pi Zero 3W at `192.168.31.225`.

This is intentionally CPU-only llama.cpp work with a real KV cache. It does not
replace the active NPU-only requirement for normal LLM/VLM project milestones.
It answers the fallback question: if ROS2 is paused/frozen and the A76 CPU cores
are available, what can Qwen do on this board?

## Board And Build

Verified board:

- Hostname: `orangepizero3w`
- Kernel: `Linux 6.6.98-sun60iw2 ... aarch64`
- RAM: `5.7 GiB`
- CPU: 6x Cortex-A55 at max `1794 MHz` plus 2x Cortex-A76 at max `2002 MHz`
- A76 CPU IDs: `6,7`
- CPU flags include NEON and dot-product: `asimd`, `asimddp`

Verified before board runs: no unrelated `llama-*`, `monitor_command.py`,
`vpm_run`, `npu_lm_runner`, `cmake`, or `ninja` jobs were active, and
`docker ps` was empty.

Verified llama.cpp build:

```text
commit: be4a6a63eb2b848e19c277bdcf2bd399e8af76d9
version line: version: 1 (be4a6a6)
compiler: GNU 12.2.0 for Linux aarch64
cmake flags: GGML_NATIVE=ON, GGML_OPENMP=ON, LLAMA_BUILD_EXAMPLES=ON
detected CPU flags include: -mcpu=cortex-a76.cortex-a55+crypto+dotprod+noi8mm+nosve
```

Verified GGUF models from the official Qwen GGUF repository:

```text
repo: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
Q4_K_M: qwen2.5-0.5b-instruct-q4_k_m.gguf
sha256: 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
Q8_0: qwen2.5-0.5b-instruct-q8_0.gguf
sha256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
```

## Method

Verified bench command shape:

```bash
taskset -c 6,7 llama-bench \
  -m qwen2.5-0.5b-instruct-<quant>.gguf \
  -p <context> -n 64 -t 2 -ngl 0 -r 1 -o md
```

For the 16k continuation pass, `--no-warmup` was used so the board would not
spend another full prompt pass on a warmup run. Throughput rows are still
reported from llama.cpp's measured `pp` and `tg` lines. The 2k/8k rows used the
default warmup path.

Important measurement note: `llama-bench` does not stream the first generated
token. Therefore the table's first-token column is an estimate from measured
throughput:

```text
first_token_ms_est = context_tokens / prefill_tok_s * 1000 + 1000 / decode_tok_s
```

The real chat section below contains measured first-output timing from the
corrected `monitor_command.py` pipe monitor.

## Thread Sweep

Verified with Q4_K_M at 2k context. This sweep was done to confirm the useful
thread shape before the context sweep.

| CPU set | Threads | Prefill pp2048 | Decode tg64 | Peak RSS | Status |
|---|---:|---:|---:|---:|---|
| `6` | 1 | 16.69 tok/s | 12.72 tok/s | 732 MiB | measured |
| `6,7` | 2 | 18.03 tok/s | 10.92 tok/s | 734 MiB | measured |
| `0-7` | 4 | 21.33 tok/s | 10.11 tok/s | 733 MiB | measured |
| `0-7` | 8 | 22.67 tok/s | 7.80 tok/s | 733 MiB | measured |

Interpretation: all-core runs improve prompt ingestion but hurt decode. One A76
thread is best for pure decode in this Q4 test; two A76 threads are the
balanced setting used for the Q4/Q8 context sweep.

## Context Sweep

All measured context rows below use `taskset -c 6,7`, `-t 2`, `-ngl 0`, and
`-n 64`.

| Quant | Context | Prefill | Decode | First-token | Peak RSS | Status |
|---|---:|---:|---:|---:|---:|---|
| Q4_K_M | 2,048 | 18.03 tok/s | 10.92 tok/s | 113,680 ms | 734 MiB | pp/tg/RSS measured, TTFT estimated |
| Q4_K_M | 8,192 | 12.64 tok/s | 10.66 tok/s | 648,195 ms | 737 MiB | pp/tg/RSS measured, TTFT estimated |
| Q4_K_M | 16,384 | 9.23 tok/s | 11.03 tok/s | 1,775,172 ms | 838 MiB | pp/tg/RSS measured, TTFT estimated |
| Q4_K_M | 32,768 | ~6.74 tok/s | ~11.0 tok/s | ~4,861,852 ms | ~1,040 MiB | estimated, run started then stopped as impractical |
| Q8_0 | 2,048 | 47.84 tok/s | 11.67 tok/s | 42,895 ms | 1,192 MiB | pp/tg/RSS measured, TTFT estimated |
| Q8_0 | 8,192 | 22.13 tok/s | 11.50 tok/s | 370,263 ms | 1,201 MiB | pp/tg/RSS measured, TTFT estimated |
| Q8_0 | 16,384 | 13.27 tok/s | 12.13 tok/s | 1,234,747 ms | 1,306 MiB | pp/tg/RSS measured, TTFT estimated |
| Q8_0 | 32,768 | ~7.96 tok/s | ~12.1 tok/s | ~4,118,112 ms | ~1,516 MiB | estimated from 8k->16k scaling |

Verified outcome: 16k is the maximum measured practical context in this run.
The 32k rows are not RAM-blocked on a 5.7 GiB board, but the expected
first-token wait is about 69-81 minutes at the measured scaling. Q4 32k was
started and stopped after it was clear it would not produce a useful timely
result; no 32k row is reported as measured.

Surprising but verified: Q8_0 is faster than Q4_K_M for this Qwen build on this
board. It is also the higher-quality quantization, so it is the preferred CPU
fallback quant despite the higher RSS.

Important decode caveat: the `tg64` values above are llama-bench's synthetic
generation test. The real 15,908-token chat run below verified that decode over
a near-16k KV cache slows much more: `2.19 tok/s` for Q8_0.

## Real Chat Runs

The chat runs used `llama-completion`, not `llama-cli`, because this llama.cpp
revision prints:

```text
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

Prompts still used explicit Qwen chat markup:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
```

Short-prompt measured runs after fixing first-byte monitoring:

| Quant | Context | First stdout | Wall | Peak RSS | Prompt eval | Decode eval | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Q4_K_M | 2,048 | 3,263 ms | 5,740 ms | 731 MiB | 38.59 tok/s | 19.27 tok/s | measured |
| Q8_0 | 2,048 | 2,958 ms | 5,182 ms | 1,196 MiB | 131.49 tok/s | 18.43 tok/s | measured |
| Q8_0 | 16,384 | 1,120,438 ms | 1,158,594 ms | 1,309 MiB | 14.23 tok/s | 2.19 tok/s | measured |

Short Q8_0 transcript:

```text
A real KV-cache significantly enhances the performance and efficiency of long-context CPU inference on a small board by providing fast access to frequently used data, improving memory utilization, and reducing latency. [end of text]
```

Long-context Q8_0 demo:

Prompt shape:

```text
280 synthetic field notes, 55,052 bytes, 287 lines
llama.cpp prompt tokens: 15,908
Expected final field-note key: ORANGE-A76-0280
```

Timing:

```text
first stdout: 1,120,438 ms (18.67 min)
wall time: 1,158,594 ms (19.31 min)
peak RSS: 1,339,908 KB (1,309 MiB)
prompt eval: 1,117,647.19 ms / 15,908 tokens, 14.23 tok/s
decode eval: 37,819.14 ms / 83 runs, 2.19 tok/s
```

Transcript:

```text
: This question is a long-context retrieval question. It asks for the answer key in the final field note, which is the 280th field note in the long field log. The question is asking for the answer key in the last field note, which is the 280th field note in the long field log. The answer key is ORANGE-A76-0100. [end of text]
```

Verified quality result: the answer is coherent but the retrieved key is wrong.
The prompt's final field note key was `ORANGE-A76-0280`; the model answered
`ORANGE-A76-0100`. This makes 16k a measured capacity/performance point, not a
passed reliability point for tail retrieval.

## Recommendation

For ROS2-frozen mode on this board, use Qwen2.5-0.5B-Instruct `Q8_0` through
llama.cpp CPU with:

```bash
taskset -c 6,7 llama-completion \
  -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -c 8192 \
  -t 2 \
  -ngl 0 \
  --no-warmup \
  --temp 0
```

Use 8k as the practical default: it gives a real KV cache and useful context
without the 16k first-token wait. Use 16k only when the long-context value is
worth a roughly 19 minute first-output wait and the task can tolerate validation
or retry; the measured near-16k tail retrieval demo answered the exact final
key incorrectly. Treat 32k as possible in memory but impractical for interactive
use on this CPU.

If the workload is decode-only after the prompt is already ingested, one A76
thread can be faster for Q4 decode. For full chat turns, two A76 threads are the
better balanced setting. All-core runs can help prompt prefill, but they consume
the A55 cores too and reduced decode throughput in the measured sweep.

## NPU Contrast

Verified CPU fallback advantage: llama.cpp gives the full Qwen runtime, a real
KV cache, and thousands of tokens of usable context on the Orange Pi CPU. The
near-16k retrieval miss above is also verified, so app-specific long-context
accuracy still needs validation.

Verified NPU path advantage from the existing project reports: SmolLM2
W=32/W=64 int16 runs on the Orange Pi NPU and keeps the A76 cores free for
robotics/orchestration, but it is fixed-window and does not provide a full
llama.cpp-style KV cache or Qwen-quality long context.

Operational framing:

- ROS2 frozen or paused -> use this Qwen CPU path when long context matters.
- ROS2 running -> use the NPU SmolLM2 path so the A76 cores stay available.

## Raw Logs

Verified board log roots:

```text
/home/orangepi/a733_npu_driver/logs/board/b4-qwen-cpu-baseline-prepare
/home/orangepi/a733_npu_driver/logs/board/b4-qwen-cpu-baseline
/home/orangepi/a733_npu_driver/logs/board/b4-qwen-cpu-baseline-long
/home/orangepi/a733_npu_driver/logs/board/b4-qwen-cpu-baseline-chat
```

Local archive after collection:

```text
logs/board/b4-qwen-cpu-baseline.tar.gz
```

## Result

B4 passed as a CPU diagnostic baseline: Qwen2.5-0.5B-Instruct GGUF runs on the
Orange Pi CPU through llama.cpp with measured Q4_K_M/Q8_0 throughput through
16k context, an honest 32k impractical estimate, and a real near-16k chat demo.
The recommended ROS2-frozen fallback is Q8_0, A76-only `taskset -c 6,7`,
`-t 2`, and an 8k default context.
