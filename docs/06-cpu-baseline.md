# 06 — CPU Baseline: Qwen2.5-0.5B with llama.cpp

Running Qwen2.5-0.5B-Instruct on the Orange Pi Zero 3W CPU through
llama.cpp with a real KV-cache. This is a diagnostic fallback path, not
the NPU-only target; it answers: "if ROS2 is paused and the A76 cores
are available, what can Qwen do on this board?"

## Prerequisites

- Orange Pi Zero 3W with internet access
- `build-essential`, `cmake`, `curl`, `git` installed
- A76 CPU IDs identified: `taskset -c 6,7` (default; check `lscpu`)

## Step 1: Build llama.cpp

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout be4a6a63
mkdir build && cd build
cmake .. \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build . --config Release -j $(nproc)
```

Verify:
```bash
./bin/llama-bench --version
./bin/llama-cli --version
./bin/llama-completion --version
```

## Step 2: Download Qwen GGUF

```bash
cd ~/a733_npu_driver
bash scripts/board/prepare-qwen-cpu-baseline.sh
```

This downloads:
- `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- `qwen2.5-0.5b-instruct-q8_0.gguf`

to `~/a733_npu_driver/models/gguf/` and runs SHA256 verification.

Or manually:
```bash
curl -L -o models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

## Step 3: Quick smoke test

```bash
Q8_MODEL=~/a733_npu_driver/models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf

# llama-bench at 2k context (A76 cores, 2 threads)
taskset -c 6,7 ~/llama.cpp/build/bin/llama-bench \
  -m "$Q8_MODEL" \
  -p 2048 -n 64 -t 2 -ngl 0 -r 1 -o md
```

Expected: prefill ~48 tok/s, decode ~12 tok/s, RSS ~1.2 GB.

## Step 4: Benchmark sweep

```bash
cd ~/a733_npu_driver
bash scripts/board/run-qwen-cpu-baseline.sh
```

This runs:
1. Thread sweep (Q4_K_M at 2k): 1/2/4/8 threads
2. Context sweep (Q4_K_M and Q8_0 at 2k/8k/16k/32k)
3. Real chat tests (short prompt + long 16k-context demo)

The script is configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `A733_QWEN_CONTEXTS` | `2048 8192 16384 32768` | Context sizes to sweep |
| `A733_QWEN_BENCH_GEN` | `64` | Generated tokens per benchmark |
| `A733_QWEN_SKIP_THREAD_SWEEP` | `0` | Set to `1` to skip |
| `A733_QWEN_SKIP_CONTEXT_SWEEP` | `0` | Set to `1` to skip |
| `A733_QWEN_SKIP_CHAT` | `0` | Set to `1` to skip |
| `A733_QWEN_LONG_CTX` | `16384` | Long-context chat ctx size |
| `A733_QWEN_CHAT_GEN` | `96` | Generated tokens in chat |

## Step 5: Interactive chat on CPU

```bash
Q8_MODEL=~/a733_npu_driver/models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf

taskset -c 6,7 ~/llama.cpp/build/bin/llama-completion \
  -m "$Q8_MODEL" \
  -c 8192 \
  -t 2 \
  -ngl 0 \
  --no-warmup \
  --temp 0 \
  --simple-io
```

Note: this llama.cpp revision requires `llama-completion`, not `llama-cli`,
for non-interactive completion. Use explicit Qwen chat markup:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Your question here<|im_end|>
<|im_start|>assistant
```

## Measured performance

### llama-bench (synthetic tg64 decode)

| Quant | Context | Prefill tok/s | Decode tok/s (tg64) | First-token (est.) | Peak RSS |
|---|---|---|---|---|---|
| Q8_0 | 2,048 | 47.8 | 11.7 | ~43 s | 1,192 MiB |
| Q8_0 | 8,192 | 22.1 | 11.5 | ~6 min | 1,201 MiB |
| Q8_0 | 16,384 | 13.3 | 12.1 | ~21 min | 1,306 MiB |
| Q4_K_M | 2,048 | 18.0 | 10.9 | ~114 s | 734 MiB |
| Q4_K_M | 8,192 | 12.6 | 10.7 | ~11 min | 737 MiB |
| Q4_K_M | 16,384 | 9.2 | 11.0 | ~30 min | 838 MiB |

### Real chat (directly measured)

| Quant | Context | First stdout | Wall time | Prompt eval | Decode eval | Peak RSS |
|---|---|---|---|---|---|---|
| Q8_0 short | 2k | 2.96 s | 5.18 s | 131.5 tok/s | 18.4 tok/s | 1,196 MiB |
| Q8_0 long | 16k | 1,120 s (18.7 min) | 1,159 s | 14.2 tok/s | 2.2 tok/s | 1,309 MiB |

### Important caveats

- **Q8_0 is faster than Q4_K_M** on this board — counterintuitively.
  Suspect AMX/SIMD soft-permute overhead in K-quants on A76.
- **Real-chat decode at 16k drops to 2.2 tok/s** — `llama-bench tg64` is
  a short synthetic run; real decode over a long KV-cache is much slower.
- **16k tail retrieval is unreliable** — the near-16k chat demo retrieved
  the final key incorrectly (`ORANGE-A76-0100` vs expected
  `ORANGE-A76-0280`). 16k is a capacity point, not a guaranteed reliability
  point.
- **32k is impractical** — estimated first-token wait of ~69-81 minutes.
  Fits in RAM but not useful interactively.
- **All-core runs hurt decode** — 8 threads gave better prefill but
  degraded decode. Stick to A76-only (`taskset -c 6,7`, `-t 2`).

## Recommendation

For the "ROS2 paused" fallback:
```
taskset -c 6,7 llama-completion \
  -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -c 8192 \
  -t 2 \
  -ngl 0 \
  --no-warmup \
  --temp 0
```

Use 8k as the practical default. Use 16k only when the long context value
is worth a ~19-minute first-output wait and the task can tolerate validation
or retry.

## NPU vs CPU trade-off

| Property | NPU (SmolLM2-135M) | CPU (Qwen-0.5B Q8_0) |
|---|---|---|
| Throughput | 21 tok/s | 18 tok/s (2k) / 2.2 tok/s (16k) |
| Context | Fixed window 32-64 tokens | Real KV-cache, thousands of tokens |
| Model quality | SmolLM2-135M | Qwen2.5-0.5B (bigger, more capable) |
| CPU usage | ~0% (NPU does the work) | 2 A76 cores at 100% |
| Works alongside ROS2 | Yes | No (needs A76 cores paused) |

## Next

- [docs/configurations.md](configurations.md) — When to use CPU vs NPU
- [08-known-limits-and-blockers.md](08-known-limits-and-blockers.md) — Limits
