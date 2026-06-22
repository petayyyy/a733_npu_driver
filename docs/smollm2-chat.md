# SmolLM2 Chat On Radxa A733 NPU

This note explains how to run the `SmolLM2-135M-Instruct` `W=64` int16 NPU chat wrapper on the Radxa Cubie A7Z.

The user-facing entry point is:

```text
scripts/board/smollm2_chat.py
```

It accepts normal text, renders the SmolLM2 chat template, tokenizes with the real `tokenizer.json`, runs the persistent VIPLite NPU runner, and decodes the generated token ids back to text.

## What Runs Where

Model-layer compute runs inside the NBG on the A733 NPU:

- token embedding `Gather`
- RMSNorm
- RoPE
- GQA attention
- SwiGLU MLP
- residuals
- final logits

CPU work is orchestration only:

- prompt formatting
- tokenization/detokenization
- calling the VIPLite runner
- sliding the fixed token window
- argmax decoding
- terminal I/O

## Prerequisites

On the Radxa board, these paths are expected:

```text
/home/radxa/a733_npu_driver/build/npu_lm_runner
/home/radxa/a733_npu_driver/models/smollm2_135m_w64_int16/network_binary.nb
/home/radxa/a733_npu_driver/work/models/smollm2-135m-instruct/tokenizer.json
/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

Build the runner if needed:

```bash
cd /home/radxa/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh
```

If the chat script or tokenizer is missing on the board, copy them from the host:

```bash
scp scripts/board/smollm2_chat.py \
  radxa@192.168.31.76:/home/radxa/a733_npu_driver/scripts/board/

ssh radxa@192.168.31.76 \
  "mkdir -p /home/radxa/a733_npu_driver/work/models/smollm2-135m-instruct"

scp work/models/smollm2-135m-instruct/tokenizer.json \
  radxa@192.168.31.76:/home/radxa/a733_npu_driver/work/models/smollm2-135m-instruct/tokenizer.json
```

## Single Prompt

Run one message and print one answer:

```bash
cd /home/radxa/a733_npu_driver

python3 scripts/board/smollm2_chat.py \
  "What is the capital of France?" \
  --steps 32 \
  --show-metrics
```

Expected shape of output:

```text
User: What is the capital of France?
Assistant:
The capital of France is Paris, ...
mean_wall_us=...
mean_profile_us=...
mean_tok_s=...
```

## Interactive Mode

Start a simple terminal chat:

```bash
cd /home/radxa/a733_npu_driver

python3 scripts/board/smollm2_chat.py --steps 32 --show-metrics
```

Then type messages at the `User:` prompt. Exit with `/exit`, `/quit`, or `Ctrl-D`.

## Configuration

Common options:

```bash
python3 scripts/board/smollm2_chat.py --help
```

Useful knobs:

```text
--model-dir PATH      NBG package directory. Default: W=64 int16 SmolLM2.
--tokenizer PATH      Path to tokenizer.json.
--runner PATH         Path to build/npu_lm_runner.
--vip-lib PATH        VIPLite shared library directory.
--seq-len N           Fixed NPU context window. Default: 64.
--steps N             Number of generated tokens. Default: 32.
--system TEXT         System message.
--no-system           Omit the system message.
--show-metrics        Print runner timing metrics.
--show-tokens         Print prompt and generated token ids.
--verbose-runner      Print the full low-level runner log.
```

For example, use a shorter answer:

```bash
python3 scripts/board/smollm2_chat.py \
  "Give me one sentence about the A733 NPU." \
  --steps 16
```

Use a custom system message:

```bash
python3 scripts/board/smollm2_chat.py \
  "Explain this in simple words." \
  --system "You answer briefly and plainly." \
  --steps 32
```

## Current Limitations

- The working real-model path is `int16`; the `pcq` int8 SmolLM2 path converts and runs, but fails the coherence check.
- The current graph is fixed-window `W=64`; it recomputes the full 64-token window for every generated token.
- There is no KV-cache.
- The Python chat wrapper launches the persistent C runner once per assistant answer. Inside one answer, the NBG is loaded once and reused for all generated tokens.
- Interactive conversation history is truncated to the last 64 tokens before each answer.

## Troubleshooting

If the runner is missing:

```bash
bash scripts/board/build-npu-lm-runner.sh
```

If VIPLite libraries are not found, pass the path explicitly:

```bash
python3 scripts/board/smollm2_chat.py \
  "Hello" \
  --vip-lib /home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

If the model is somewhere else:

```bash
python3 scripts/board/smollm2_chat.py \
  "Hello" \
  --model-dir /path/to/smollm2_135m_w64_int16
```

If the tokenizer is somewhere else:

```bash
python3 scripts/board/smollm2_chat.py \
  "Hello" \
  --tokenizer /path/to/tokenizer.json
```
