# 04 — Interactive Chat Shell (SmolLM2 on NPU)

Using the Python chat shell (`chat_shell.py`) to chat with SmolLM2 on the
Orange Pi Zero 3W, with streaming tokens and fixed-window display.

## Prerequisites

- Board NPU is working ([02-board-bringup.md](02-board-bringup.md))
- SmolLM2-135M W=32 int16 NBG deployed ([03-run-llm-npu.md](03-run-llm-npu.md))
- Persistent runner built on the board
- Tokenizer files present on the board
- Python 3 installed on the board

## Build the runner (if not already)

```bash
cd /home/orangepi/a733_npu_driver

bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
```

Expected: `built=/home/orangepi/a733_npu_driver/build/npu_lm_runner`

## Run the chat shell

```bash
cd /home/orangepi/a733_npu_driver

python3 scripts/board/chat_shell.py \
  --model /home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb \
  --tokenizer /home/orangepi/a733_npu_driver/work/models/smollm2-135m-instruct \
  --runner /home/orangepi/a733_npu_driver/build/npu_lm_runner \
  --vip-lib /home/orangepi/lib \
  --window 32 \
  --max-new-tokens 32 \
  --greedy
```

### What happens at startup

```
model=.../network_binary.nb
nbg_size=280,882,632 bytes
device=/dev/vipcore present=True
NPU-only: model layers on NPU, CPU does tokenize/argmax/loop.
vip_init=OK
cid=0x1000003b
create_network_us=147590
prepare_network_us=6594
nbg_loaded_once=1
READY seq_len=32 vocab=49152 temperature=0
```

The NBG is loaded once. All model-layer compute stays on the NPU.

### During chat

| Key | Action |
|---|---|
| Type a message + Enter | Submit prompt, see streamed tokens |
| `/reset` | Clear the conversation window |
| `/exit` | Quit |

The live window counter shows `[window N/32]` as tokens accumulate. Once the
window is full, the shell applies the fixed-window constraint: only the last
32 tokens are sent to the NPU for each subsequent generated token.

Example session:
```
user> Hello, how are you?
[context] fixed window is 32 tokens; using the last 32
[window 32/32] Hello[window 32/32] ,[window 32/32]  I[window 32/32] 'm[window 32/32]  here[window 32/32]  to[window 32/32]  help[window 32/32] ...
[reply] tokens=12 tok_s=20.970 window=32
```

## Chat shell options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | (required) | Path to `network_binary.nb` |
| `--tokenizer PATH` | (required) | Path to HF tokenizer directory |
| `--runner PATH` | `build/npu_lm_runner` | Runner binary path |
| `--vip-lib PATH` | `/home/orangepi/lib` | VIPLite library directory |
| `--window N` | `32` | Fixed window size (must match the NBG) |
| `--max-new-tokens N` | `32` | Max tokens per reply |
| `--temperature F` | `0.0` | Sampling temperature (>0 for random) |
| `--greedy` | off | Use greedy argmax (same as temp=0) |
| `--system TEXT` | ChatML default | System prompt override |

## How it works

1. User message is rendered into SmolLM2 ChatML format:
   `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
2. The rendered prompt is tokenized with the HF tokenizer
3. If the token sequence is longer than `--window`, only the last `W` tokens
   are sent to the NPU (the shell shows `[context] fixed window is W tokens;
   using the last W`)
4. The runner (`npu_lm_runner --protocol`) loads the NBG once and accepts
   `RUN <token-ids>` commands over stdio
5. For each `RUN`, the NPU produces logits; the shell extracts the
   last-position argmax, appends it to the window, and decodes it for display
6. Steps 4-5 repeat until `--max-new-tokens` or an EOS token is reached
7. Tok/s is computed and displayed at the end of each reply

## Troubleshooting

### Runner not found
```
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib
```

### Tokenizer not found
Make sure the Hugging Face tokenizer files are on the board:
```bash
ls ~/a733_npu_driver/work/models/smollm2-135m-instruct/tokenizer.json
```
If missing, copy from host:
```bash
scp -r work/models/smollm2-135m-instruct user@board:~/a733_npu_driver/work/models/
```

### VIPLite library not found
```bash
ls /home/orangepi/lib/libVIPhal.so
```
If missing, locate with `find / -name 'libVIPhal.so' 2>/dev/null` and pass
`--vip-lib <path>`.

### Invalid responses / garbled text
- Ensure you're using the int16 NBG, not pcq
- Ensure `--window` matches the NBG export window size
- Check that the tokenizer matches the model (SmolLM2 uses vocab 49152)

## Next

- [05-run-vlm-npu.md](05-run-vlm-npu.md) — VLM vision pipeline on NPU
- [docs/configurations.md](configurations.md) — All configs at a glance
