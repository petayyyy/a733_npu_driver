# B2 Chat Shell

Date: 2026-06-24

## Scope

Task B2 requested a clean interactive chat wrapper for the working
SmolLM2-135M-Instruct int16 NBG on the Orange Pi Zero 3W. The required path is
NPU-only for model-layer compute, with CPU limited to tokenization,
fixed-window orchestration, sampling/argmax, detokenization, and terminal I/O.

## Implementation

Verified changes:

- `scripts/board/npu_lm_runner.c` now supports `--protocol`, a stdio line
  protocol that loads/prepares the NBG once and then accepts repeated
  `RUN <W token ids>` commands.
- Each protocol response returns one generated token plus timing:
  `TOKEN id=<id> profile_us=<us> cycle=<cycles> wall_us=<us> top5=<...>`.
- `scripts/board/chat_shell.py` is the user-facing REPL. It starts one runner
  process, applies the SmolLM2 ChatML-style prompt, tokenizes from the model's
  HF `tokenizer.json`, slides the fixed window, streams decoded text as tokens
  arrive, shows live `used / W` counters, prints reply tok/s, supports
  `--model`, `--tokenizer`, `--window`, `--max-new-tokens`,
  `--temperature`, `--greedy`, and `/reset`.
- The shell attempts to use the official `tokenizers` Python package when it
  is installed; on the verified Orange Pi image it used the built-in
  `tokenizer.json` reader already present in the repo.

## Board Validation

Verified before each NPU launch:

- `/dev/vipcore` had no users.
- No unrelated `npu_lm_runner`, `vpm_run`, `llama`, `monitor_command.py`,
  `chat_shell.py`, `cmake`, or `ninja` jobs were active.

Verified runner rebuild on the Orange Pi:

```bash
cd /home/orangepi/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
```

Verified final scripted REPL command:

```bash
cd /home/orangepi/a733_npu_driver
printf 'Write a friendly one sentence greeting.\nNow make it shorter.\n/reset\nWrite a tiny greeting.\n/exit\n' |
python3 scripts/board/chat_shell.py \
  --model /home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb \
  --tokenizer /home/orangepi/a733_npu_driver/work/models/smollm2-135m-instruct \
  --window 32 \
  --max-new-tokens 12 \
  --greedy
```

Verified final board log:

```text
/home/orangepi/a733_npu_driver/logs/board/b2-chat-shell-20260624T073253Z-final.log
```

Startup excerpt:

```text
model=/home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb
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

Transcript excerpt:

```text
user> assistant>
[context] fixed window is 32 tokens; using the last 32
"[window 32/32]
Hello[window 32/32]
,[window 32/32]
 welcome[window 32/32]
 to[window 32/32]
 our[window 32/32]
 chat[window 32/32]
.[window 32/32]
 I[window 32/32]
'm[window 32/32]
 here[window 32/32]
 to[window 32/32]
[reply] tokens=12 tok_s=20.970 window=32

user> assistant>
[context] fixed window is 32 tokens; using the last 32
"[window 32/32]
Hello[window 32/32]
,[window 32/32]
 welcome[window 32/32]
 to[window 32/32]
 our[window 32/32]
 chat[window 32/32]
.[window 32/32]
 I[window 32/32]
'm[window 32/32]
 here[window 32/32]
 to[window 32/32]
[reply] tokens=12 tok_s=20.919 window=32

user> context reset

user> assistant>
[context] fixed window is 32 tokens; using the last 32
Hello[window 32/32]
![window 32/32]
 I[window 32/32]
'm[window 32/32]
 here[window 32/32]
 to[window 32/32]
 help[window 32/32]
 you[window 32/32]
 with[window 32/32]
 your[window 32/32]
 homework[window 32/32]
.[window 32/32]
[reply] tokens=12 tok_s=20.999 window=32
```

Verified after validation:

- `/dev/vipcore` had no users.
- No `npu_lm_runner` or `chat_shell.py` process remained.

## Result

B2 passes for the working SmolLM2-135M-Instruct W=32 int16 NBG on the Orange Pi
Zero 3W. A human can run one command, chat interactively, see streamed tokens,
see the fixed-window counter and tok/s, use `/reset`, and keep the NBG loaded
once through the persistent runner protocol.
