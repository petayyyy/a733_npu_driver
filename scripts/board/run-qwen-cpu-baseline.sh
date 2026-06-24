#!/usr/bin/env bash
set -euo pipefail

ROOT="${A733_ROOT:-$HOME/a733_npu_driver}"
LLAMA_BIN_DIR="${A733_LLAMA_BIN_DIR:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${A733_GGUF_DIR:-$ROOT/models/gguf}"
LOG_DIR="${A733_QWEN_LOG_DIR:-$ROOT/logs/board/b4-qwen-cpu-baseline}"
MONITOR="${A733_MONITOR:-$ROOT/scripts/board/monitor_command.py}"

Q4_MODEL="${A733_QWEN_Q4_MODEL:-$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
Q8_MODEL="${A733_QWEN_Q8_MODEL:-$MODEL_DIR/qwen2.5-0.5b-instruct-q8_0.gguf}"
CONTEXTS="${A733_QWEN_CONTEXTS:-2048 8192 16384 32768}"
BENCH_GEN="${A733_QWEN_BENCH_GEN:-64}"
BENCH_REPS="${A733_QWEN_BENCH_REPS:-1}"
THREAD_SWEEP_REPS="${A733_QWEN_THREAD_SWEEP_REPS:-2}"
BENCH_EXTRA_ARGS="${A733_QWEN_BENCH_EXTRA_ARGS:-}"
CHAT_GEN="${A733_QWEN_CHAT_GEN:-96}"
LONG_CTX="${A733_QWEN_LONG_CTX:-16384}"
LONG_NOTES="${A733_QWEN_LONG_NOTES:-280}"
SKIP_THREAD_SWEEP="${A733_QWEN_SKIP_THREAD_SWEEP:-0}"
SKIP_CONTEXT_SWEEP="${A733_QWEN_SKIP_CONTEXT_SWEEP:-0}"
SKIP_CHAT="${A733_QWEN_SKIP_CHAT:-0}"

LLAMA_BENCH="$LLAMA_BIN_DIR/llama-bench"
LLAMA_CLI="$LLAMA_BIN_DIR/llama-cli"
LLAMA_COMPLETION="$LLAMA_BIN_DIR/llama-completion"
export LD_LIBRARY_PATH="$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"

for path in "$LLAMA_BENCH" "$LLAMA_CLI" "$LLAMA_COMPLETION" "$MONITOR" "$Q4_MODEL" "$Q8_MODEL"; do
  if [ ! -e "$path" ]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

{
  echo "date_utc=$(date -u -Iseconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "nproc=$(getconf _NPROCESSORS_ONLN)"
  echo "llama_bin_dir=$LLAMA_BIN_DIR"
  echo "q4_model=$Q4_MODEL"
  echo "q8_model=$Q8_MODEL"
  echo "contexts=$CONTEXTS"
  echo "bench_gen=$BENCH_GEN"
  echo "bench_reps=$BENCH_REPS"
  echo "thread_sweep_reps=$THREAD_SWEEP_REPS"
  echo "bench_extra_args=$BENCH_EXTRA_ARGS"
  echo "long_ctx=$LONG_CTX"
  echo "long_notes=$LONG_NOTES"
  echo "skip_thread_sweep=$SKIP_THREAD_SWEEP"
  echo "skip_context_sweep=$SKIP_CONTEXT_SWEEP"
  echo "skip_chat=$SKIP_CHAT"
} > "$LOG_DIR/env.txt"

lscpu > "$LOG_DIR/lscpu.txt" 2>&1 || true
cat /proc/cpuinfo > "$LOG_DIR/cpuinfo.txt" 2>&1 || true
free -h > "$LOG_DIR/free-before.txt" 2>&1 || true
df -h / /home > "$LOG_DIR/df-before.txt" 2>&1 || true
"$LLAMA_CLI" --version > "$LOG_DIR/llama-cli-version.txt" 2>&1 || true
"$LLAMA_COMPLETION" --version > "$LOG_DIR/llama-completion-version.txt" 2>&1 || true
"$LLAMA_BENCH" --help > "$LOG_DIR/llama-bench-help.txt" 2>&1 || true
"$LLAMA_CLI" --help > "$LOG_DIR/llama-cli-help.txt" 2>&1 || true
"$LLAMA_COMPLETION" --help > "$LOG_DIR/llama-completion-help.txt" 2>&1 || true
sha256sum "$Q4_MODEL" "$Q8_MODEL" > "$LOG_DIR/model-sha256.txt"

run_monitored() {
  local name="$1"
  shift
  echo "RUN $name"
  python3 "$MONITOR" \
    --metrics-json "$LOG_DIR/$name.metrics.json" \
    --stdout "$LOG_DIR/$name.stdout.log" \
    --stderr "$LOG_DIR/$name.stderr.log" \
    -- "$@"
}

run_bench() {
  local quant="$1"
  local model="$2"
  local ctx="$3"
  local threads="$4"
  local cpuset="$5"
  local reps="$6"
  local name="bench-${quant}-ctx${ctx}-t${threads}-cpu${cpuset//,/-}"
  run_monitored "$name" \
    taskset -c "$cpuset" "$LLAMA_BENCH" \
      -m "$model" \
      -p "$ctx" \
      -n "$BENCH_GEN" \
      -t "$threads" \
      -ngl 0 \
      -r "$reps" \
      $BENCH_EXTRA_ARGS \
      -o md
}

if [ "$SKIP_THREAD_SWEEP" != "1" ]; then
  echo "THREAD_SWEEP_Q4_2K"
  run_bench q4 "$Q4_MODEL" 2048 1 6 "$THREAD_SWEEP_REPS"
  run_bench q4 "$Q4_MODEL" 2048 2 6,7 "$THREAD_SWEEP_REPS"
  run_bench q4 "$Q4_MODEL" 2048 4 0-7 "$THREAD_SWEEP_REPS"
  run_bench q4 "$Q4_MODEL" 2048 8 0-7 "$THREAD_SWEEP_REPS"
fi

if [ "$SKIP_CONTEXT_SWEEP" != "1" ]; then
  echo "CONTEXT_SWEEP_A76X2"
  for ctx in $CONTEXTS; do
    run_bench q4 "$Q4_MODEL" "$ctx" 2 6,7 "$BENCH_REPS"
    run_bench q8 "$Q8_MODEL" "$ctx" 2 6,7 "$BENCH_REPS"
  done
fi

if [ "$SKIP_CHAT" != "1" ]; then
  cat > "$LOG_DIR/short-chat.prompt" <<'PROMPT'
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
In one sentence, explain why a real KV-cache matters for long-context CPU inference on a small board.<|im_end|>
<|im_start|>assistant
PROMPT

  python3 - "$LOG_DIR/long-chat.prompt" "$LONG_NOTES" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
note_count = int(sys.argv[2])
sections = []
for i in range(1, note_count + 1):
    code = f"ORANGE-A76-{i:04d}"
    sections.append(
        "Field note {i}: the rover crossed marker {code}. "
        "The battery was stable, the NPU remained reserved for robotics, "
        "and the CPU fallback log repeated that the answer key is {code}."
        .format(i=i, code=code)
    )
text = "\n".join(sections)
out.write_text(
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Read the long field log below. What was the answer key in the final field note, "
    "and why is this a long-context retrieval question?\n\n"
    f"{text}\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n",
    encoding="utf-8",
)
PY
fi

run_chat() {
  local quant="$1"
  local model="$2"
  local ctx="$3"
  local prompt_file="$4"
  local label="$5"
  local name="chat-${label}-${quant}-ctx${ctx}-t2-cpu6-7"
  run_monitored "$name" \
    taskset -c 6,7 "$LLAMA_COMPLETION" \
      -m "$model" \
      -f "$prompt_file" \
      -c "$ctx" \
      -t 2 \
      -n "$CHAT_GEN" \
      -ngl 0 \
      -no-cnv \
      -st \
      --simple-io \
      --no-warmup \
      --no-display-prompt \
      --no-context-shift \
      --color off \
      --temp 0
}

if [ "$SKIP_CHAT" != "1" ]; then
  run_chat q4 "$Q4_MODEL" 2048 "$LOG_DIR/short-chat.prompt" short
  run_chat q8 "$Q8_MODEL" 2048 "$LOG_DIR/short-chat.prompt" short
  run_chat q8 "$Q8_MODEL" "$LONG_CTX" "$LOG_DIR/long-chat.prompt" long
fi

free -h > "$LOG_DIR/free-after.txt" 2>&1 || true
df -h / /home > "$LOG_DIR/df-after.txt" 2>&1 || true
echo "logs=$LOG_DIR"
