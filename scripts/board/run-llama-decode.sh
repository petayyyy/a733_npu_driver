#!/usr/bin/env bash
set -euo pipefail

A733_LLAMA_BIN_DIR="${A733_LLAMA_BIN_DIR:-$HOME/llama.cpp/build/bin}"
A733_GGUF_MODEL="${A733_GGUF_MODEL:-}"
A733_LLAMA_LOG_DIR="${A733_LLAMA_LOG_DIR:-logs/board/llama-decode}"
A733_LLAMA_USER_PROMPT="${A733_LLAMA_USER_PROMPT:-Write one short sentence about embedded AI.}"
A733_LLAMA_SYSTEM_PROMPT="${A733_LLAMA_SYSTEM_PROMPT:-You are a concise embedded AI assistant.}"
A733_LLAMA_N_PREDICT="${A733_LLAMA_N_PREDICT:-64}"
A733_LLAMA_BENCH_PROMPT="${A733_LLAMA_BENCH_PROMPT:-128}"
A733_LLAMA_BENCH_GEN="${A733_LLAMA_BENCH_GEN:-64}"
A733_LLAMA_BENCH_THREADS="${A733_LLAMA_BENCH_THREADS:-1,2,4,8}"
A733_LLAMA_BENCH_REPS="${A733_LLAMA_BENCH_REPS:-3}"

if [ -z "$A733_GGUF_MODEL" ]; then
  echo "A733_GGUF_MODEL must point to a GGUF model file" >&2
  exit 1
fi

if [ ! -f "$A733_GGUF_MODEL" ]; then
  echo "model not found: $A733_GGUF_MODEL" >&2
  exit 1
fi

for exe in llama-simple llama-bench; do
  if [ ! -x "$A733_LLAMA_BIN_DIR/$exe" ]; then
    echo "missing executable: $A733_LLAMA_BIN_DIR/$exe" >&2
    exit 1
  fi
done

mkdir -p "$A733_LLAMA_LOG_DIR"
export LD_LIBRARY_PATH="$A733_LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"

if [ -n "${A733_LLAMA_CHAT_PROMPT:-}" ]; then
  chat_prompt="$A733_LLAMA_CHAT_PROMPT"
else
  chat_prompt="$(printf '<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$A733_LLAMA_SYSTEM_PROMPT" "$A733_LLAMA_USER_PROMPT")"
fi

"$A733_LLAMA_BIN_DIR/llama-bench" \
  -m "$A733_GGUF_MODEL" \
  -p "$A733_LLAMA_BENCH_PROMPT" \
  -n "$A733_LLAMA_BENCH_GEN" \
  -t "$A733_LLAMA_BENCH_THREADS" \
  -ngl 0 \
  -r "$A733_LLAMA_BENCH_REPS" \
  -o md \
  > "$A733_LLAMA_LOG_DIR/llama-bench.md" 2>&1

"$A733_LLAMA_BIN_DIR/llama-simple" \
  -m "$A733_GGUF_MODEL" \
  -n "$A733_LLAMA_N_PREDICT" \
  -ngl 0 \
  "$chat_prompt" \
  > "$A733_LLAMA_LOG_DIR/llama-simple-chatprompt.txt" 2>&1

if [ -x "$A733_LLAMA_BIN_DIR/llama-simple-chat" ]; then
  {
    printf '%s\n' "$A733_LLAMA_USER_PROMPT"
  } | timeout "${A733_LLAMA_CHAT_TIMEOUT:-90}" \
    "$A733_LLAMA_BIN_DIR/llama-simple-chat" \
      -m "$A733_GGUF_MODEL" \
      -c "${A733_LLAMA_CHAT_CONTEXT:-512}" \
      -ngl 0 \
      > "$A733_LLAMA_LOG_DIR/llama-simple-chat.txt" 2>&1 || true
fi

echo "logs written to $A733_LLAMA_LOG_DIR"
