#!/usr/bin/env bash
set -euo pipefail

ROOT="${A733_ROOT:-$HOME/a733_npu_driver}"
LOG_DIR="${A733_QWEN_PREP_LOG_DIR:-$ROOT/logs/board/b4-qwen-cpu-baseline-prepare}"
GGUF_DIR="${A733_GGUF_DIR:-$ROOT/models/gguf}"
LLAMA_JOBS="${A733_LLAMA_JOBS:-8}"

Q4_URL="${A733_QWEN_Q4_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
Q8_URL="${A733_QWEN_Q8_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf}"
Q4_MODEL="$GGUF_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
Q8_MODEL="$GGUF_DIR/qwen2.5-0.5b-instruct-q8_0.gguf"

mkdir -p "$LOG_DIR" "$GGUF_DIR"

exec > >(tee -a "$LOG_DIR/prepare.log") 2> >(tee -a "$LOG_DIR/prepare.err" >&2)

echo "date_utc=$(date -u -Iseconds)"
echo "hostname=$(hostname)"
echo "kernel=$(uname -a)"
echo "nproc=$(getconf _NPROCESSORS_ONLN)"
free -h
df -h / /home || true

echo "build llama.cpp targets"
A733_LLAMA_JOBS="$LLAMA_JOBS" \
A733_LLAMA_TARGETS="llama-cli llama-completion llama-bench" \
  bash "$ROOT/scripts/board/build-llama-cpp.sh"

download_if_needed() {
  local url="$1"
  local out="$2"
  if [ -s "$out" ]; then
    echo "already present: $out ($(stat -c '%s' "$out") bytes)"
    return
  fi
  echo "download: $url"
  curl -L -C - -o "$out" "$url"
}

download_if_needed "$Q4_URL" "$Q4_MODEL"
download_if_needed "$Q8_URL" "$Q8_MODEL"

sha256sum "$Q4_MODEL" "$Q8_MODEL" | tee "$LOG_DIR/model-sha256.txt"
ls -lh "$Q4_MODEL" "$Q8_MODEL"
df -h / /home || true
echo "prepare_done=$(date -u -Iseconds)"
