#!/bin/bash
# Download remaining Qwen2.5 models
set -euo pipefail
MODEL_DIR="/home/orangepi/a733_npu_driver/models"
mkdir -p "$MODEL_DIR"

DL() {
  local f=$1 url=$2
  local p="$MODEL_DIR/$f"
  [ -f "$p" ] && { echo "EXISTS: $f"; return; }
  echo "DOWNLOAD: $f"
  wget -q --show-progress -O "$p.tmp" "$url" && mv "$p.tmp" "$p"
  echo "DONE: $f ($(ls -lh "$p" | awk '{print $5}'))"
}

# Already done: qwen2.5-0.5b-instruct-q4_k_m.gguf, qwen2.5-0.5b-instruct-q8_0.gguf

DL "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

DL "qwen2.5-1.5b-instruct-q8_0.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"

DL "qwen2.5-3b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

DL "qwen2.5-7b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

echo "ALL DOWNLOADS DONE"
