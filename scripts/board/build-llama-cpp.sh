#!/usr/bin/env bash
set -euo pipefail

A733_LLAMA_DIR="${A733_LLAMA_DIR:-$HOME/llama.cpp}"
A733_LLAMA_REPO="${A733_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
A733_LLAMA_REF="${A733_LLAMA_REF:-master}"
A733_LLAMA_BUILD_DIR="${A733_LLAMA_BUILD_DIR:-$A733_LLAMA_DIR/build}"
A733_LLAMA_TARGETS="${A733_LLAMA_TARGETS:-llama-cli llama-completion llama-bench llama-simple llama-simple-chat}"

for tool in git cmake g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
done

if [ ! -d "$A733_LLAMA_DIR/.git" ]; then
  mkdir -p "$(dirname "$A733_LLAMA_DIR")"
  git clone --depth 1 "$A733_LLAMA_REPO" "$A733_LLAMA_DIR"
fi

git -C "$A733_LLAMA_DIR" fetch --depth 1 origin "$A733_LLAMA_REF"
git -C "$A733_LLAMA_DIR" checkout FETCH_HEAD

cmake -S "$A733_LLAMA_DIR" -B "$A733_LLAMA_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF

for target in $A733_LLAMA_TARGETS; do
  cmake --build "$A733_LLAMA_BUILD_DIR" --config Release -j "${A733_LLAMA_JOBS:-1}" --target "$target"
done

echo "llama.cpp commit:"
git -C "$A733_LLAMA_DIR" rev-parse HEAD

echo "built binaries:"
find "$A733_LLAMA_BUILD_DIR/bin" -maxdepth 1 -type f -executable -name 'llama-*' -printf '%f\n' | sort
