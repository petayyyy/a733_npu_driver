#!/bin/bash
# V2d: End-to-end hybrid VLM: NPU vision + CPU LLM via mmproj
set -euo pipefail

# Usage: run-v2d-e2e.sh <image.jpg> [prompt]
IMAGE="${1:?Usage: $0 <image.jpg> [prompt]}"
PROMPT="${2:-Describe this image.}"
N_GEN="${3:-128}"

# Paths
VLM_DIR="/home/orangepi/a733_npu_driver/models/vlm"
NBG_DIR="/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16"
VIPM_RUN="/opt/vpm_run/vpm_run"
LLAMA_CLI="/home/orangepi/llama.cpp/build/bin/llama-cli"
MMPROJ="$VLM_DIR/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
MODEL="$VLM_DIR/SmolVLM-256M-Instruct-Q8_0.gguf"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/orangepi/lib}"
TMPDIR="/tmp/v2d_e2e_$$"

mkdir -p "$TMPDIR"
echo "=== V2d E2E Hybrid VLM ==="
echo "Image: $IMAGE"
echo "Prompt: $PROMPT"
echo ""

# Step 0: Check files exist
for f in "$VIPM_RUN" "$LLAMA_CLI" "$MMPROJ" "$MODEL" "$NBG_DIR/network_binary.nb"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Step 1: Preprocess image for NPU (Python normalization)
echo "=== Step 1: Preprocessing image ==="
python3 -c "
from PIL import Image
import numpy as np, struct, sys

img = Image.open('$IMAGE').convert('RGB')
img = img.resize((512, 512), Image.BICUBIC)
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - 0.5) / 0.5
arr = arr.transpose(2, 0, 1)
scale = 2.0 ** 15
int16_arr = np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)
with open('$TMPDIR/npu_input.dat', 'wb') as f:
    int16_arr.tofile(f)
print(f'Input: {int16_arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}], fl=15')
" 2>&1 || { echo "Image preprocessing failed"; exit 1; }

# Step 2: Run NPU vision encoder
echo ""
echo "=== Step 2: NPU Vision Encode ==="
export LD_LIBRARY_PATH

printf '[network]\n./network_binary.nb\n[input]\n./npu_input.dat\n' > "$NBG_DIR/sample_v2d.txt"
cp "$TMPDIR/npu_input.dat" "$NBG_DIR/npu_input.dat"

cd "$NBG_DIR"
START_TIME=$(date +%s%3N)
$VIPM_RUN -s sample_v2d.txt -b 0 --save_txt 1 > "$TMPDIR/vpm_run.log" 2>&1
NPU_RC=$?
END_TIME=$(date +%s%3N)
NPU_MS=$((END_TIME - START_TIME))

echo "NPU exit: $NPU_RC, wall: ${NPU_MS}ms"
grep -E "profile inference time|vpm run ret|prepare network|create network" "$TMPDIR/vpm_run.log" || true

if [ $NPU_RC -ne 0 ] || [ ! -f output_0.txt ]; then
    echo "ERROR: NPU run failed"
    cat "$TMPDIR/vpm_run.log"
    exit 1
fi

# Step 3: Convert NPU output to float32 binary
echo ""
echo "=== Step 3: Converting NPU output ==="
N_VALUES=$(wc -l < output_0.txt)
echo "Output values: $N_VALUES (expected 36864)"

python3 -c "
import struct
vals = []
with open('output_0.txt') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: vals.append(float(line))
        except: pass
print(f'Parsed {len(vals)} values, range [{min(vals):.4f}, {max(vals):.4f}]')
with open('$TMPDIR/embeddings.bin', 'wb') as f:
    for v in vals:
        f.write(struct.pack('<f', float(v)))
print(f'Saved embeddings.bin ({len(vals)*4} bytes)')
" 2>&1

cd /

# Step 4: Run llama-cli with NPU embeddings
echo ""
echo "=== Step 4: LLM Decode with NPU vision ==="
echo "Prompt: $PROMPT"
echo ""

EMBEDDINGS_FILE="$TMPDIR/embeddings.bin"

# Verify embeddings file
if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "ERROR: embeddings.bin not found"
    exit 1
fi
EMB_SIZE=$(stat -c%s "$EMBEDDINGS_FILE")
echo "Embeddings file: $EMB_SIZE bytes"

echo ""
echo "=== Answer ==="

export A733_NPU_EMBEDDINGS="$EMBEDDINGS_FILE"
export LD_LIBRARY_PATH="/home/orangepi/llama.cpp/build/bin"

START_LLM=$(date +%s%3N)
$LLAMA_CLI \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    --chat-template "smolvlm" \
    -n $N_GEN \
    -t 2 \
    --temp 0.0 \
    --no-conversation \
    2>"$TMPDIR/llama_stderr.log" | tee "$TMPDIR/llama_stdout.log"
LLAMA_RC=$?
END_LLM=$(date +%s%3N)
LLM_MS=$((END_LLM - START_LLM))

echo ""
echo "=== Timing ==="
echo "NPU vision: ${NPU_MS}ms"
echo "LLM total: ${LLM_MS}ms"
echo "Llama exit: $LLAMA_RC"

# Cleanup temp
rm -rf "$TMPDIR"
exit $LLAMA_RC
