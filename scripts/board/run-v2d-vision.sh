#!/bin/bash
set -euo pipefail
# V2d: Run SmolVLM vision encoder NBG on NPU with custom image input

NBG_DIR="${1:-/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16}"
INPUT_DAT="${2:-$NBG_DIR/dog_input.dat}"
OUTPUT_DIR="${3:-$NBG_DIR/output}"
VIPM_RUN="${VIPM_RUN:-/opt/vpm_run/vpm_run}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/orangepi/lib}"

mkdir -p "$OUTPUT_DIR"

# Create sample.txt pointing to network_binary.nb and our input
cat > "$NBG_DIR/sample_dog.txt" <<EOF
[network]
./network_binary.nb
[input]
./dog_input.dat
EOF

echo "=== V2d Vision NBG Run ==="
echo "NBG: $NBG_DIR/network_binary.nb"
echo "Input: $INPUT_DAT"
echo ""

export LD_LIBRARY_PATH
cd "$NBG_DIR"

$VIPM_RUN sample_dog.txt > "$OUTPUT_DIR/run_dog.log" 2>&1
RC=$?

echo "Exit code: $RC"
cat "$OUTPUT_DIR/run_dog.log"

# Check for output
if [ -f output_0.txt ]; then
    lines=$(wc -l < output_0.txt)
    echo ""
    echo "Output lines: $lines (expected 36864 for 64x576)"
    cp output_0.txt "$OUTPUT_DIR/output_0_dog.txt"
    echo "Output saved to $OUTPUT_DIR/output_0_dog.txt"
fi

# Also save float32 binary for embedding injection
if [ -f output_0.txt ] && [ -f "$NBG_DIR/nbg_meta.json" ]; then
    echo ""
    echo "=== Converting output to float32 binary ==="
    python3 -c "
import json, struct
with open('$NBG_DIR/nbg_meta.json') as f:
    meta = json.load(f)
out_info = list(meta['Outputs'].values())[0]['quantize']
fl = int(out_info['fl'])
scale = 2.0 ** fl

with open('output_0.txt') as f:
    lines = f.readlines()

floats = []
for line in lines:
    line = line.strip()
    if ':' in line:
        line = line.split(':', 1)[1]
    for x in line.replace(',', ' ').split():
        try:
            floats.append(float(x) / scale)
        except:
            pass

print(f'Dequantized {len(floats)} values (fl={fl}, scale={scale})')

with open('$OUTPUT_DIR/embeddings_dog_f32.bin', 'wb') as f:
    for v in floats:
        f.write(struct.pack('<f', v))
print(f'Saved: $OUTPUT_DIR/embeddings_dog_f32.bin')
"
fi

exit $RC
