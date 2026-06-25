# 03 — Run LLM on NPU: SmolLM2

How to convert and run SmolLM2-135M/360M-Instruct on the A733 NPU with
int16 quantization, fixed-window decode, and coherent output.

## What this achieves

- Full 30-layer SmolLM2 decoder runs on the NPU
- NPU-only: embeddings, attention, MLP, RMSNorm, and logits are in the NBG graph
- CPU handles only: tokenizer/detokenizer, fixed-window sliding, argmax
- Coherent text at W=32 and W=64; incoherent at W≥128 (no KV-cache)

## The fixed-window concept

The NBG graph is static-shape. It takes a window of exactly `W` token IDs
(where `W` is fixed at export time) and produces logits for the next token.
There is no KV-cache: every token generated requires a full `W`-token
recompute through all 30 decoder layers. The window slides by removing the
oldest token and appending the newly generated token.

**Practical consequence**: throughput is O(W²) and coherence degrades above
W=64 because no long-range attention state is preserved.

## Prerequisites

- Host: [01-setup-host.md](01-setup-host.md) completed
- Board: [02-board-bringup.md](02-board-bringup.md) completed
- SmolLM2-135M-Instruct checkpoint from Hugging Face:
  - Download `config.json`, `model.safetensors`, `tokenizer.json`,
    `tokenizer_config.json`, `generation_config.json` to
    `work/models/smollm2-135m-instruct/`
- SmolLM2-360M-Instruct (optional, for the larger model):
  - Same files under `work/models/smollm2-360m-instruct/`
  - Use `scripts/host/download_b1_smollm2_models.py` to download

## Step 1: Generate the ONNX graph

### 135M at W=32 (recommended starting point)

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/smollm2-135m-instruct \
    --output-dir work/generated/smollm2_135m_w32 \
    --seq-len 32
```

### 135M at W=64

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/smollm2-135m-instruct \
    --output-dir work/generated/smollm2_135m_w64 \
    --seq-len 64
```

### 360M at W=32

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/smollm2-360m-instruct \
    --output-dir work/generated/smollm2_360m_w32 \
    --seq-len 32
```

### Verify ONNX builder correctness (optional)

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/compare_onnxruntime_to_oracle.py \
    --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
    --tokens work/generated/smollm2_135m_w32/token_ids.npy \
    --oracle work/generated/smollm2_135m_w32_oracle/fp_oracle.npz \
    --model-info work/generated/smollm2_135m_w32/model_info.json \
    --output-json logs/host/smollm2-135m-w32-onnxruntime-vs-fp.json
```

Expected: cosine = 1.000000000, top-1 match.

## Step 2: Convert to int16 NBG

```bash
# With Docker resource limits (recommended for large models)
DOCKER_RUN_ARGS="--cpus 10 --memory 24g" \
  scripts/host/convert_onnx_to_nbg.sh \
    --name smollm2_135m_w32_int16 \
    --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
    --dataset work/generated/smollm2_135m_w32/dataset.txt \
    --quant int16 \
    --inputs token_ids \
    --input-size-list 32 \
    --outputs logits
```

Expected: `Error(0),Warning(0)` in the export log.

Output package:
```
work/model-packages/smollm2_135m_w32_int16/int16/
├── network_binary.nb    (~281 MB for 135M W=32, ~673 MB for 360M W=32)
├── nbg_meta.json
├── input_0.dat
├── output_0.txt
└── sample.txt
```

### Conversion summary for all working configs

| Model | W | ONNX size | NBG size | Conversion time (with 10 CPU/24 GB RAM) |
|---|---|---|---|---|
| SmolLM2-135M | 32 | ~538 MB | 281 MB | ~15 min |
| SmolLM2-135M | 64 | ~652 MB | 282 MB | ~15 min |
| SmolLM2-135M | 128 | ~652 MB | 287 MB | ~16 min |
| SmolLM2-135M | 256 | ~652 MB | 337 MB | ~18 min |
| SmolLM2-360M | 32 | ~1.2 GB | 673 MB | ~30 min |
| SmolLM2-360M | 64 | ~1.2 GB | 675 MB | ~30 min |

W=128 and W=256 export but produce **incoherent** output on the board. They
are listed for completeness only.

## Step 3: Upload to board

```bash
# Create models directory on board
ssh user@board "mkdir -p ~/a733_npu_driver/models/smollm2_135m_w32_int16"

# Upload NBG package
scp -r work/model-packages/smollm2_135m_w32_int16/int16/* \
  user@board:~/a733_npu_driver/models/smollm2_135m_w32_int16/

# Also upload tokenizer
scp -r work/models/smollm2-135m-instruct \
  user@board:~/a733_npu_driver/work/models/
```

## Step 4: Run on board

### Verify the NBG loads and runs

```bash
cd ~/a733_npu_driver

# Ensure runner is built (see 02-board-bringup.md)
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out build/npu_lm_runner

# Run a quick smoke test
export A733_VIP_LIB=/home/orangepi/lib
bash scripts/board/run-npu-lm-runner.sh \
  --model models/smollm2_135m_w32_int16/network_binary.nb
```

Expected output:
```
vip_init=OK
cid=0x1000003b
create_network_us=<...>
prepare_network_us=<...>
nbg_loaded_once=1
<generated text>
```

### Coherent vs incoherent configs

| Model | W | Coherent? | Decode tok/s | First-token ms | Peak RSS MiB |
|---|---|---|---|---|---|
| SmolLM2-135M | 32 | yes (16/16 FP token match) | 20.7 | 48 | 272 |
| SmolLM2-135M | 64 | yes (weak) | 14.0 | 72 | 274 |
| SmolLM2-135M | 128 | no | 6.0 | 166 | 282 |
| SmolLM2-135M | 256 | no | 1.2 | 860 | 375 |
| SmolLM2-360M | 32 | yes | 8.4 | 114 | 646 |
| SmolLM2-360M | 64 | yes | 4.9 | 212 | 649 |
| SmolLM2-360M | 128 | no | 2.0 | 502 | 681 |
| SmolLM2-360M | 256 | no | 1.2 | 834 | 711 |

**Recommendation**: use 135M/W32 for fastest interactive chat, 360M/W32 for
smarter responses. Avoid W≥128 unless you want to study the coherence cliff.

## Int8 (pcq) does not work for coherent LLM

The `pcq` per-channel int8 quantization produces an NBG (~154 MB for
135M/W32, roughly half the size of int16) that runs on the NPU but generates
incoherent text (repeated tokens, garbled words). The root cause is
quantization error in attention and MLP activations at full model depth,
not an unsupported-op issue. The same graph is coherent in int16.

## SmolLM2-1.7B is blocked

All windows (32/64/128/256) pass the ONNX builder gate but fail NBG
generation on the host: `gen_nbg` segfaults during export, producing a
0-byte `network_binary.nb`. The ONNX external-data file is ~6.85 GB. See
[docs/08-known-limits-and-blockers.md](08-known-limits-and-blockers.md).

## Next

- [04-chat-shell.md](04-chat-shell.md) — Interactive chat with streaming tokens
- [docs/configurations.md](configurations.md) — All configs at a glance
- [docs/RESULTS.md](RESULTS.md) — Full measured results
