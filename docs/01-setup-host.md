# 01 — Host Setup

Setting up the x86 host with the ACUITY Docker toolchain for ONNX-to-NBG
conversion, host oracle generation, and board communication.

## Prerequisites

- x86_64 host running Windows (PowerShell) or Linux (bash)
- Docker Engine installed and running
- Python 3 (for host helpers and output comparison)
- At least 30 GB free disk space (ONNX files for real models can reach ~2 GB,
  and NBG packages can reach ~1 GB)

## Step 1: Workspace directories

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1
```

This creates the local working directories (`work/`, `logs/`, `models/`) if
they don't exist. It checks whether Docker and the expected ACUITY image
(`ubuntu-npu:v2.0.10.1`) are available. It does not download anything by default.

## Step 2: ACUITY Docker image

Pull or load the ACUITY Docker image:

```bash
docker pull ubuntu-npu:v2.0.10.1
```

If you have a pre-exported tarball:

```bash
docker load < ubuntu-npu-v2.0.10.1.tar
```

Verify:

```bash
docker run --rm ubuntu-npu:v2.0.10.1 pegasus.py --help
```

The image contains:
- ACUITY toolkit 6.30.22 (`/root/acuity-toolkit-whl-6.30.22/bin/`)
- `pegasus.py` import, quantize, inference, and export
- `pegasus_export_ovx_nbg.sh` for A733 NBG package generation
- `pegasus_quantize_hybird.sh` for hybrid quantization (partially tested; see t6)

## Step 3: SSH access to the board

For board automation from the host, the host scripts use Paramiko for
SSH/SFTP. The helper is:

```
scripts/host/ssh_exec.py
```

Usage:

```bash
python3 scripts/host/ssh_exec.py \
  --host <board-ip> \
  --user <ssh-user> \
  --password <password> \
  --cmd "uname -a"
```

Alternatively, use `scp` directly to upload NBG packages:

```bash
scp -r work/model-packages/my_model/int16/ user@board:~/
```

## Step 4: Converting ONNX to NBG

The main conversion wrapper is `scripts/host/convert_onnx_to_nbg.sh`. It runs
inside the ACUITY Docker container and performs:
1. ONNX import
2. Quantization (uint8, int16, bf16, fp16, or pcq)
3. Host inference (generates ACUITY golden output)
4. NBG export for target `VIP9000NANODI_PLUS_PID0X1000003B`

Basic usage:

```bash
# Convert with int16 quantization (recommended for SmolLM2)
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32 \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

With Docker resource limits:

```bash
DOCKER_RUN_ARGS="--cpus 10 --memory 24g" \
  scripts/host/convert_onnx_to_nbg.sh \
  --name my_large_model \
  --onnx path/to/model.onnx \
  --dataset path/to/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

The output package lands under `work/model-packages/<name>/<quant>/` and
contains:
- `network_binary.nb` — The NBG file for the board
- `nbg_meta.json` — Metadata (input/output shapes, dtypes, dfp positions)
- `input_0.dat` — Host golden input (for verification)
- `output_0.txt` — ACUITY host golden output (for comparison)

### Quantization options

| Flag | Effect | Status for LLM |
|---|---|---|
| `--quant int16` | Dynamic fixed point int16 | Working for SmolLM2, fails quality on Qwen |
| `--quant pcq` | Per-channel int8 (asymmetric affine) | Exports but incoherent for LLM |
| `--quant bf16` | BFloat16 | Host quality OK, export blocked for Qwen |
| `--quant fp16` | Float16 | Exports but fails quality gate |
| `--quant uint8` | Uniform uint8 | OK for CNNs only |

### Advanced: seed quantize (skip quantization)

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name my_model \
  --onnx path/to/model.onnx \
  --seed-quantize path/to/existing.quantize \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

## Step 5: Host oracle gate

Before uploading an NBG to the board, compare ACUITY host output against the
CPU FP reference:

```bash
# Compare ACUITY golden output vs FP oracle
python3 scripts/host/compare_outputs.py \
  --golden work/model-packages/my_model/int16/output_0.txt \
  --board path/to/oracle_output.txt \
  --output-json results.json
```

```bash
# Compare ONNX Runtime output vs FP oracle (builder verification)
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/compare_onnxruntime_to_oracle.py \
    --onnx work/generated/my_model/real_llm.onnx \
    --tokens work/generated/my_model/token_ids.npy \
    --oracle work/generated/my_model_oracle/fp_oracle.npz \
    --model-info work/generated/my_model/model_info.json \
    --output-json logs/host/my-model-onnxruntime-vs-fp.json
```

The comparison tool reports:
- top-5 index match (yes/no)
- max abs diff, mean abs diff, RMSE
- cosine similarity

A pass threshold of cosine > 0.99 with top-1 match is used for the host gate.

## Step 6: Generating ONNX graphs

The real LLM ONNX generator is `scripts/host/make_real_llm_onnx.py`. It reads
a Hugging Face model directory (`config.json` + `model.safetensors`) and builds
a fixed-window decoder graph with last-token logits.

```bash
# Generate SmolLM2-135M W=32 ONNX
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/smollm2-135m-instruct \
    --output-dir work/generated/smollm2_135m_w32 \
    --seq-len 32
```

For Qwen2.5-0.5B (note: blocked for NBG export; see docs/08):

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  DOCKER_RUN_ARGS="--cpus 10 --memory 24g" \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/qwen25-0.5b-instruct \
    --output-dir work/generated/qwen25_05b_w32 \
    --seq-len 32
```

### Other ONNX generators

| Script | Purpose |
|---|---|
| `make_tiny_decoder_block_onnx.py` | Tiny transformer decoder block probe |
| `make_tiny_faithful_block_onnx.py` | Architecturally faithful tiny LM (RMSNorm, RoPE, GQA) |
| `make_tiny_lm_onnx.py` | Tiny LM with Gather embedding |
| `make_tiny_vlm_bridge_onnx.py` | VLM bridge (image embed + tokens → logits) |
| `make_real_llm_onnx.py` | Real full-depth LLM (SmolLM2, Qwen) |

## Step 7: Board upload

Upload the NBG package to the board:

```bash
scp -r work/model-packages/smollm2_135m_w32_int16/int16/* \
  user@board:~/a733_npu_driver/models/smollm2_135m_w32_int16/
```

Or use the SSH helper:

```bash
python3 scripts/host/ssh_exec.py \
  --host <board-ip> --user <user> --password <pw> \
  --upload work/model-packages/smollm2_135m_w32_int16/int16/ \
  --remote-path ~/a733_npu_driver/models/smollm2_135m_w32_int16/
```

## Next

- Continue to [02-board-bringup.md](02-board-bringup.md) to prepare the board.
- Or directly to [03-run-llm-npu.md](03-run-llm-npu.md) if the board is already ready.
