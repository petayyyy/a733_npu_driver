# T8 Qwen Int16 Port Gate

Date: 2026-06-24

## Purpose

Verified: task T8 started with the required host coherence gate for the
un-smoothed Qwen2.5-0.5B-Instruct W=32 `int16` package. Orange Pi board work
was not started because Gate A failed on host.

## Rebuild

Verified: the previous cleanup had removed rebuildable `work/generated/` and
`work/model-packages/` artifacts, so the un-smoothed package had to be
regenerated from retained HF source files under:

```text
work/models/qwen25-0.5b-instruct/
```

Verified ONNX rebuild:

```bash
docker --config work/docker-config run --rm \
  -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/qwen25-0.5b-instruct \
    --output-dir work/generated/qwen25_05b_w32 \
    --seq-len 32
```

Verified generated ONNX:

```text
work/generated/qwen25_05b_w32/real_llm.onnx
size: 1,976,297,294 bytes
seq_len: 32
layers: 24
hidden_size: 896
vocab_size: 151,936
smoothquant_scales: null
```

Verified: `scripts/host/convert_onnx_to_nbg.sh` now restores the AI SDK
`pegasus_*.sh` helper scripts and `env.sh` into the ignored SDK `models/`
workspace before launching ACUITY. This makes rebuilds work after the documented
cleanup of `work/ai-sdk/ZIFENG278-ai-sdk/models/`.

Verified int16 rebuild:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name qwen25_05b_w32_int16 \
  --onnx work/generated/qwen25_05b_w32/real_llm.onnx \
  --dataset work/generated/qwen25_05b_w32/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

Verified conversion logs:

```text
logs/host/t8-qwen25-05b-w32-int16-convert.log
logs/host/t8-qwen25-05b-w32-int16-convert.err.log
```

Verified ACUITY result:

```text
import: SUCCESS
quantize: SUCCESS
inference: Error(0),Warning(0)
export: Error(0),Warning(0)
Create Neural Network: 14.448s
Verify Graph: 67.124s
Run graph once: 30.921s
```

Verified rebuilt package:

```text
work/model-packages/qwen25_05b_w32_int16/int16/network_binary.nb
size: 1,064,540,800 bytes
input: token_ids int32 1x32
output: logits int16 dynamic fixed point 1x1x151936, fl=10
```

## Host Gate A

Verified FP oracle command:

```bash
docker --config work/docker-config run --rm \
  -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/dump_real_llm_oracle.py \
    --model-dir work/models/qwen25-0.5b-instruct \
    --tokens work/generated/qwen25_05b_w32/token_ids.npy \
    --output work/generated/qwen25_05b_w32_oracle/fp_oracle.npz \
    --seq-len 32
```

Verified oracle top-5:

```text
198: 17.771484375
271: 15.885287285
1406: 13.113974571
715: 12.735839844
14621: 11.821043015
```

Verified ACUITY host-vs-oracle comparison:

```bash
python scripts/host/compare_acuity_host_to_oracle.py \
  --package-dir work/model-packages/qwen25_05b_w32_int16/int16 \
  --oracle work/generated/qwen25_05b_w32_oracle/fp_oracle.npz \
  --model-info work/generated/qwen25_05b_w32/model_info.json \
  --output-json logs/host/t8-qwen25-05b-w32-int16-vs-fp.json
```

Verified Gate A result:

```text
logits cosine: 0.236065208
top1_match: false
host top-1: 67390
oracle top-1: 198
max_abs_diff: 34.059097
mean_abs_diff: 5.836180
rmse: 7.309898
```

## Result

Verified: Gate A failed. The un-smoothed full Qwen2.5-0.5B-Instruct W=32
`int16` ACUITY host output is not coherent against the FP oracle. This is the
major new finding T8 explicitly called out as a stop condition.

Verified: Orange Pi Zero 3W board work was not started. No package was uploaded
to `192.168.31.225`, and no power-cycle or board reset was requested.

## Next

Do not proceed to Step B until the un-smoothed `int16` host mismatch is
understood or a replacement host-quality Qwen package passes the gate. The most
direct next debug is host-only: compare full un-smoothed Qwen per-layer outputs
against FP oracle to find the first layer or final path where cosine collapses.
