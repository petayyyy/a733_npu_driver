# T9 Qwen BF16/FP16 Host Gate

Date: 2026-06-24

## Purpose

Verified: task T9 investigated why the un-smoothed Qwen2.5-0.5B-Instruct
W=32 `int16` package from T8 failed the host gate, then tried floating-16
ACUITY export paths before any Orange Pi work.

## Step 1 - Builder Versus Quantization

Verified: the existing Qwen W=32 ONNX graph was run through ONNX Runtime and
compared with the FP oracle:

```bash
docker --config work/docker-config run --rm --cpus 10 --memory 24g \
  -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/compare_onnxruntime_to_oracle.py \
    --onnx work/generated/qwen25_05b_w32/real_llm.onnx \
    --tokens work/generated/qwen25_05b_w32/token_ids.npy \
    --oracle work/generated/qwen25_05b_w32_oracle/fp_oracle.npz \
    --model-info work/generated/qwen25_05b_w32/model_info.json \
    --output-json logs/host/t9-qwen25-w32-onnxruntime-vs-fp.json \
    --threads 10
```

Verified result:

```text
logits cosine: 0.9999999999967962
top1_match: true
ONNX Runtime top-1: 198
FP oracle top-1: 198
max_abs_diff: 0.000029087
mean_abs_diff: 0.000004833
```

Conclusion: verified the ONNX builder is correct for this token window. The
T8 `int16` failure is in ACUITY dynamic-fixed-point quantization, not in Qwen
config handling or graph construction.

## Step 2a - BF16

Verified code change: `scripts/host/convert_onnx_to_nbg.sh` now accepts
`--quant bf16`. The underlying ACUITY SDK wrapper already maps it to
`--quantizer qbfloat16 --qtype qbfloat16`.

Verified BF16 conversion command used Docker with the requested host resources:

```bash
DOCKER_RUN_ARGS="--cpus 10 --memory 24g" \
scripts/host/convert_onnx_to_nbg.sh \
  --name qwen25_05b_w32_bf16 \
  --onnx work/generated/qwen25_05b_w32/real_llm.onnx \
  --dataset work/generated/qwen25_05b_w32/dataset.txt \
  --quant bf16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

Verified BF16 was accepted by ACUITY:

```text
quantize command: --quantizer qbfloat16 --qtype qbfloat16
quantize table: work/ai-sdk/ZIFENG278-ai-sdk/models/qwen25_05b_w32_bf16/qwen25_05b_w32_bf16_bf16.quantize
quantize table size: 127,214 bytes
ACUITY host inference: completed
```

Verified BF16 host-quality gate passed:

```text
logits cosine: 0.9906279877646436
top1_match: true
host top-1: 198
oracle top-1: 198
max_abs_diff: 1.746928
mean_abs_diff: 0.252472
```

Verified BF16 NBG export failed:

```text
Create Neural Network: 13031ms
Verify...
E [main.c:vnn_VerifyGraph:93] CHECK STATUS(-3:The requested set of parameters produce a configuration that cannot be supported. )
E [main.c:main:236] CHECK STATUS(-3:The requested set of parameters produce a configuration that cannot be supported. )
Fatal model generation error: 64768
Error(1),Warning(0)
missing export directory: .../qwen25_05b_w32_bf16_bf16_nbg_unify
```

Verified logs:

```text
logs/host/t9-qwen25-05b-w32-bf16-convert.log
logs/host/t9-qwen25-05b-w32-bf16-convert.err.log
logs/host/t9-qwen25-05b-w32-bf16-host-vs-fp.json
```

BF16 blocker for vendor: full Qwen2.5-0.5B-Instruct W=32 graph, input
`token_ids` int32 `1x32`, output logits `1x1x151936`, quantizer
`qbfloat16`, qtype `qbfloat16`. ACUITY host inference passes the FP oracle
gate, but NBG generation fails during `vnn_VerifyGraph` with status `-3` and
`Fatal model generation error: 64768`. ACUITY did not emit a node name for the
failure; the full stdout/stderr logs above are preserved.

## Step 2b - FP16

Verified code change: `scripts/host/convert_onnx_to_nbg.sh` now accepts
`--quant fp16` through direct `pegasus.py` calls:

```text
quantize: --quantizer float16 --qtype float16
inference: --dtype quantized --model-quantize <name>_fp16.quantize
export: --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B
```

Verified ACUITY CLI exposes `float16` in `pegasus quantize --help`.

Verified first FP16 wrapper attempt failed before quantization because the
direct command was run from the SDK `models/` directory rather than from the
model subdirectory. This was a wrapper bug, not an ACUITY qtype result. Logs:

```text
logs/host/t9-qwen25-05b-w32-fp16-convert.log
logs/host/t9-qwen25-05b-w32-fp16-convert.err.log
```

Verified retry after fixing `pushd/popd` completed quantize, host inference,
and NBG export:

```text
quantize table: qwen25_05b_w32_fp16_fp16.quantize, 123,914 bytes
network_binary.nb: 991,416,168 bytes
output metadata: dtype float16, shape 1x1x151936
export: Error(0),Warning(0)
```

Verified package:

```text
work/model-packages/qwen25_05b_w32_fp16/fp16/network_binary.nb
work/model-packages/qwen25_05b_w32_fp16/fp16/nbg_meta.json
```

Verified FP16 host-quality gate failed:

```text
logits cosine: 0.5408570190232891
top1_match: true
host top-1: 198
oracle top-1: 198
max_abs_diff: 13.209313
mean_abs_diff: 1.841242
```

Top-1 survived for this one sample, but the cosine is far below the required
`>0.99` gate, so the FP16 NBG is not a valid board candidate.

Verified logs:

```text
logs/host/t9-qwen25-05b-w32-fp16-convert.retry1.log
logs/host/t9-qwen25-05b-w32-fp16-convert.retry1.err.log
logs/host/t9-qwen25-05b-w32-fp16-vs-fp.json
```

## Result

Verified: T9 did not proceed to Orange Pi Zero 3W at `192.168.31.225`.

Reason:

- BF16 fixes host quality, but full Qwen BF16 NBG export is blocked by ACUITY
  `vnn_VerifyGraph` status `-3`.
- FP16 exports to NBG, but fails the host quality gate with logits cosine
  `0.540857019`.

No package was uploaded to the Orange Pi, and no board reset or power-cycle was
requested.

## Next

Verified next actionable blocker: send the BF16 export failure packet to the
vendor or test a vendor-recommended BF16 export setting. If continuing without
vendor input, the next host-only research path is a mixed-precision seed that
keeps only the Qwen outlier-sensitive regions in BF16 while avoiding the full
BF16 NBG generation failure. That is a new experiment, not a passed T9 gate.
