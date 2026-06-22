# T2 Architecturally Faithful Tiny Block

Date: 2026-06-22

## Purpose

Verified: T2 checks the real small-decoder operator set before scaling model
size. The generated fixed-shape graph uses RMSNorm, RoPE, multi-head causal
attention with GQA, SwiGLU, token embedding `Gather`, final RMSNorm, and logits
`MatMul`.

Verified: CPU work in this task was limited to allowed orchestration and
validation: ONNX generation, ACUITY conversion, SSH upload/run control, and
ACUITY-vs-board output comparison. Verified model-layer compute ran inside the
NBG on the A733 NPU during the board validation.

## Deliverables

Verified:

- Generator: `scripts/host/make_tiny_faithful_block_onnx.py`.
- ONNX package input: `work/generated/tiny_faithful_block/`.
- Int16 NBG package: `work/model-packages/tiny_faithful_block/int16/`.
- Board package path:
  `/home/radxa/a733_npu_driver/models/tiny_faithful_block_t2_int16`.
- Full host conversion log:
  `logs/host/t2-faithful-block-convert-int16.log`.
- Full board run log: `logs/board/t2-faithful-block-vpm-run.log`.
- Output comparison log: `logs/board/t2-faithful-block-compare.log`.

## Model

Verified model constants:

- Batch: `1`.
- Window: `16`.
- Hidden dim: `64`.
- Layers: `2`.
- Attention heads: `4`.
- KV heads: `2`.
- Head dim: `16`.
- GQA repeat: `2`.
- SwiGLU intermediate dim: `192`.
- Vocab: `256`.
- Validation tokens: `1 5 9 2 13 21 34 55 89 144 233 3 8 15 42 7`.

Verified ONNX graph summary:

- Input: `token_ids`, dtype `int32`, shape `1x16`.
- Output: `logits`, dtype `float32`, shape `1x16x256`.
- Node count: `129`.
- Ops: `Add`, `Concat`, `Gather`, `MatMul`, `Mul`, `Neg`,
  `Reciprocal`, `ReduceMean`, `Reshape`, `Sigmoid`, `Slice`, `Softmax`,
  `Sqrt`, `Tile`, `Transpose`.

Verified real-architecture components:

- RMSNorm: `Mul`, `ReduceMean`, `Add`, `Sqrt`, `Reciprocal`, `Mul`, gamma
  `Mul`.
- RoPE: fixed cos/sin initializers, `Slice`, `Neg`, `Concat`, `Mul`, `Add`.
- GQA: K/V projected to `n_kv_heads=2`, then repeated to `n_heads=4` with
  `Reshape -> Tile -> Reshape`.
- Multi-head attention: batched `MatMul`, causal mask add, `Softmax`, value
  `MatMul`, output projection.
- SwiGLU: gate/up projections, `Sigmoid`, `Mul`, down projection.
- Logits: final RMSNorm and `MatMul` to `256` vocabulary logits.

## Conversion

Verified command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name tiny_faithful_block \
  --onnx work/generated/tiny_faithful_block/tiny_faithful_block.onnx \
  --dataset work/generated/tiny_faithful_block/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 16 \
  --outputs logits
```

Verified: ACUITY import, quantization, host inference, and A733 NBG export
completed inside Docker image `ubuntu-npu:v2.0.10.1` for target
`VIP9000NANODI_PLUS_PID0X1000003B`.

Verified: final NBG export ended with `Error(0),Warning(0)`.

Verified: the quantization phase printed `Warning(5)` for pre-existing range
metadata on five RMSNorm square edges. Verified these were warnings, not op
rejects; later inference and final NBG export completed successfully.

Verified package metadata:

```text
network_binary.nb: 409,136 bytes
input: token_ids_166 / token_ids, shape 1x16, dtype int32
output: attach_logits/out0_0 / attach_logits/out0, shape 1x16x256
output quantization: int16 dynamic fixed point, fl=13
```

Verified ACUITY export simulator timing:

```text
Create Neural Network: 39ms or 39291us
Verify Graph: 6857ms or 6857649us
Run the 1 time: 10.53ms or 10530.27us
```

## Board Run

Verified board command:

```bash
cd /home/radxa/a733_npu_driver/models/tiny_faithful_block_t2_int16
LD_LIBRARY_PATH=/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0:$LD_LIBRARY_PATH \
  /home/radxa/ai-sdk/examples/vpm_run/vpm_run \
  -s sample.txt -l 3 -d 0 -b 0 --show_top5 1 --save_txt 1
```

Verified board runtime evidence:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
input 0 dim 16 1 0 0, data_format=8, quant_format=0, name=input/output[0], none-quant
ouput 0 dim 256 16 1 0, data_format=5, name=uid_179_out_0, dfp=13
memory pool size=0byte
profile inference time=186us, cycle=169477
profile inference time=251us, cycle=169553
profile inference time=245us, cycle=170025
vpm run ret=0
```

Verified NPU top-5 from `vpm_run`:

```text
2218: 2.604614
3769: 2.562866
1427: 2.286499
10: 2.278076
3461: 2.196411
```

## Output Comparison

Verified command:

```bash
python scripts/host/compare_outputs.py \
  work/model-packages/tiny_faithful_block/int16/host_output_0.txt \
  logs/board/t2-faithful-block-output_0.txt
```

Verified result:

```text
length: 4096
top-5 index match: no
golden top-5: 2218:2.59948730, 3769:2.55163574, 1427:2.28137207, 10:2.26977539, 2498:2.21118164
board top-5: 2218:2.60461426, 3769:2.56286621, 1427:2.28649902, 10:2.27807617, 3461:2.19641113
max abs diff: 0.073730469
mean abs diff: 0.003069133
RMSE: 0.006297570
cosine: 0.999967503
```

Verified: the T2 success gate requires cosine greater than `0.999`, and the
measured ACUITY-host-vs-NPU cosine was `0.999967503`.

Verified: the global top-5 mismatch is limited to the fifth-ranked logit in a
4096-value tensor. Verified the success gate still passes because the cosine is
above threshold.

## Fallback / Blocker Check

Verified: searching the host and board logs for `fallback`, `unsupported`,
`not support`, and related failure strings found no model-op fallback or
unsupported-op blocker.

Verified: the only `failed` strings in the host log were TensorFlow CUDA
diagnostics inside the CPU-only ACUITY container path, and the log explicitly
states to ignore the CUDA runtime message when no GPU is present.

Verified: ACUITY imported and exported the required T2 op set cleanly:
RMSNorm components, RoPE `Slice/Neg/Concat`, GQA `Tile`, batched multi-head
attention `MatMul/Softmax`, SwiGLU `Sigmoid/Mul`, token `Gather`, and logits
`MatMul`.

## Result

Verified: T2 passed. The RMSNorm + RoPE + SwiGLU + GQA tiny block compiled to
int16 NBG, ran on the A733 VIP9000 NPU, returned `vpm run ret=0`, and matched
the ACUITY golden output with cosine `0.999967503`.

Assumption: the small top-5 rank difference is quantization/runtime numeric
drift rather than a semantic graph issue, because the full-tensor cosine is
well above the T2 threshold and no fallback or unsupported-op messages appeared
in the logs.
