# T0 ACUITY Conversion Flow

Date: 2026-06-22

## Purpose

T0 turns the previously manual ACUITY ONNX-to-NBG flow into a reusable host
script for downstream NPU-only LLM/VLM experiments.

## Deliverables

- `scripts/host/convert_onnx_to_nbg.sh` - verified: runs ACUITY inside Docker
  image `ubuntu-npu:v2.0.10.1` and performs import, quantize, host inference,
  and `--pack-nbg-unify` export for target
  `VIP9000NANODI_PLUS_PID0X1000003B`.
- `scripts/host/compare_outputs.py` - verified: compares ACUITY host golden
  tensor text against board `output_0.txt` and prints top-5 match, max/mean
  absolute difference, RMSE, and cosine.
- `work/model-packages/tiny_lm_gather/int16/` - verified generated package
  containing `network_binary.nb`, `nbg_meta.json`, LF-only `sample.txt`,
  `input_0.dat`, `host_output_0.txt`, and `tokens.txt`.

## Conversion Command

Verified command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name tiny_lm_gather \
  --onnx work/generated/tiny_lm_gather/tiny_lm_gather.onnx \
  --dataset work/generated/tiny_lm_gather/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 4 \
  --outputs logits
```

Verified result: ACUITY import, quantize, host inference, and NBG export
completed. Export ended with `Error(0),Warning(0)`.

Verified package artifacts:

| Artifact | Size |
|---|---:|
| `network_binary.nb` | 87,016 bytes |
| `nbg_meta.json` | 738 bytes |
| `sample.txt` | 52 bytes |
| `input_0.dat` | 16 bytes |
| `host_output_0.txt` | 1,249 bytes |

Verified `sample.txt` is LF-only; hex dump shows line endings as `0A` and no
`0D`.

## Input Metadata Note

Verified: ACUITY's generated input metadata can classify `.npy` tensor datasets
as image inputs and set `reverse_channel: true`. For `tiny_lm_gather`, that
would reverse token IDs from `1 5 9 2` to `2 9 5 1`. The conversion script now
normalizes all-`.npy` datasets to tensor-style input metadata:
`category: undefined` and `reverse_channel: false`.

Assumption: image datasets should keep ACUITY's generated image preprocessing
metadata unless a model-specific input metadata file is added later.

## Board Validation

Verified: the regenerated package was uploaded to the Radxa board at
`192.168.31.76` under:

```text
/home/radxa/a733_npu_driver/models/tiny_lm_gather_t0_int16
```

Verified board command used `vpm_run` from:

```text
/home/radxa/ai-sdk/examples/vpm_run/vpm_run
```

Verified board evidence:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
input 0 dim 4 1 0 0, data_format=8, quant_format=0, name=input/output[0], none-quant
ouput 0 dim 16 4 1 0, data_format=5, name=uid_80_out_0, dfp=14
profile inference time=108us
profile inference time=65us
profile inference time=64us
vpm run ret=0
```

Verified top-5 from the regenerated package on NPU:

```text
37: 1.063477
24: 1.034241
45: 0.768982
22: 0.651611
49: 0.589417
```

## Output Comparison

Verified command:

```bash
python scripts/host/compare_outputs.py \
  work/model-packages/tiny_lm_gather/int16/host_output_0.txt \
  logs/board/t0-tiny-lm-gather-output_0.txt
```

Verified output:

```text
length: 64
top-5 index match: yes
golden top-5: 37:1.06365967, 24:1.03387451, 45:0.76910400, 22:0.65173340, 49:0.58996582
board top-5: 37:1.06347656, 24:1.03424072, 45:0.76898193, 22:0.65161133, 49:0.58941650
max abs diff: 0.000610352
mean abs diff: 0.000153542
RMSE: 0.000204006
cosine: 0.999999929
```

## Result

T0 is complete. Verified success gate: one command reconverts
`tiny_lm_gather` to an A733 int16 NBG package, the regenerated package runs on
the A733 NPU, top-5 indices match the ACUITY host golden output, and cosine is
greater than `0.9999`.
