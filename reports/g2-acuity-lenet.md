# Gate G2 Report - ACUITY LeNet Conversion

Date: 2026-06-20 local / 2026-06-19 UTC

## Source

Radxa ACUITY setup reference:

- https://docs.radxa.com/en/cubie/a7z/app-dev/npu-dev/cubie-acuity-env

The page confirms the A733 ACUITY workflow uses:

- Docker image name: `ubuntu-npu:v2.0.10.1`
- SDK clone: `https://github.com/ZIFENG278/ai-sdk.git`
- A733 build setting: `AI_SDK_PLATFORM=a733`
- NPU target setting: `NPU_VERSION=v3`

## Host Toolchain

- Docker Desktop is running.
- Public mirror image pulled as `khalida5/ubuntu-npu:v2.0.10`.
- Local tag added as `ubuntu-npu:v2.0.10.1`.
- ACUITY toolkit path in container:
  `/root/acuity-toolkit-whl-6.30.22/bin`
- Vivante IDE path in container:
  `/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools`

## Model

- Model: LeNet from `ZIFENG278/ai-sdk/models/lenet`
- Original framework: Caffe
- Input: `1x1x28x28`
- Target optimization: `VIP9000NANODI_PLUS_PID0X1000003B`

## ACUITY Commands

```bash
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3
pegasus_one lenet
../scripts/pegasus_quantize.sh lenet int16
../scripts/pegasus_inference.sh lenet int16
../scripts/pegasus_export_ovx.sh lenet int16
```

The Windows checkout of `ai-sdk` had CRLF line endings; the ignored working
copy was normalized to LF before running the shell scripts in Docker.

## Generated Artifacts

| Quantization | NBG path | Size |
|---|---:|---:|
| uint8 | `models/lenet/wksp/lenet_uint8_nbg_unify/network_binary.nb` | 443,144 bytes |
| int16 | `models/lenet/wksp/lenet_int16_nbg_unify/network_binary.nb` | 845,256 bytes |

## Board Validation

Both NBGs were uploaded to the Radxa Cubie A7Z and run through the SDK
`examples/vpm_run` binary built on the board.

### uint8

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
input 0 dim 28 28 1 1, data_format=2, quant_format=2
profile inference time=81us, cycle=65290
vpm run ret=0
```

### int16

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
input 0 dim 28 28 1 1, data_format=5, quant_format=1, dfp=7
profile inference time=176us, cycle=122948
vpm run ret=0
```

## Gate Result

G2 is passed for a known SDK CNN: ACUITY converts LeNet to A733-compatible NBG
for both uint8 and int16, and the generated NBG files run successfully on the
A733 VIP9000 through `/dev/vipcore` and VIPLite.

## Next Step

Move from SDK sample LeNet to a custom ONNX CNN or vision encoder. The same
container, `NPU_VERSION=v3`, and board-side `vpm_run` path are now validated.
