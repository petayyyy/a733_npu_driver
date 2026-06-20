# Phase 3a Report - MobileCLIP-S0 Vision Encoder

Date: 2026-06-20 local / 2026-06-20 UTC

## Purpose

This report extends the Phase 3a tiny CLIP probe to a real small vision encoder.
The goal was to validate that ACUITY can import, quantize, export, and run a
semantically useful static CLIP-style vision encoder on the A733 VIP9000 NPU.

This is still not an end-to-end VLM demo. The remaining Phase 3a work is to
connect the NPU-produced image embedding to an NPU-side projector and language
decoder path.

## Source Model

- Model: `Xenova/mobileclip_s0`
- File: `onnx/vision_model.onnx`, staged locally as `mobileclip_s0_vision.onnx`
- URL: https://huggingface.co/Xenova/mobileclip_s0
- ONNX IR/opset: IR 7, opset 12
- Input: dynamic ONNX `pixel_values`, fixed for ACUITY as `1x3x256x256`
- Output: `image_embeds`, shape `1x512`

The Hugging Face image processor uses RGB conversion, resize/crop to `256x256`,
rescale by `1/255`, and no mean/std normalization.

## ONNX Operator Coverage

The ONNX vision graph contains both CNN-style and transformer-style operators:

| Operator | Count |
|---|---:|
| Conv | 95 |
| MatMul | 9 |
| Softmax | 2 |
| AveragePool | 1 |
| BatchNormalization | 2 |
| ReduceMean | 3 |
| Gather | 9 |
| Reshape | 9 |
| Transpose | 10 |
| Add | 54 |
| Mul | 89 |
| Div | 30 |
| Erf | 30 |
| Sigmoid | 3 |
| Concat | 9 |

This makes it a more useful Phase 3a target than the random tiny CLIP probe:
the encoder is real, static, and compact, while still exercising transformer
subgraphs that matter for the NPU-only VLM path.

## ACUITY Setup

The model was staged in the ignored SDK workspace as
`models/mobileclip_s0_vision` with:

```text
mobileclip_s0_vision.onnx
inputs_outputs.txt
mobileclip_s0_vision_inputmeta.yml
dataset.txt
space_shuttle_224x224.jpg
```

`inputs_outputs.txt` fixed the dynamic ONNX input:

```bash
--inputs pixel_values --input-size-list '3,256,256' --outputs 'image_embeds'
```

After import, ACUITY renamed the input lid to `pixel_values_257`. The input
metadata therefore uses that lid, RGB order, and per-channel scale
`0.00392156862745098`.

## ACUITY Commands

```bash
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3
../scripts/pegasus_import.sh mobileclip_s0_vision
../scripts/pegasus_quantize.sh mobileclip_s0_vision int16
../scripts/pegasus_inference.sh mobileclip_s0_vision int16
../scripts/pegasus_export_ovx.sh mobileclip_s0_vision int16
```

Result:

- Import: success.
- Quantize int16: success.
- CPU inference: success.
- NBG export: `Error(0),Warning(0)`.
- ACUITY simulator during export: average `179.99ms`.

## Generated Artifact

| Quantization | NBG path | Size |
|---|---:|---:|
| int16 | `models/mobileclip_s0_vision/wksp/mobileclip_s0_vision_int16_nbg_unify/network_binary.nb` | 19,376,840 bytes |

Input/output metadata:

| Tensor | Shape | Quantization |
|---|---|---|
| `pixel_values_257` | `1x3x256x256` | int16 dynamic fixed point, `dfp=15` |
| `attach_image_embeds/out0_0` | `1x512` | int16 dynamic fixed point, `dfp=17` |

## Board Validation

The NBG was uploaded to the Radxa Cubie A7Z and run through the SDK
`examples/vpm_run` binary built on the board.

Command:

```bash
cd /home/radxa/a733_npu_driver/models/mobileclip_s0_vision_int16
/home/radxa/ai-sdk/examples/vpm_run/vpm_run -s sample.txt -l 3 -d 0
```

Evidence:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
device[0] core_count=1
input 0 dim 256 256 3 1, data_format=5, quant_format=1, dfp=15
ouput 0 dim 512 1 0 0, data_format=5, dfp=17
profile inference time=22537us
profile inference time=22642us
profile inference time=22583us
vpm run ret=0
```

Top-5 output dimensions from `-b 0 --show_top5 1 --save_txt 1`:

```text
198: 0.165749
 61: 0.138969
438: 0.101570
329: 0.099518
109: 0.095856
```

## Output Comparison

The board-side `output_0.txt` was compared with ACUITY's host-side int16
inference tensor:

| Comparison | Length | Top-5 indices match | Max abs diff | Mean abs diff | RMSE | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| ACUITY int16 vs NPU int16 | 512 | yes | 0.002471924 | 0.000398278 | 0.000532371 | 0.999884700 |

Top-5 comparison:

```text
host: 198:0.16550446, 61:0.13938904, 438:0.10161591, 329:0.09975433, 109:0.09584045
npu:  198:0.16574860, 61:0.13896942, 438:0.10157013, 329:0.09951782, 109:0.09585571
```

The embedding ordering is preserved and the numeric drift is small for an int16
NPU path.

## Raw Logs

The raw board archive is stored locally under ignored logs:

```text
logs/board/g3a-mobileclip-s0-vision.tar.gz
logs/board/g3a-mobileclip-s0-vision/
logs/board/g3a-mobileclip-s0-vision-int16-output_0.txt
```

## Result

The static vision-encoder subgate of Phase 3a now passes with a real
MobileCLIP-S0 encoder: ACUITY exported the graph to an int16 A733 NBG, VIPLite
executed it on the real VIP9000 NPU, and the `1x512` embedding matches the
host-side int16 inference within a tight tolerance.

G3a is not complete yet. The remaining work is NPU-side projector/decoder
bring-up and integration of this embedding output into an end-to-end
image-to-text path.
