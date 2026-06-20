# Phase 3a Probe Report - Tiny CLIP Vision Encoder

Date: 2026-06-20 local / 2026-06-20 UTC

## Purpose

This is a Phase 3a compatibility probe, not a semantic VLM demo. The goal was
to answer a narrower question first: can the public ACUITY/VIPLite A733 path
import, quantize, export, and run a small transformer-style vision encoder with
MatMul, Softmax, LayerNorm, Gather, and MLP blocks?

## Source Model

- Model: `hf-internal-testing/tiny-random-CLIPModel`
- File: `onnx/vision_model.onnx`
- URL: https://huggingface.co/hf-internal-testing/tiny-random-CLIPModel
- ONNX opset: 14
- Input: dynamic ONNX `pixel_values`, fixed for ACUITY as `1x3x30x30`
- Output: `image_embeds`, shape `1x64`

The model is randomly initialized and tiny. It is useful for op coverage and
runtime validation, but not for real image-text quality.

## ONNX Operator Coverage

The downloaded ONNX vision model contains the transformer-relevant operators
needed for this probe:

| Operator | Count |
|---|---:|
| MatMul | 41 |
| Softmax | 5 |
| ReduceMean | 24 |
| Gather | 2 |
| Conv | 1 |
| Reshape | 41 |
| Transpose | 26 |
| Add | 65 |
| Mul | 15 |
| Div | 12 |
| Sqrt | 12 |
| Sigmoid | 5 |

ACUITY imported the graph and lowered the LayerNorm pattern to
`layernormalize` layers.

## ACUITY Setup

The model was staged in the ignored SDK workspace as `models/clip_tiny_vision`
with:

```text
clip_tiny_vision.onnx
inputs_outputs.txt
clip_tiny_vision_inputmeta.yml
dataset.txt
space_shuttle_224x224.jpg
```

`inputs_outputs.txt` fixed the dynamic ONNX input:

```bash
--inputs pixel_values --input-size-list '3,30,30' --outputs 'image_embeds'
```

After import, ACUITY renamed the input lid to `pixel_values_287`, so
`clip_tiny_vision_inputmeta.yml` uses that lid. Preprocess follows the CLIP
image processor: RGB, size `30x30`, mean in pixel scale, and scale
`1 / (std * 255)`.

## ACUITY Commands

```bash
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3
../scripts/pegasus_import.sh clip_tiny_vision
../scripts/pegasus_quantize.sh clip_tiny_vision int16
../scripts/pegasus_inference.sh clip_tiny_vision int16
../scripts/pegasus_export_ovx.sh clip_tiny_vision int16
```

Result:

- Import: success.
- Quantize int16: success.
- CPU inference: success.
- NBG export: `Error(0),Warning(0)`.

## Generated Artifact

| Quantization | NBG path | Size |
|---|---:|---:|
| int16 | `models/clip_tiny_vision/wksp/clip_tiny_vision_int16_nbg_unify/network_binary.nb` | 720,824 bytes |

Input/output metadata:

| Tensor | Shape | Quantization |
|---|---|---|
| `pixel_values_287` | `1x3x30x30` | int16 dynamic fixed point, `dfp=13` |
| `attach_image_embeds/out0_0` | `1x64` | int16 dynamic fixed point, `dfp=13` |

## Board Validation

The NBG was uploaded to the Radxa Cubie A7Z and run through the SDK
`examples/vpm_run` binary built on the board.

Command:

```bash
cd /home/radxa/a733_npu_driver/models/clip_tiny_vision_int16
/home/radxa/ai-sdk/examples/vpm_run/vpm_run -s sample.txt -l 3 -d 0
```

Evidence:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
device[0] core_count=1
input 0 dim 30 30 3 1, data_format=5, quant_format=1, dfp=13
ouput 0 dim 64 1 0 0, data_format=5, dfp=13
profile inference time=2173us
profile inference time=2170us
profile inference time=2169us
vpm run ret=0
```

Top-5 output dimensions from `-b 0 --show_top5 1 --save_txt 1`:

```text
34: 2.159668
24: 2.027832
40: 1.979370
7: 1.169556
13: 1.120361
```

## Output Comparison

The board-side `output_0.txt` was compared with ACUITY's host-side int16
inference tensor:

| Comparison | Top-5 set matches | Max abs diff | Mean abs diff |
|---|---:|---:|---:|
| ACUITY int16 vs NPU int16 | yes | 0.016845703 | 0.003650665 |

The top-5 embedding dimensions are preserved. The remaining numeric difference
is acceptable for this probe and should be revisited with a real encoder and
task-level metric.

## Raw Logs

The raw board archive is stored locally under ignored logs:

```text
logs/board/g3a-clip-tiny-vision.tar.gz
logs/board/g3a-clip-tiny-vision/
logs/board/g3a-clip-tiny-vision-int16-output_0.txt
```

## Result

The static vision-encoder part of Phase 3a is technically viable on this tiny
CLIP-like graph: ACUITY can compile transformer-style vision blocks to A733 NBG,
and VIPLite can run the generated graph on the real VIP9000 NPU.

G3a is not complete yet. The remaining work is to repeat this with a real small
vision encoder and wire its embedding output to a CPU-side llama.cpp decoder for
an end-to-end image-to-text path.
