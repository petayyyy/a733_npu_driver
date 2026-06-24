# B3 VLM Orange Pi Validation

Date: 2026-06-24

## Scope

Task B3 asked to close the remaining Orange Pi Zero 3W gap for the previously
proven Radxa VLM path: MobileCLIP-S0 vision encoder on the NPU, followed by the
tiny VLM bridge NBG with image projector, token `Gather`, concat, decoder, and
logits on the NPU.

## Host Preparation

Verified host-only: Docker was run with `DOCKER_RUN_ARGS="--cpus 10 --memory
24g"` for the B3 ACUITY conversions.

Verified host-only: `scripts/host/make_mobileclip_input.py` prepared a real
image tensor from
`work/ai-sdk/ZIFENG278-ai-sdk/examples/resnet50/input_data/dog_224_224.jpg`
using the retained MobileCLIP-S0 preprocessor config:

- output tensor: `work/generated/b3_mobileclip_s0_vision/pixel_values.npy`
- shape: `1x3x256x256`
- rescale: `1/255`
- normalization: none

Verified host-only: regenerated the MobileCLIP-S0 vision encoder package:

- package: `work/model-packages/b3_mobileclip_s0_vision_int16/int16/`
- NBG size: `19,376,840` bytes
- input: `pixel_values`, int16 dynamic fixed point, `1x3x256x256`, `fl=15`
- output: `attach_image_embeds/out0`, int16 dynamic fixed point, `1x512`,
  `fl=17`
- ACUITY export: `Error(0),Warning(0)`
- ACUITY host top-5 embedding indices:
  `34:0.18962097`, `198:0.13613892`, `137:0.12991333`,
  `7:0.11938477`, `433:0.11223602`

Verified host-only: regenerated the tiny VLM bridge package using the B3
MobileCLIP host embedding as the `image_embed` input and token window
`1 5 9 2`:

- package: `work/model-packages/b3_tiny_vlm_bridge_int16/int16/`
- NBG size: `94,400` bytes
- input 0: `image_embed`, int16 dynamic fixed point, `1x512`, `fl=17`
- input 1: `token_ids`, int32, `1x4`
- output: `attach_logits/out0`, int16 dynamic fixed point, `1x5x16`, `fl=14`
- ACUITY export: `Error(0),Warning(0)`
- ACUITY host full-tensor top-5:
  `49:0.91656494`, `76:0.82794189`, `62:0.71771240`,
  `41:0.63885498`, `58:0.62103271`
- ACUITY host last-token top-5:
  `12:0.82794189`, `0:0.60864258`, `8:0.56988525`,
  `15:0.39642334`, `3:0.26391602`

Verified host-only: fixed a reusable multi-input packaging issue in
`scripts/host/convert_onnx_to_nbg.sh`. ACUITY generates `dataset0.txt` and
`dataset1.txt` references for separated multi-input databases; the wrapper now
copies sibling `dataset[0-9]*.txt` files into the SDK model workspace before
quantization.

Verified host-only: added `scripts/host/pack_nbg_input_from_text.py` so a
board-produced `output_0.txt` embedding can be packed back into the bridge
`input_0.dat` according to `nbg_meta.json`. The helper was checked against the
packaged host bridge input; both files had SHA256
`41D40B7388E9BAE9C783396E38FE2AE4DBB0C04D70CC47608235D464CEF3F402`.

## Orange Pi Validation

Verified on board: SSH later became available at `orangepi@192.168.31.225`.
Before every NPU run, checked that no unrelated `npu_lm_runner`, `vpm_run`,
`chat_shell.py`, `llama`, `monitor_command.py`, `cmake`, or `ninja` process was
active, and checked `/dev/vipcore` with `fuser`. No conflicting process or
`/dev/vipcore` user was present.

Verified on board: both packages were uploaded under:

- `/home/orangepi/a733_npu_driver/models/b3_mobileclip_s0_vision_int16/`
- `/home/orangepi/a733_npu_driver/models/b3_tiny_vlm_bridge_int16/`

Verified on board: added `scripts/board/run-b3-vpm-package.sh` and used
`/opt/vpm_run/vpm_run` with `LD_LIBRARY_PATH=/home/orangepi/lib`. The runner
records `vpm_run` output, saves `output_0.txt`, and samples process RSS from
`/proc/<pid>/status`.

## Vision Encoder Result

Verified on board: MobileCLIP-S0 vision encoder ran on the Orange Pi NPU.

- log: `logs/board/b3-mobileclip-orangepi-run.log`
- output: `logs/board/b3-mobileclip-orangepi-output_0.txt`
- loops: `5`
- VIPLite: `2.0.3.2-AW-2024-08-30`
- NPU identity: `cid=0x1000003b`
- `vpm run ret=0`
- create network: `12,698us`
- prepare network: `2,319us`
- working memory pool: `1,573,888` bytes
- profile latency: mean `22,605.8us`, min `22,567us`, max `22,660us`
- `vpm_run` wall run time: mean `22,767.4us`, min `22,717us`, max `22,825us`
- peak RSS sampler result: `14,080 KB`; peak VmHWM: `21,248 KB`

Verified comparison against ACUITY host int16 reference:

- length: `512`
- top-5 index match: yes
- host top-5:
  `34:0.18962097`, `198:0.13613892`, `137:0.12991333`,
  `7:0.11938477`, `433:0.11223602`
- Orange Pi top-5:
  `34:0.18962097`, `198:0.13615417`, `137:0.12958527`,
  `7:0.11984253`, `433:0.11214447`
- max abs diff: `0.001373291`
- mean abs diff: `0.000309840`
- RMSE: `0.000397144`
- cosine: `0.999955977`

## Bridge Result

Verified on board: the Orange Pi MobileCLIP output was packed into the VLM
bridge `image_embed` input and the bridge NBG ran end-to-end with tokens
`1 5 9 2`.

- packed board-derived bridge input SHA256:
  `BF552D11B70E40877CDF9242F5DBEA83B1CC6D1D6F052A1EDFF49C63AE31A1BC`
- log: `logs/board/b3-tiny-vlm-bridge-orangepi-run.log`
- output: `logs/board/b3-tiny-vlm-bridge-orangepi-output_0.txt`
- loops: `20`
- VIPLite: `2.0.3.2-AW-2024-08-30`
- NPU identity: `cid=0x1000003b`
- `vpm run ret=0`
- create network: `532us`
- prepare network: `246us`
- working memory pool: `0` bytes
- profile latency: mean `62.6us`, min `59us`, max `75us`
- `vpm_run` wall run time: mean `140.4us`, min `121us`, max `283us`
- profile-rate equivalent for one next-token bridge forward:
  `15,974.441 tok/s`
- peak RSS sampler result: `1,792 KB`; peak VmHWM: `1,792 KB`

Verified full-output comparison against the ACUITY host int16 reference:

- length: `80`
- top-5 index match: yes
- host top-5:
  `49:0.91656494`, `76:0.82794189`, `62:0.71771240`,
  `41:0.63885498`, `58:0.62103271`
- Orange Pi top-5:
  `49:0.91632080`, `76:0.82867432`, `62:0.71801758`,
  `41:0.63824463`, `58:0.62139893`
- max abs diff: `0.009826660`
- mean abs diff: `0.001402283`
- RMSE: `0.002544185`
- cosine: `0.999982426`

Verified last-token logits comparison:

- length: `16`
- top-5 index match: yes
- host last-token top-5:
  `12:0.82794189`, `0:0.60864258`, `8:0.56988525`,
  `15:0.39642334`, `3:0.26391602`
- Orange Pi last-token top-5:
  `12:0.82867432`, `0:0.60858154`, `8:0.56958008`,
  `15:0.39593506`, `3:0.26232910`
- coherent next-token result for this tiny fixed vocab: top-1 token `12`
- max abs diff: `0.001586914`
- mean abs diff: `0.000514984`
- RMSE: `0.000649709`
- cosine: `0.999999036`

Verified after validation: no B3 `vpm_run`/runner process remained and
`/dev/vipcore` had no users.

## Current Result

B3 passes. The MobileCLIP-S0 vision encoder and the tiny VLM bridge both run on
the Orange Pi Zero 3W NPU through the board's VIPLite runtime, and the bridge
uses the board-produced image embedding rather than only the host embedding.

## VLM Usability Assessment

Verified assessment: the Orange Pi VLM story is a working proof-of-concept, not
yet a useful production VLM. The model-layer pieces that ran on NPU in B3 are:

- MobileCLIP-S0 vision encoder NBG;
- tiny bridge image projector/adapter;
- token embedding `Gather`;
- image/text concat;
- tiny decoder attention, MLP, reductions, and logits.

Verified on Orange Pi: actual VIPLite execution, MobileCLIP cosine on board,
bridge end-to-end logits on board, encoder latency/RSS, and bridge
latency/tok-rate/RSS.

Honest limitations remain unchanged:

- image decode/resize/rescale is CPU-side preprocessing;
- text side is fixed-window and has no dynamic KV cache;
- the bridge is tiny and proves the data path, not a useful full VLM by itself;
- a nanoLLaVA/SmolVLM-class model would need a real projector plus a much
  larger fixed-window decoder NBG, and the same int16 outlier/quality risks
  seen on larger LLM work would apply.

Net: VLM-on-NPU is a working Orange Pi story at proof-of-concept scale. A full
small VLM would still require replacing the tiny bridge/decoder with a real
small-VLM projector and decoder, accepting fixed-window constraints unless a
KV-cache-capable runtime path is built, and re-validating int16 quality and NBG
size/weight-bandwidth limits.
