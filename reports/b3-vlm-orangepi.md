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
future board-produced `output_0.txt` embedding can be packed back into the
bridge `input_0.dat` according to `nbg_meta.json`. This is the intended
handoff for an encoder-output-to-bridge run once the Orange Pi is reachable.
The helper was checked against the packaged host bridge input; both files had
SHA256 `41D40B7388E9BAE9C783396E38FE2AE4DBB0C04D70CC47608235D464CEF3F402`.

## Orange Pi Blocker

Blocked before board execution: the Orange Pi at `192.168.31.225` did not
accept SSH connections, so no board idle check, upload, VIPLite run, latency,
RSS, or output comparison could be performed.

Verified from host:

- `scripts/host/ssh_exec.py` failed before authentication with
  `WinError 10013`.
- Windows OpenSSH with `BatchMode=yes` failed with
  `ssh: connect to host 192.168.31.225 port 22: Permission denied`.
- A Docker `/dev/tcp` scan reported all checked ports closed:
  `22`, `2222`, `2200`, `80`, `443`, `8080`, `8888`, and `5900`.
- Clean blocker log:
  `logs/board/b3-orangepi-ssh-blocker.log`.

Because SSH was unavailable, the required user rule "check that nothing else is
running on the Orange Pi before launch" could not be satisfied. No B3 model was
started on the Orange Pi.

## Current Result

Blocked: B3 does not pass yet.

Host-side readiness is complete: both NBG packages were regenerated from
retained sources, exported successfully, and packaged with host reference
outputs. The Orange Pi validation gate remains open until SSH or another file
transfer/execution channel is restored.

## VLM Usability Assessment

Host-only assessment: the Orange Pi VLM story is still a partial
proof-of-concept until the B3 board run is completed. The model-layer pieces
that are expected to run on NPU are:

- MobileCLIP-S0 vision encoder NBG;
- tiny bridge image projector/adapter;
- token embedding `Gather`;
- image/text concat;
- tiny decoder attention, MLP, reductions, and logits.

Verified from earlier Radxa work and host rebuilds: these graphs export for the
A733 target without unsupported-op blockers. Not yet verified on Orange Pi for
B3: actual VIPLite execution, MobileCLIP cosine on board, bridge end-to-end
logits on board, encode latency/RSS, and bridge tok/s/RSS.

Honest limitations remain unchanged:

- image decode/resize/rescale is CPU-side preprocessing;
- text side is fixed-window and has no dynamic KV cache;
- the bridge is tiny and proves the data path, not a useful full VLM;
- a nanoLLaVA/SmolVLM-class model would need a real projector plus a much
  larger fixed-window decoder NBG, and the same int16 outlier/quality risks
  seen on larger LLM work would apply.

## Next

Once SSH is available on `192.168.31.225`, resume with:

1. Verify board idle state and `/dev/vipcore` users before any NPU run.
2. Upload `b3_mobileclip_s0_vision_int16` and `b3_tiny_vlm_bridge_int16`.
3. Run the vision encoder through the Orange Pi VIPLite layout
   `/home/orangepi/lib`.
4. Download `output_0.txt`, compare it to the ACUITY host embedding, and record
   cosine, latency, and RSS.
5. Pack the board-produced embedding into the bridge input with
   `scripts/host/pack_nbg_input_from_text.py`.
6. Run the bridge NBG, compare logits/top-1 against the host reference, and
   record tok/s/RSS.
