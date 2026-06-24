# T10b Orange Pi SmolLM2 Int16 Port

Date: 2026-06-24

## Scope

Task T10b requested an independent port of the already-working
SmolLM2-135M-Instruct int16 NBG path to the Orange Pi Zero 3W 6GB, regardless
of the Qwen mixed-BF16 result.

## Board

Verified board info from `logs/board/t10b-orangepi-board-info.log`:

- Host: `192.168.31.225`
- User: `orangepi`
- Hostname: `orangepizero3w`
- Kernel: `Linux orangepizero3w 6.6.98-sun60iw2 ... aarch64`
- RAM: `5.7Gi` total, `5.5Gi` available at probe time
- Root disk: `29G`, `6.6G` available
- NPU device: `/dev/vipcore`, `crw-rw-rw-`, major/minor `199,0`

## Host Regeneration

Verified cleanup had removed the previous rebuildable SmolLM2 artifacts, so
the W=32 ONNX and int16 package were regenerated from the retained HF files.

Generated ONNX:

```text
work/generated/smollm2_135m_w32/real_llm.onnx
size: 538,377,902 bytes
input: token_ids, int32 1x32
output: logits, 1x1x49152
```

Generated package:

```text
work/model-packages/smollm2_135m_w32_int16/int16/network_binary.nb
size: 280,882,632 bytes
```

Verified `nbg_meta.json`:

- input `token_ids`: int32 `1x32`
- output `attach_logits/out0`: int16 dynamic fixed point `1x1x49152`, `fl=9`

Verified ACUITY export log:

```text
logs/host/t10b-smollm2-w32-int16-convert.log
Verify Graph: 48689ms
Run the 1 time: 16293.50ms
wrote package: work\model-packages\smollm2_135m_w32_int16\int16
done: work/model-packages/smollm2_135m_w32_int16/int16
```

## Deployment

Verified deployment to the Orange Pi:

```text
/home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb
/home/orangepi/a733_npu_driver/work/models/smollm2-135m-instruct/tokenizer.json
/home/orangepi/a733_npu_driver/scripts/board/
```

The deployed NBG is `268M` on the board (`280,882,632` bytes on host).

## Runner Build

Verified the Orange Pi image does not expose the previous Radxa-style
`/home/radxa/ai-sdk` layout. Runtime files are available as:

```text
/home/orangepi/yolo_shm/vip_lite.h
/home/orangepi/lib/libNBGlinker.so
/home/orangepi/lib/libVIPhal.so
```

Verified `scripts/board/npu_lm_runner.c` and
`scripts/board/build-npu-lm-runner.sh` were updated for this older VIPLite
header variant:

- `A733_VIP_LEGACY_DEVICE_ID` maps device selection to
  `VIP_NETWORK_PROP_SET_DEVICE_ID`.
- `A733_VIP_NO_CORE_INDEX` skips the unavailable core-index property.
- `build-npu-lm-runner.sh` accepts `--vip-inc` and `--vip-lib`, and auto-adds
  these compatibility flags by inspecting `vip_lite.h`.

Verified build-script command on the board:

```text
logs/board/t10b-orangepi-build-runner-script.log
bash scripts/board/build-npu-lm-runner.sh \
  --vip-inc /home/orangepi/yolo_shm \
  --vip-lib /home/orangepi/lib \
  --out /home/orangepi/a733_npu_driver/build/npu_lm_runner
built=/home/orangepi/a733_npu_driver/build/npu_lm_runner
extra_cflags= -DA733_VIP_LEGACY_DEVICE_ID -DA733_VIP_NO_CORE_INDEX
```

Verified binary:

```text
/home/orangepi/a733_npu_driver/build/npu_lm_runner
size: 72K
```

## NPU Validation

Verified verbose NPU run:

```text
logs/board/t10b-orangepi-smollm2-verbose.log
vip_lite_driver_version=0x00020003
VIPLite driver software version 2.0.3.2-AW-2024-08-30
vip_init=OK
cid=0x1000003b
device_count=1
nbg_path=/home/orangepi/a733_npu_driver/models/smollm2_135m_w32_int16/network_binary.nb
nbg_size=280882632
network_core_count=1
nbg_loaded_once=1
final_tokens=...
mean_wall_us=47282.000
mean_profile_us=42777.750
mean_tok_s=21.150
Hello! I'm
```

Verified longer coherence run:

```text
logs/board/t10b-orangepi-smollm2-greeting.log
User: Write a friendly one sentence greeting.
Assistant:
"Hello, welcome to our chat. I'm here to help you explore your interests and passions, and I'm happy
```

Verified metrics for the longer run:

```text
logs/board/t10b-orangepi-smollm2-greeting.err.log
mean_wall_us=46943.333
mean_profile_us=42653.958
mean_tok_s=21.302
```

Note: the local PowerShell wrapper returned exit code `1` for chat runs because
remote stderr contained the context notice and metrics; the remote command
itself printed `[remote-exit] 0` in each captured log.

## NPU-Only Status

Verified: model-layer compute for SmolLM2 W=32 runs in the NBG through VIPLite on
the Orange Pi NPU. CPU work in this validation is limited to tokenization,
fixed-window management, runner launch, argmax/sampling, detokenization, and
logging.

## Result

T10b passed. SmolLM2-135M-Instruct W=32 int16 was regenerated, deployed to the
Orange Pi Zero 3W 6GB, rebuilt against that image's VIPLite runtime, and
validated with coherent NPU-generated text at about `21 tok/s`.
