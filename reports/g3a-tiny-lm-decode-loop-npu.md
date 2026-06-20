# G3a Tiny LM Decode Loop NPU Validation

Date: 2026-06-20

## Purpose

Validate a fixed-window autoregressive language-model loop where every
model-layer forward pass runs on the A733 VIP9000 NPU, while CPU-side work is
limited to loop orchestration, token-window updates, and logits postprocessing.

This is not a CPU decoder path. The NPU graph still performs token embedding
`Gather`, position embedding add, causal attention, MLP, LayerNorm-style
reductions, and logits projection. The CPU only writes the next `token_ids`
input window and chooses the next tiny token from NPU-produced logits.

## Model And Runtime

The loop uses the previously validated tiny LM NBG:

- NBG: `models/tiny_lm_gather/wksp/tiny_lm_gather_int16_nbg_unify/network_binary.nb`
- NBG size: `87,016` bytes
- Input: `token_ids`, shape `1x4`, dtype `int32`
- Output: logits, shape `1x4x16`, int16 dynamic fixed point `dfp=14`
- Initial prompt tokens: `1 5 9 2`
- Generation steps: `8`

Board runner:

```bash
scripts/board/run-tiny-lm-decode-loop.sh \
  --model-dir /home/radxa/a733_npu_driver/models/tiny_lm_gather_int16 \
  --prompt "1 5 9 2" \
  --steps 8 \
  --seq-len 4 \
  --vocab 16
```

At each step the script:

1. Writes the current fixed `1x4` int32 token window to `input_0.dat`.
2. Runs the tiny LM NBG through SDK `examples/vpm_run`.
3. Reads `output_0.txt`.
4. Selects `argmax` from the last-position logits only.
5. Appends that token and slides the next `1x4` window.

## Board Validation

The loop was run on the Radxa Cubie A7Z through the SDK `vpm_run` binary built
on the board.

Evidence:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b, device_count=1
vpm run ret=0
```

The VIPLite banner, A733 optimize ID, and `vpm run ret=0` were observed for
all 8 decode steps.

Generated tiny-token sequence:

```text
1 5 9 2 1 8 4 5 8 4 8 4
```

Step summary:

| Step | Input window | Next token | Last-position top-5 | NPU profile |
|---:|---|---:|---|---:|
| 0 | `1 5 9 2` | 1 | `1:0.589417,10:0.502991,13:0.490234,2:0.459656,15:0.412598` | 99us |
| 1 | `5 9 2 1` | 8 | `8:0.745300,0:0.632324,14:0.532166,9:0.378967,3:0.363220` | 68us |
| 2 | `9 2 1 8` | 4 | `4:0.479858,6:0.318115,12:0.172485,3:0.171936,0:0.131775` | 68us |
| 3 | `2 1 8 4` | 5 | `5:0.745789,8:0.629333,12:0.457458,0:0.407593,11:0.352905` | 69us |
| 4 | `1 8 4 5` | 8 | `8:1.102051,6:0.683228,0:0.504639,14:0.391541,11:0.332153` | 98us |
| 5 | `8 4 5 8` | 4 | `4:0.470093,12:0.218872,5:0.195251,6:0.190979,0:0.179443` | 138us |
| 6 | `4 5 8 4` | 8 | `8:0.692505,5:0.594666,0:0.559021,12:0.415100,11:0.385376` | 137us |
| 7 | `5 8 4 8` | 4 | `4:0.464539,12:0.219910,5:0.199463,6:0.185059,0:0.167419` | 70us |

Profile summary:

| Steps | Min | Max | Mean |
|---:|---:|---:|---:|
| 8 | 68us | 138us | 93.375us |

## Result

Gate G3a fixed-window decode-loop subgate passed: an autoregressive loop can
drive repeated tiny LM forward passes where model-layer compute stays on the
A733 NPU and CPU-side work is limited to allowed orchestration and
postprocessing.

Current limitation: this runner intentionally uses `vpm_run` once per generated
token, so it proves the compute placement and data contract, not optimized
token throughput. The next implementation step is a persistent VIPLite/awnn
runner that loads the NBG once and submits repeated token windows without
process and graph-load overhead.

## Raw Logs

Raw board logs are stored locally under ignored logs:

```text
logs/board/g3a-tiny-lm-decode-loop.tar.gz
logs/board/g3a-tiny-lm-decode-loop/
```
