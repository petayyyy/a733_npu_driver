# T1 Persistent Tiny LM NPU Runner

Date: 2026-06-22

## Purpose

Replace the previous per-token `vpm_run` relaunch loop with a persistent
VIPLite runner for the existing tiny LM NBG. The runner loads and prepares the
NBG once, then repeatedly overwrites the `1x4` int32 token window, submits the
network to the A733 VIP9000 NPU, reads the last-position logits, and appends the
argmax token.

Verified: model-layer compute remains inside the NBG graph on the NPU:
token embedding `Gather`, position embedding add, causal attention, MLP,
LayerNorm-style reductions, and logits projection. CPU work in this task is
limited to allowed orchestration: file/model loading, token-window updates,
VIPLite calls, logits argmax, and logging.

## Deliverables

Verified on board:

- `scripts/board/npu_lm_runner.c`: persistent C runner using direct VIPLite API.
- `scripts/board/build-npu-lm-runner.sh`: board build helper for VIPLite 2.0.
- `scripts/board/run-npu-lm-runner.sh`: logged board run helper.

Build command used on the Radxa board:

```bash
cd /home/radxa/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh
```

Run command used on the Radxa board:

```bash
cd /home/radxa/a733_npu_driver
bash scripts/board/run-npu-lm-runner.sh --label int16-tiny-lm
```

## VIPLite Lifecycle

Verified from the SDK `examples/vpm_run/vpm_run.c`, the persistent runner keeps
the same core lifecycle but moves destroy/re-init outside the token loop:

1. `vip_init`
2. `vip_create_network`
3. `vip_create_buffer` for input and output
4. `vip_prepare_network`
5. `vip_set_input` and `vip_set_output`
6. For each token: map/write input, flush input, `vip_run_network`, invalidate
   output, map/read output, query `VIP_NETWORK_PROP_PROFILING`
7. `vip_finish_network`
8. `vip_destroy_buffer`
9. `vip_destroy_network`
10. `vip_destroy`

Verified: `create_network_us=457` and `prepare_network_us=218` appeared once in
the persistent run log, before `nbg_loaded_once=1`.

## Board Validation

Verified hardware/runtime:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
cid=0x1000003b
device_count=1
```

Verified model contract:

```text
nbg_size=87016
input_dims=4x1 input_format=8 input_quant=0 input_elements=4 input_bytes=16
output_dims=16x4x1 output_format=5 output_quant=1 output_dfp=14 output_elements=64 output_bytes=128
```

Note: ACUITY metadata describes the logical output as `1x4x16`; VIPLite reports
the same contiguous tensor as `16x4x1`. The runner matches the prior
`vpm_run` loop by selecting the last 16 linear output values.

Verified generated sequence:

```text
1 5 9 2 1 8 4 5 8 4 8 4
```

This matches the prior per-token `vpm_run` decode-loop sequence.

Step summary from the persistent runner:

| Step | Input window | Next token | Last-position top-5 | NPU profile | Wall |
|---:|---|---:|---|---:|---:|
| 0 | `1 5 9 2` | 1 | `1:0.589417,10:0.502991,13:0.490234,2:0.459656,15:0.412598` | 67us | 247us |
| 1 | `5 9 2 1` | 8 | `8:0.745300,0:0.632324,14:0.532166,9:0.378967,3:0.363220` | 60us | 137us |
| 2 | `9 2 1 8` | 4 | `4:0.479858,6:0.318115,12:0.172485,3:0.171936,0:0.131775` | 62us | 135us |
| 3 | `2 1 8 4` | 5 | `5:0.745789,8:0.629333,12:0.457458,0:0.407593,11:0.352905` | 61us | 133us |
| 4 | `1 8 4 5` | 8 | `8:1.102051,6:0.683228,0:0.504639,14:0.391541,11:0.332153` | 59us | 129us |
| 5 | `8 4 5 8` | 4 | `4:0.470093,12:0.218872,5:0.195251,6:0.190979,0:0.179443` | 61us | 131us |
| 6 | `4 5 8 4` | 8 | `8:0.692505,5:0.594666,0:0.559021,12:0.415100,11:0.385376` | 60us | 130us |
| 7 | `5 8 4 8` | 4 | `4:0.464539,12:0.219910,5:0.199463,6:0.185059,0:0.167419` | 61us | 129us |

Persistent runner summary:

| Metric | Value |
|---|---:|
| Mean NPU profile time | 61.375us |
| Mean per-token runner wall time | 146.375us |
| Mean runner throughput | 6831.768 tok/s |

## Reload Baseline

Verified by rerunning the old `scripts/board/run-tiny-lm-decode-loop.sh` path on
the same board/model/prompt. It reproduced the same token sequence:

```text
1 5 9 2 1 8 4 5 8 4 8 4
```

Old per-token `vpm_run` component timings from the baseline run:

| Metric | Min | Max | Mean |
|---|---:|---:|---:|
| `create network 0` | 430us | 519us | 463.125us |
| `prepare network 0` | 214us | 230us | 221.875us |
| `read input and golden 0` | 83us | 99us | 87.875us |
| `run time for this network 0` | 200us | 304us | 247.000us |
| `profile inference time` | 67us | 111us | 82.375us |

Verified external wall time around the old shell loop:

```text
old_loop_elapsed_ns=9895542883
```

That is `1,236,942.860us/token` over 8 generated tokens.

Assumption: the external old-loop wall number includes shell, Python, file I/O,
`vpm_run` process launch, logging, and NBG reload overhead. For isolating the
SDK-visible reload cost, the more comparable old component sum is create +
prepare + read + run = `1,019.875us/token`.

## Timing Delta

Verified:

- Persistent in-process wall: `146.375us/token`.
- Old `vpm_run` component sum: `1,019.875us/token`.
- Old full shell loop wall: `1,236,942.860us/token`.

Computed from verified numbers:

- Persistent runner is about `6.97x` faster than the old per-token
  create+prepare+read+run component sum.
- Persistent runner is about `8450x` faster than the previous full shell
  loop. This larger ratio includes process launch, Python/file/logging, and
  other script overhead in addition to NBG reload.

## Result

T1 success gate passed. Verified on the A733 board: the persistent runner loads
and prepares the tiny LM NBG once, reproduces the prior generated token
sequence, reports stable lower per-token wall time, and keeps model-layer
compute on the NPU.

## Raw Logs

Raw logs are stored locally under ignored logs:

```text
logs/board/t1-persistent-runner-run.log
logs/board/t1-persistent-runner-summary.env
logs/board/t1-reload-baseline-run.log
logs/board/t1-reload-baseline-steps.tsv
logs/board/t1-reload-baseline-tokens.txt
```
