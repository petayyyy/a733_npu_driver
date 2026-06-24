# T10 Qwen Mixed BF16 Host Gate

Date: 2026-06-24

## Scope

Task T10 requested a host-first mixed-precision Qwen2.5-0.5B-Instruct W=32
attempt before any Orange Pi upload:

- embedding and lm_head/logits projection on int16;
- outlier-heavy transformer regions on BF16;
- Gate A: NBG export succeeds, logits cosine vs FP oracle is greater than
  `0.99`, and top-1 matches.

If no mixed split satisfied the gate, the required stop condition was to file a
precise vendor blocker instead of uploading Qwen to the board.

## Seed Tooling

Verified: added `scripts/host/make_qwen2_mixed_bf16_quantize.py` to create
mixed ACUITY quantize tables from the existing Qwen int16 and BF16 tables.

The tool supports:

- BF16-base tables with selected int16 qparams/layers for embeddings and logits;
- int16-base tables with auto-selected BF16 outlier qparams;
- JSON sidecar summaries with the selected qparams/layers.

`python -m py_compile scripts/host/make_qwen2_mixed_bf16_quantize.py` passed.

All Docker conversion/export runs used `DOCKER_RUN_ARGS='--cpus 10 --memory
24g'`.

## Attempt 1: BF16 Base With Int16 Slice/Logits Path

Model name: `qwen25_05b_w32_mixed_bf16`

Verified seed summary:

- Output table:
  `work/generated/qwen25_05b_w32_mixed_bf16/qwen25_05b_w32_mixed_bf16_bf16.quantize`
- `critical_qparams_replaced=3`
- `critical_qparams_added=6`
- `critical_customized_layers=1`

Verified failure:

```text
logs/host/t10-qwen25-w32-mixed-bf16-convert.err.log
E [ops/vsi_nn_op_strided_slice.c:op_check:507]Inputs/Outputs data type not support:  BFLOAT16, DFP INT16
E [vsi_nn_graph.c:setup_node:551]Check node[1664] SLICE fail
E 23:11:11 Fatal model generation error: 65280
missing export directory: work\ai-sdk\ZIFENG278-ai-sdk\models\qwen25_05b_w32_mixed_bf16\wksp\qwen25_05b_w32_mixed_bf16_bf16_nbg_unify
```

Result: invalid mixed boundary. ACUITY/VIPLite does not accept this
BF16-to-DFP-int16 `SLICE` output configuration.

## Attempt 2: BF16 Transformer, Int16 Embedding And Final Projection

Model name: `qwen25_05b_w32_mixed_bf16_v2`

This removed the final `Slice`/`Reshape` int16 overrides and kept int16 only on
the token embedding and final logits projection path.

Verified seed summary:

- Output table:
  `work/generated/qwen25_05b_w32_mixed_bf16_v2/qwen25_05b_w32_mixed_bf16_v2_bf16.quantize`
- qparam prefixes:
  `@attach_logits/out0_0:`, `@fullconnect_1973:`, `@hidden0_1554:`,
  `@token_embed_1593:`
- layer prefix: `fullconnect_1973`
- `critical_qparams_replaced=3`
- `critical_qparams_added=4`
- `critical_customized_layers=1`

Verified ACUITY host inference produced the expected top-1 token `198`:

```text
top5:
198: 17.46484375
271: 15.4775390625
715: 12.7998046875
1406: 12.6064453125
2303: 11.77734375
```

Verified export failure:

```text
logs/host/t10-qwen25-w32-mixed-bf16-v2-convert.err.log
E [main.c:vnn_VerifyGraph:93]CHECK STATUS(-3:The requested set of parameters produce a configuration that cannot be supported. )
E 23:26:12 Fatal model generation error: 64768
missing export directory: work\ai-sdk\ZIFENG278-ai-sdk\models\qwen25_05b_w32_mixed_bf16_v2\wksp\qwen25_05b_w32_mixed_bf16_v2_bf16_nbg_unify
```

Result: the quality-relevant host candidate still fails NBG generation at
`vnn_VerifyGraph`; no `network_binary.nb` exists for Gate A.

## Attempt 3: Int16 Base With Auto BF16 Outlier Regions

Model name: `qwen25_05b_w32_mixed_bf16_outliers`

Command profile: int16 base plus `--auto-qwen-outliers --outlier-min-abs 1000`.

Verified seed summary:

- Output table:
  `work/generated/qwen25_05b_w32_mixed_bf16_outliers/qwen25_05b_w32_mixed_bf16_outliers_bf16.quantize`
- Sidecar:
  `work/generated/qwen25_05b_w32_mixed_bf16_outliers/qwen25_05b_w32_mixed_bf16_outliers_bf16.json`
- `critical_qparams_replaced=153`
- `removed_customized_layers=110`
- selected qparam prefixes: `146`
- largest selected outliers included RMS-squared tensors around
  `2,652,062.75`, for example `layer17_mlp_rms_squared_533`,
  `layer18_mlp_rms_squared_455`, and `layer18_attn_rms_squared_627`.

Verified ACUITY host inference again kept top-1 token `198`:

```text
top5:
198: 17.6044921875
271: 15.3466796875
715: 13.1640625
1406: 12.380859375
2303: 12.1396484375
```

Verified export failure:

```text
logs/host/t10-qwen25-w32-mixed-bf16-outliers-convert.err.log
E [ops/vsi_nn_op_matrixmul.c:op_check:130]Inputs/Outputs data type not support: DFP INT16, DFP INT16,  BFLOAT16
E [vsi_nn_graph.c:setup_node:551]Check node[41] MATRIXMUL fail
E 23:39:20 Fatal model generation error: 65280
missing export directory: work\ai-sdk\ZIFENG278-ai-sdk\models\qwen25_05b_w32_mixed_bf16_outliers\wksp\qwen25_05b_w32_mixed_bf16_outliers_bf16_nbg_unify
```

Result: this outlier split creates an unsupported mixed `MATRIXMUL` boundary:
DFP-int16 inputs with a BF16 output.

## Vendor Blocker

Verified blocker: Qwen2.5-0.5B-Instruct W=32 has no mixed BF16/int16 split
tested in T10 that satisfies Gate A. The tested quality-relevant split with
BF16 transformer compute and int16 embedding/final projection reaches host
inference top-1 match but fails NBG generation with `vnn_VerifyGraph` status
`-3`. The more selective outlier split fails earlier on node-level
`MATRIXMUL` dtype support.

Reproduction context:

- Model: Qwen2.5-0.5B-Instruct fixed-window W=32 decoder.
- Input: `token_ids`, int32 `1x32`.
- Output: logits `1x1x151936`.
- Toolchain: ACUITY `6.30.22`, Vivante IDE `5.11.0`.
- Target optimize string: `VIP9000NANODI_PLUS_PID0X1000003B`.
- Failing logs:
  - `logs/host/t10-qwen25-w32-mixed-bf16-v2-convert.err.log`
  - `logs/host/t10-qwen25-w32-mixed-bf16-outliers-convert.err.log`

Required vendor answer: which BF16/int16 boundaries are legal for
`SLICE`/`MATRIXMUL` and why the BF16-heavy Qwen W=32 graph fails
`vnn_VerifyGraph` without a node-level diagnostic.

## Result

T10 Qwen Gate A failed because no tested mixed BF16/int16 candidate produced an
exported NBG. Qwen was not uploaded to `192.168.31.225`, and no Orange Pi reset
or power-cycle was requested.
