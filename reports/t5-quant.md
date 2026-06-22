# T5 SmolLM2 Int8 Quality Fix

Date: 2026-06-22

## Purpose

Verified: task T5 is to recover an int8 or hybrid SmolLM2-135M path after the
plain `pcq` package was proven to execute mechanically but fail coherence. The
active success gate remains NPU-only model-layer execution: token embedding,
attention, MLP, norms, and logits must stay inside the NBG graph.

## Attempt 1: ACUITY Hybrid PCQ

Verified: ACUITY exposes a `pegasus quantize --hybrid` option in
`ubuntu-npu:v2.0.10.1`. The SDK also ships
`pegasus_quantize_hybird.sh`, which invokes:

```text
pegasus.py quantize ... --compute-entropy --hybrid \
  --model-quantize <name>_pcq.quantize \
  --quantizer perchannel_symmetric_affine --qtype int8
```

Verified code change: `scripts/host/convert_onnx_to_nbg.sh` now accepts
`--hybrid` and routes hybrid conversion through the SDK hybrid quantize script.

Verified unseeded result: running hybrid directly does not create a quantize
table from scratch. ACUITY fails before inference/export with:

```text
quantize file 'smollm2_135m_w32_hybrid_pcq_pcq.quantize' does not exist
```

Verified logs:

```text
logs/host/t5-smollm2-w32-hybrid-pcq-convert.log
logs/host/t5-smollm2-w32-hybrid-pcq-convert.err.log
```

Verified follow-up code change: in `--hybrid` mode, the converter now runs the
normal `pegasus_quantize.sh` pass first to seed `<name>_pcq.quantize`, then runs
`pegasus_quantize_hybird.sh` as a second pass.

Verified seeded run status: the seeded run imported the W=32 SmolLM2 graph and
entered the normal `pcq --rebuild` seed pass. It reached:

```text
End quantization...
Dump net quantize tensor table to .../smollm2_135m_w32_hybrid_pcq_pcq.quantize
```

Verified observed artifact at stop time:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_hybrid_pcq/smollm2_135m_w32_hybrid_pcq_pcq.quantize
size: 0 bytes
mtime: 2026-06-22T20:03:56+03:00
```

Verified: the T5 Docker container was still CPU-active, but a parallel Qwen
conversion container from another chat was also CPU-active. To avoid interfering
with that Qwen task, only the T5 container (`nostalgic_yonath`) was stopped.
The Qwen container (`tender_buck`) was left running.

Verified seeded logs:

```text
logs/host/t5-smollm2-w32-hybrid-seeded-pcq-convert.log
logs/host/t5-smollm2-w32-hybrid-seeded-pcq-convert.err.log
```

Verified rerun alone: after the parallel Qwen container was gone, the same
seeded hybrid command was rerun. It again reached `End quantization...` and
`Dump net quantize tensor table`, but `<name>_pcq.quantize` stayed at `0`
bytes while the ACUITY quantize process remained CPU-active.

Verified rerun logs:

```text
logs/host/t5-smollm2-w32-hybrid-rerun-pcq-convert.log
logs/host/t5-smollm2-w32-hybrid-rerun-pcq-convert.err.log
```

Verified fallback code change: `scripts/host/convert_onnx_to_nbg.sh` now
supports `--hybrid-seed-quantize PATH`, which copies an existing quantize table
into the ACUITY model directory and runs only `pegasus_quantize_hybird.sh`.

Verified fallback seed: the existing calibrated PCQ table was copied from:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_calib/smollm2_135m_w32_calib_pcq.quantize
```

Verified fallback result: the hybrid pass consumed the seed and produced
`smollm2_135m_w32_hybrid_pcq_pcq.quantize.json` with 589 `dtype_converter`
ops, including int16-to-int8 and int16-to-float32 converters. However, ACUITY
then tried to dump the YAML quantize table, truncated
`smollm2_135m_w32_hybrid_pcq_pcq.quantize` to `0` bytes, and remained
CPU-active. No inference/export package was produced.

Verified fallback logs:

```text
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.log
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.err.log
```

Conclusion for attempt 1: ACUITY hybrid/w8a16 is currently blocked in quantize
table emission, not in graph import. The blocker is reproducible with both a
fresh seed pass and an existing calibrated seed table.

## Attempt 2: Mixed PCQ Seed

Verified code change: `scripts/host/convert_onnx_to_nbg.sh` now supports
`--seed-quantize PATH` for non-hybrid runs. This copies an existing quantize
table to `<name>_<quant>.quantize`, skips `pegasus_quantize.sh`, then runs
ACUITY inference and export only.

Verified code change: added `scripts/host/make_smollm2_mixed_quantize.py`.
It creates a mixed quantize table by using the verified calibrated PCQ table as
the base and replacing critical tensor/weight entries from the verified int16
table:

- token embedding: `token_embed_1920`, `hidden0_1883`
- final RMSNorm and last-token slice: `final_rms*`, `final_last_token_2`
- lm_head/logits path: `reshape_2278`, `fullconnect_2279`, `reshape_2280`,
  `attach_logits/out0_0`

Verified generated mixed seed:

```text
work/generated/smollm2_135m_w32_mixed_pcq/smollm2_135m_w32_mixed_pcq_pcq.quantize
size: 9,209,753 bytes
copied int16 quantize_parameters: 17
copied int16 customized_quantize_layers: 9
```

Verified spot check: the generated seed sets logits/head/embedding entries to
`qtype: i16` and `quantizer: dynamic_fixed_point`, while nearby transformer MLP
entries such as `reshape_2281` and `fullconnect_2282` remain
`qtype: i8` / `asymmetric_affine`.

## Current Result

Verified: no new T5 NBG package has been produced yet, and no new board run has
been attempted for T5 after the hybrid blocker. A Qwen conversion container
from the parallel chat is currently active, so no heavy T5 Docker conversion was
started after preparing the mixed seed.

## Next

Verified next action after the Qwen task is done or stopped: run attempt 2
using the mixed seed without rebuilding the quantize table:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_mixed_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --seed-quantize work/generated/smollm2_135m_w32_mixed_pcq/smollm2_135m_w32_mixed_pcq_pcq.quantize
```

If export succeeds, upload the package to the board and compare the first six
generated tokens for `The capital of France is` against the existing FP/int16
oracle sequence `504 3575 282 4649 314 7042`.
