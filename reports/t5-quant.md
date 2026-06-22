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

## Current Result

Verified: no T5 hybrid NBG package has been produced yet, and no board run has
been attempted for T5. This is a pause to avoid disturbing the parallel Qwen
work, not a quality failure of the hybrid quantization method.

## Next

Verified next action after the Qwen task is done or stopped: rerun the seeded
hybrid conversion:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_hybrid_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --hybrid
```

Assumption: if the seed quantize-table dump remains slow or stuck when run
alone, a faster follow-up is to seed `smollm2_135m_w32_hybrid_pcq_pcq.quantize`
from the already verified
`work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_calib/smollm2_135m_w32_calib_pcq.quantize`
table, then run only the ACUITY hybrid pass.
