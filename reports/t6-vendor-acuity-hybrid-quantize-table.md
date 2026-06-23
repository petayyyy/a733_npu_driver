# T6 Vendor Blocker: ACUITY Hybrid Quantize Table Dump

Date: 2026-06-23

## Summary

ACUITY hybrid quantization for the full SmolLM2-135M W=32 graph consistently
reaches `End quantization...`, starts dumping the rewritten YAML quantize table,
truncates the target `.quantize` file to `0` bytes, and then remains CPU-active
without producing an inference/exportable package.

This blocks the T5 recovery paths that require `w8a16` or hybrid quantization.
The same graph imports, quantizes, exports, and runs coherently in int16.

## Environment

```text
Docker image: ubuntu-npu:v2.0.10.1
ACUITY path: /root/acuity-toolkit-whl-6.30.22/bin
Target: VIP9000NANODI_PLUS_PID0X1000003B
Model: SmolLM2-135M-Instruct fixed W=32 decoder graph
ONNX: work/generated/smollm2_135m_w32/real_llm.onnx
Dataset: work/generated/smollm2_135m_w32_calib/dataset.txt
Input: token_ids, int32, 1x32
Output: logits, 1x1x49152
```

## Repro 1: Calibrated PCQ Seed + Hybrid

Command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_hybrid_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --hybrid \
  --hybrid-seed-quantize work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_calib/smollm2_135m_w32_calib_pcq.quantize
```

Observed:

```text
ACUITY loads existing quantization tensor table.
ACUITY emits smollm2_135m_w32_hybrid_pcq_pcq.quantize.json.
The JSON graph contains 589 dtype_converter ops.
ACUITY logs End quantization...
ACUITY logs Dump net quantize tensor table to .../smollm2_135m_w32_hybrid_pcq_pcq.quantize.
The YAML .quantize file becomes 0 bytes and stays 0 bytes while ACUITY remains CPU-active.
No inference/export package is produced.
```

Logs:

```text
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.log
logs/host/t5-smollm2-w32-hybrid-seeded-from-calib-convert.err.log
```

Artifacts:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_hybrid_pcq/smollm2_135m_w32_hybrid_pcq_pcq.quantize
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_hybrid_pcq/smollm2_135m_w32_hybrid_pcq_pcq.quantize.json
```

## Repro 2: Mixed PCQ Seed + Hybrid

Command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_mixed_hybrid_pcq \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits \
  --hybrid \
  --hybrid-seed-quantize work/generated/smollm2_135m_w32_mixed_pcq/smollm2_135m_w32_mixed_pcq_pcq.quantize
```

Observed:

```text
ACUITY loads existing mixed quantization tensor table.
ACUITY emits smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.json.
The JSON graph contains 587 dtype_converter ops.
ACUITY logs End quantization...
ACUITY logs Dump net quantize tensor table to .../smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.
The YAML .quantize file becomes 0 bytes and remains 0 bytes.
After an extra 90 seconds, Docker was still CPU-active at about 99.6 percent and 2.6 GiB RSS.
No inference/export package is produced.
```

Logs:

```text
logs/host/t5-smollm2-w32-mixed-hybrid-pcq-convert.log
logs/host/t5-smollm2-w32-mixed-hybrid-pcq-convert.err.log
```

Artifacts:

```text
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_mixed_hybrid_pcq/smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize
work/ai-sdk/ZIFENG278-ai-sdk/models/smollm2_135m_w32_mixed_hybrid_pcq/smollm2_135m_w32_mixed_hybrid_pcq_pcq.quantize.json
```

## Expected Behavior

ACUITY should either:

- write a valid non-empty YAML `.quantize` table and proceed to inference/export, or
- fail with a non-zero exit code and a diagnostic explaining the unsupported hybrid graph/table state.

## Actual Behavior

ACUITY reports quantization completion, truncates the target YAML `.quantize`
file to zero bytes, and stays CPU-active indefinitely. The process does not
return a diagnostic and does not produce an NBG package.

## Documentation Gap

This behavior appears undocumented in the available ACUITY/SDK materials in
this workspace. The bundled scripts show that `pegasus_quantize_hybird.sh`
expects a quantize table, and the logs show that ACUITY can emit an intermediate
`.quantize.json`, but there is no documented contract for:

- whether hybrid quantization must always start from an existing YAML
  `.quantize` table;
- how the YAML table should represent the `dtype_converter` nodes introduced by
  the hybrid rewrite;
- whether converter-heavy mixed int8/int16 graphs have a supported limit;
- why ACUITY truncates the YAML output before finishing instead of returning a
  clear error.

This makes the issue difficult to distinguish from a user-side table-format
mistake without vendor guidance.

## Hypothesis

The failure is likely in ACUITY's hybrid quantize-table serialization/rewrite
path, after graph analysis has already succeeded. Evidence:

- ONNX import succeeds.
- ACUITY consumes the seed quantize table.
- ACUITY emits a rewritten `.quantize.json`.
- The rewritten JSON contains hundreds of `dtype_converter` nodes
  (`589` in repro 1, `587` in repro 2).
- ACUITY logs `End quantization...`.
- The hang starts at `Dump net quantize tensor table to ... .quantize`, after
  the YAML target has already been truncated to `0` bytes.

Working hypothesis: the hybrid pass creates a converter-heavy mixed-qtype graph
that the internal JSON representation can hold, but the YAML `.quantize`
serializer cannot finish writing. The trigger may be the large number of
`dtype_converter` edges, repeated range metadata, or unsupported qtype
transitions such as int16-to-int8 and int16-to-float32 around transformer
residual/logits boundaries.

## Workarounds To Try

These are not confirmed fixes, but they are concrete next experiments for an
agent before escalating again.

### 1. Rebuild YAML From The Completed JSON

ACUITY produces a non-empty `.quantize.json` before the YAML dump hangs. Try
constructing the final YAML `.quantize` table from that JSON instead of relying
on ACUITY's hanging YAML writer.

Suggested experiment:

```text
input:  smollm2_135m_w32_hybrid*_pcq.quantize.json
output: smollm2_135m_w32_hybrid*_pcq.quantize
```

Then rerun inference/export using the reconstructed YAML as a seed table via
the existing `--seed-quantize` path, skipping the hybrid quantize dump step if
possible.

Expected value: this tests whether the JSON contains enough valid quantization
state for export, and isolates the blocker to YAML serialization rather than
the quantization analysis itself.

### 2. Reduce The Number Of `dtype_converter` Nodes

The hybrid rewrite creates hundreds of `dtype_converter` nodes. Try reducing
converter pressure before the YAML dump.

Possible approaches:

- keep larger connected subgraphs in the same qtype so fewer int8/int16
  boundaries need converters;
- make only the most sensitive paths int16, such as embeddings, final RMSNorm,
  final hidden slice, lm_head, and logits;
- avoid alternating qtypes inside every transformer layer;
- generate a smaller mixed seed table from the known-good int16 and pcq tables,
  then rerun hybrid;
- compare converter counts in the emitted JSON and stop once the count drops
  substantially below the current `587-589` range.

Expected value: if the YAML dump succeeds after converter count is reduced, the
vendor request can point to a concrete converter-table scale/serialization
limit instead of a generic hybrid failure.

## Control Results

The same SmolLM2 W=32 graph is valid on this toolchain in other modes:

```text
int16 export: succeeds, NBG 280,882,632 bytes
int16 board run: coherent, first six generated tokens match FP oracle
pcq export: succeeds, NBG about 153,984,304 bytes
pcq board run: executes mechanically but generates incoherent tokens
mixed PCQ seed export: succeeds, NBG 205,233,968 bytes, output int16 fl=10
mixed PCQ board run: executes mechanically but generates incoherent tokens
```

This points to a hybrid quantize-table serialization or rewrite bug rather than
a basic ONNX import, VIPLite export, or board-runtime issue.
