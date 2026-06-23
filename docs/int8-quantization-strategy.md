# INT8 Quantization Strategy

This note condenses the deep analysis in
`research/a733-vip9000-int8-int4-quantization-root-cause.md`
and the master-agent summary in `promts/master_about_trouble.md`.

## Current Decision

The production path is no longer blocked on int8.

First get Qwen2.5-0.5B-Instruct working in int16 on the Orange Pi Zero 3W
with 6 GB LPDDR5. The 1 GB Radxa board was the practical memory blocker for
full Qwen int16 and full Qwen PCQ. The Orange Pi board should remove that
constraint, so int16 can be the first useful Qwen result.

INT8 remains useful, but only as a later optimization after the int16 Orange Pi
path is working.

## What Worked

- SmolLM2-135M int16 is the known coherent baseline.
- The persistent runner path is usable enough to validate real token output on
  the board.
- The Qwen2.5-0.5B int16 host/export path exists from T4/T5.
- The vendor packet now documents the ACUITY hybrid quantize-table failure and
  gives concrete workarounds to try.

## What Did Not Work

- Plain PCQ int8 for real SmolLM2 quality is not acceptable: output becomes
  garbage and does not match the CPU oracle closely enough.
- Mixed PCQ experiments did not recover coherent text.
- ACUITY hybrid quantization can emit a complete `.quantize.json`, but the
  final YAML `.quantize` dump is truncated to zero bytes and ACUITY hangs.
- QDQ-ONNX import is not a viable scale-control path because ACUITY does not
  preserve the required Q/DQ quantization semantics for this flow.
- Waiting for ORT VIPLite EP, etnaviv transformer support, runtime LLM.int8()
  outlier splitting, or dynamic activation quantization is outside the useful
  path for this repo.

## Why INT8 Failed So Far

The working hypothesis is that the quality loss is dominated by activation
quantization inside the transformer body, not by the graph edges. Weight-only
or weight-mostly quantization should be less destructive, especially if
activation outliers are smoothed offline before ONNX export.

The ACUITY hybrid YAML failure appears to be a separate tooling bug. It happens
after import and quantization analysis have already succeeded, while writing the
final YAML table for a graph with hundreds of `dtype_converter` nodes. That is
why the vendor request is important, but it should not block the main model
bring-up.

## Next Tasks

### T6-port

Run Qwen2.5-0.5B-Instruct int16 on the Orange Pi Zero 3W. This is the next
production gate.

Prompt: `promts/t6_new.md`

Success means coherent Qwen text on the Orange Pi NPU with tok/s, RSS, NBG
size, create time, first-token latency, and NPU profile time recorded.

### T7-w8a16

Try W8A16 as an optimization after the int16 path is working:

- int8 per-channel PCQ weights;
- int16 activations;
- offline SmoothQuant-style smoothing before ONNX export;
- ACUITY host simulator plus per-layer cosine gate before board testing;
- widen any collapsing layer back to int16.

Prompt: `promts/t7.md`

Success means coherent output with smaller NBG/RSS than int16. If W8A16 still
fails after per-layer widening, keep int16 as the production path and treat the
logged per-layer cosine as the useful result.

## Vendor Workaround Experiments

The vendor package now asks for guidance on an undocumented ACUITY hybrid
failure and gives two practical experiments:

1. Reconstruct YAML `.quantize` from the completed `.quantize.json`.
2. Reduce the `dtype_converter` count by keeping larger connected subgraphs at
   one precision.

Packet: `reports/t6-vendor-acuity-hybrid-quantize-table.md`
