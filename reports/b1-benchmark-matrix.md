# B1 Benchmark Matrix

Date: 2026-06-24

Status: in progress.

## Scope

B1 asks for an honest SmolLM2 int16 fixed-window matrix on the Orange Pi Zero
3W at `192.168.31.225`, using host gates before board time.

The fixed comparison prompt for B1 is the SmolLM2 chat rendering of:

```text
The capital of France is
```

The full prompt token sequence is:

```text
1 9690 198 2683 359 253 5356 5646 11173 3365 3511 308 34519 28 7018 411 407 19712 8182 2 198 1 4093 198 504 3575 282 4649 314 2 198 1 520 9531 198
```

For each fixed window, the generator receives the rightmost `min(W, 35)`
tokens and left-pads with zero when `W` is larger than the prompt. This matches
`make_real_llm_onnx.py` and the direct persistent runner behavior.

## Current Matrix

| Model | W | ONNX Runtime vs FP oracle | ACUITY int16 host vs FP oracle | Literal B1 host gate | Board result | NBG size |
| --- | ---: | --- | --- | --- | --- | ---: |
| SmolLM2-135M-Instruct | 32 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.777693043`, top-1 mismatch `1672` vs `504` | fail by `>=0.99` rule | not run in B1 yet; Orange Pi was not idle | 280,882,632 bytes |

## Notes

- Verified Docker resource arguments were used for B1 container work:
  `--cpus 10 --memory 24g`.
- Verified the B1-specific 135M/W32 ONNX graph is correct against the FP oracle:
  `logs/host/b1-smollm2-135m-w32-onnxruntime-vs-fp.json`.
- Verified ACUITY int16 `pegasus inference` host output for the same B1 sample
  fails the literal host-cosine gate:
  `logs/host/b1-smollm2-135m-w32-int16-host-vs-fp.json`.
- Verified a raw-prompt 135M/W32 probe also fails ACUITY host cosine
  (`0.749492804`) even though this model/window is already known to produce
  coherent text on Orange Pi from T10b. This is recorded as a method risk for
  using `pegasus inference` as a hard board-run filter.
- Verified Orange Pi was not idle before board work: another agent was running
  `monitor_command.py ... b4-qwen-cpu-baseline ... llama-bench`. No B1 board
  run was started while that process was present.
