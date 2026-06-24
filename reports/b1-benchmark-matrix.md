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
| SmolLM2-135M-Instruct | 32 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.777693043`, top-1 mismatch `1672` vs `504` | fail by `>=0.99` rule | not run by literal host-gate rule | 280,882,632 bytes |
| SmolLM2-135M-Instruct | 64 | verified cosine `1.000000000`, top-1 `2` match | verified cosine `-0.761936835`, top-1 mismatch `347` vs `2` | fail by `>=0.99` rule | not run by literal host-gate rule | 282,310,408 bytes |
| SmolLM2-135M-Instruct | 128 | verified cosine `1.000000000`, top-1 `198` match | verified cosine `-0.590862935`, top-1 mismatch `407` vs `198` | fail by `>=0.99` rule | not run by literal host-gate rule | 286,894,144 bytes |
| SmolLM2-135M-Instruct | 256 | verified cosine `1.000000000`, top-1 `198` match | verified cosine `-0.784897858`, top-1 mismatch `46161` vs `198` | fail by `>=0.99` rule | not run by literal host-gate rule | 337,090,160 bytes |
| SmolLM2-360M-Instruct | 32 | verified cosine `1.000000000`, top-1 `57` match | verified cosine `0.131091912`, top-1 mismatch `99` vs `57` | fail by `>=0.99` rule | not run by literal host-gate rule | 672,715,688 bytes |
| SmolLM2-360M-Instruct | 64 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.607897325`, top-1 mismatch `6643` vs `504` | fail by `>=0.99` rule | not run by literal host-gate rule | 674,536,104 bytes |
| SmolLM2-360M-Instruct | 128 | verified cosine `1.000000000`, top-1 `198` match | verified cosine `0.476642772`, top-1 `198` match | fail by `>=0.99` rule | not run by literal host-gate rule | 693,294,624 bytes |
| SmolLM2-360M-Instruct | 256 | verified cosine `0.999999995`, top-1 `198` match | verified cosine `0.287806907`, top-1 mismatch `36616` vs `198` | fail by `>=0.99` rule | not run by literal host-gate rule | 708,452,104 bytes |
| SmolLM2-1.7B-Instruct | 32 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.268220295`, top-1 mismatch `31532` vs `504`; NBG export failed | fail by `>=0.99` rule | not run by literal host-gate rule; `gen_nbg` segfaulted before valid NBG | 0 bytes (`gen_nbg` failed) |
| SmolLM2-1.7B-Instruct | 64 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.285665032`, top-1 mismatch `198` vs `504`; NBG export failed | fail by `>=0.99` rule | not run by literal host-gate rule; ACUITY export left 0-byte NBG | 0 bytes (`gen_nbg` failed) |
| SmolLM2-1.7B-Instruct | 128 | verified cosine `1.000000000`, top-1 `504` match | verified cosine `0.222273584`, top-1 mismatch `970` vs `504`; NBG export failed | fail by `>=0.99` rule | not run by literal host-gate rule; ACUITY export left 0-byte NBG | 0 bytes (`gen_nbg` failed) |
| SmolLM2-1.7B-Instruct | 256 | verified cosine `1.000000000`, top-1 `504` match | pending | pending | not run yet | pending |

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
- Verified the missing public checkpoints were downloaded:
  - `work/models/smollm2-360m-instruct/model.safetensors`, `723,674,912`
    bytes, unsharded.
  - `work/models/smollm2-1.7b-instruct/model.safetensors`, `3,422,777,952`
    bytes, unsharded.
- Verified `make_real_llm_onnx.py` now emits ONNX external data for large
  graphs. The 1.7B ONNX wrappers are small `.onnx` files with
  `real_llm.onnx.data` sidecars of about `6.85 GB`.
- Verified all 12 model/window ONNX graphs pass the FP oracle gate with top-1
  match. Evidence is in `logs/host/b1-smollm2-*-onnxruntime-vs-fp.json`.
