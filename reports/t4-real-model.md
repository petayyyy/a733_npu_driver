# T4 Real Model On NPU

Date: 2026-06-22

## Purpose

Verified: T4 starts from the T1 persistent runner, T2 faithful decoder operator
set, and T3 last-logits `pcq` path. The first real target is
`HuggingFaceTB/SmolLM2-135M-Instruct`; Qwen2.5-0.5B is intentionally held until
SmolLM2 either passes or produces a precise blocker.

Verified: CPU work in this task is limited to allowed orchestration and build
steps: downloading HF checkpoint files, generating ONNX, running ACUITY, and
later tokenization/argmax/logging. The ONNX graph places token embedding,
attention, MLP, RMSNorm, and logits inside the graph for NPU execution.

## Code Changes

Verified:

- Added `scripts/host/make_real_llm_onnx.py`.
- Added `scripts/host/smollm2_tokenizer.py`.
- Added `scripts/host/smollm2_numpy_reference.py` as a CPU correctness oracle
  for the same fixed-window graph semantics.
- Added `scripts/host/make_smollm2_calibration.py`.
- Added `scripts/board/run-npu-lm-runner-rss.sh`.
- The generator reads `config.json` and a single `model.safetensors` directly;
  no CPU framework is used for model-layer execution.
- The generator supports fixed windows through `--seq-len`, with T4 starting at
  `W=32`.
- The graph emits last-token logits only:
  `final hidden [1,W,576] -> Slice(axis=1, W-1:W) -> [1,1,576] -> MatMul -> [1,1,49152]`.

## SmolLM2 W=32 ONNX Build

Verified source files:

```text
work/models/smollm2-135m-instruct/config.json
work/models/smollm2-135m-instruct/model.safetensors
work/models/smollm2-135m-instruct/tokenizer.json
work/models/smollm2-135m-instruct/tokenizer_config.json
work/models/smollm2-135m-instruct/generation_config.json
```

Verified model config:

```text
layers=30
hidden_size=576
intermediate_size=1536
attention_heads=9
kv_heads=3
head_dim=64
vocab_size=49152
rope_theta=100000
rms_norm_eps=1e-5
tie_word_embeddings=true
```

Verified command:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ubuntu-npu:v2.0.10.1 \
  python3 scripts/host/make_real_llm_onnx.py \
    --model-dir work/models/smollm2-135m-instruct \
    --output-dir work/generated/smollm2_135m_w32 \
    --seq-len 32
```

Verified generated artifacts:

```text
work/generated/smollm2_135m_w32/real_llm.onnx      651,500,555 bytes
work/generated/smollm2_135m_w32/token_ids.npy      256 bytes
work/generated/smollm2_135m_w32/dataset.txt
work/generated/smollm2_135m_w32/inputs_outputs.txt
work/generated/smollm2_135m_w32/model_info.json
```

Verified validation token window:

```text
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 9690 198 2683 359 260 1730 30 2
```

Assumption: the default token window is acceptable for ACUITY calibration while
the tokenizer-driven prompt/decode path is added for the board run after NBG
conversion succeeds.

## SmolLM2 W=32 PCQ Conversion

Verified command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32 \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32/dataset.txt \
  --quant pcq \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

Verified logs:

```text
logs/host/t4-smollm2-w32-pcq-convert.log
logs/host/t4-smollm2-w32-pcq-convert.err.log
```

Verified conversion result:

```text
ONNX import: SUCCESS
quantization: Error(0),Warning(61)
host inference: completed
final NBG export: Error(0),Warning(0)
package: work/model-packages/smollm2_135m_w32/pcq/
network_binary.nb: 153,990,896 bytes
```

Verified package metadata:

```text
input:  token_ids, int32, shape 1x32
output: logits, int8 asymmetric affine, shape 1x1x49152
output scale: 0.1845247447490692
output zero_point: -55
```

Verified ACUITY export simulator timing:

```text
Create Neural Network: 21508ms
Verify Graph: 44701ms
Run the 1 time: 7742.66ms
```

Verified: the `Warning(61)` entries are range-metadata warnings of the same
class seen in T2/T3, e.g. `Edge "..._rms_squared_..." has set the range
already`. Verified no `unsupported`, `not support`, or `fallback` blocker
appeared in the conversion logs.

## SmolLM2 W=32 PCQ Board Result

Verified: both `pcq` packages load and run on the A733 NPU through the T1
persistent runner. Verified runtime evidence:

```text
VIPLite: 2.0.3.2-AW-2024-08-30
cid=0x1000003b
nbg_loaded_once=1
vpm run ret=0 for sample.txt validation run
```

Verified original one-sample `pcq` package:

```text
package: /home/radxa/a733_npu_driver/models/smollm2_135m_w32_pcq
mean_wall_us=34243.125
mean_profile_us=25632.312
mean_tok_s=29.203
decoded: ... <|im_start|>assistant
 the the the  the the **   the  the the   **
```

Verified multi-window calibrated `pcq` package:

```text
package: /home/radxa/a733_npu_driver/models/smollm2_135m_w32_calib_pcq
network_binary.nb: 153,984,304 bytes
output scale: 0.11849070340394974
output zero_point: -70
mean_wall_us=30187.250
mean_profile_us=25540.562
mean_tok_s=33.127
decoded: ... <|im_start|>assistant
 the  the$ interspers strugg strugg strugg strugg  the the" Kaw strugg strugg
```

Verified: the `pcq` output is not coherent and does not match the CPU oracle.
For the same default chat prompt, the CPU oracle starts:

```text
The capital of France is Paris. Paris is a city located in the northern part
```

Verified exact first-token mismatch for calibration sample
`The capital of France is` with pad token `2`:

```text
CPU FP fixed-window oracle top-5:
253:23.340405, 354:21.232018, 260:21.220097, 441:20.994089, 582:20.156054

calibrated pcq ACUITY host top-5:
37353:20.97285461, 44696:19.19549370, 48398:19.19549370,
2581:19.07700348, 21560:18.95851326

calibrated pcq board vpm_run top-5:
2581:17.773605, 260:17.536625, 357:17.536625, 34:17.062662, 33:16.825680
```

Verified calibrated `pcq` board-vs-ACUITY-host comparison for the same
`sample.txt` input:

```text
length: 49152
top-5 index match: no
max abs diff: 8.294349670
mean abs diff: 2.909518997
RMSE: 3.063358207
cosine: 0.992959037
```

Conclusion: verified `pcq` conversion and NPU execution work mechanically at
full SmolLM2 scale, but `pcq` does not pass the T4 coherence check. Assumption:
this is an int8 quantization/range issue for real transformer logits at scale,
not an unsupported-op issue, because the same graph is coherent in FP CPU
oracle and in int16 NPU execution below.

## SmolLM2 W=32 Int16 Control

Verified command:

```bash
scripts/host/convert_onnx_to_nbg.sh \
  --name smollm2_135m_w32_int16 \
  --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
  --dataset work/generated/smollm2_135m_w32_calib/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

Verified conversion result:

```text
ONNX import: SUCCESS
quantization: Error(0),Warning(61)
final NBG export: Error(0),Warning(0)
network_binary.nb: 280,882,632 bytes
output: int16 dynamic fixed point, fl=10, shape 1x1x49152
ACUITY export simulator one run: 15703.15ms
```

Verified board path:

```text
/home/radxa/a733_npu_driver/models/smollm2_135m_w32_int16
```

Verified board runtime through the persistent runner:

```text
VIPLite: 2.0.3.2-AW-2024-08-30
cid=0x1000003b
nbg_loaded_once=1
input: int32 1x32
output: int16 1x1x49152, dfp=10
memory_pool_bytes=0
peak_rss_kb=278176
```

Verified coherent int16 NPU output for the same default chat prompt:

```text
<|im_start|>assistant
The capital of France is Paris, located in the northern part of the country.
```

Verified CPU oracle output for the same prompt:

```text
<|im_start|>assistant
The capital of France is Paris. Paris is a city located in the northern part
```

Verified first six generated tokens match CPU oracle exactly:

```text
504 3575 282 4649 314 7042
The capital of France is Paris
```

Verified int16 benchmark with RSS sampler:

```text
create_network_us=296038
prepare_network_us=7281
first_step_wall_us=46046
first_step_profile_us=41052
mean_wall_us=46905.250
mean_profile_us=41882.625
mean_tok_s=21.320
peak_rss_kb=278176
```

Verified usable context window: `W=32` tokens. The current graph is fixed-window
and recomputes the whole 32-token window for every decode step; no dynamic
KV-cache is used.

## Result

Verified: SmolLM2-135M-Instruct runs as a real full-depth fixed-window NPU graph
on the A733 through the persistent runner and produces coherent text with
`int16` quantization. Verified all model-layer compute in this path is inside
the NBG: token embedding, RMSNorm, RoPE, GQA attention, SwiGLU MLP, residuals,
and logits. CPU work is orchestration, tokenizer/detokenizer, prompt window
updates, argmax, and logging.

Verified: the requested `pcq` int8 path converts and executes, but fails the
coherence check and is a precise quality blocker for an int8 T4 deliverable.
Qwen2.5-0.5B is deferred until the SmolLM2 `pcq` blocker is accepted/escalated
or the T4 target is explicitly treated as int16.

## Next

If int8 is mandatory, proceed to T6 with the `pcq` quality blocker and include
the logs above. If int16 is accepted for T4, the next technical step is trying
`W=64` for SmolLM2 before any Qwen2.5-0.5B attempt.
