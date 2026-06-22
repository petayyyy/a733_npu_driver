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

## SmolLM2 W=32 Result

Verified: SmolLM2-135M-Instruct runs as a real full-depth fixed-window NPU graph
on the A733 through the persistent runner and produces coherent text with
`int16` quantization. Verified all model-layer compute in this path is inside
the NBG: token embedding, RMSNorm, RoPE, GQA attention, SwiGLU MLP, residuals,
and logits. CPU work is orchestration, tokenizer/detokenizer, prompt window
updates, argmax, and logging.

Verified: the requested `pcq` int8 path converts and executes, but fails the
coherence check and is a precise quality blocker for an int8 T4 deliverable.

## SmolLM2 W=64 Int16

Verified: after W=32 passed on the int16 path, the same real SmolLM2 graph was
rebuilt for `W=64`.

Verified generated artifacts:

```text
work/generated/smollm2_135m_w64/real_llm.onnx      651,529,233 bytes
work/generated/smollm2_135m_w64_calib/dataset.txt  12 calibration windows
```

Verified conversion result:

```text
logs/host/t4-smollm2-w64-int16-convert.log
logs/host/t4-smollm2-w64-int16-convert.err.log
ONNX import: SUCCESS
quantization: Error(0),Warning(61)
final NBG export: Error(0),Warning(0)
network_binary.nb: 282,310,408 bytes
output: int16 dynamic fixed point, fl=10, shape 1x1x49152
ACUITY export simulator one run: 15859.64ms
```

Verified board path:

```text
/home/radxa/a733_npu_driver/models/smollm2_135m_w64_int16
```

Verified board runtime:

```text
VIPLite: 2.0.3.2-AW-2024-08-30
cid=0x1000003b
nbg_loaded_once=1
input: int32 1x64
output: int16 1x1x49152, dfp=10
memory_pool_bytes=345088
peak_rss_kb=280904
```

Verified W=64 CPU oracle output:

```text
<|im_start|>assistant
The capital of France is Paris. It is a city that has a rich history
```

Verified W=64 NPU int16 output:

```text
<|im_start|>assistant
The capital of France is Paris, a city that is known for its rich history
```

Verified W=64 int16 benchmark with RSS sampler:

```text
create_network_us=770561
prepare_network_us=27492
first_step_wall_us=71560
first_step_profile_us=64861
mean_wall_us=69655.812
mean_profile_us=64891.688
mean_tok_s=14.356
peak_rss_kb=280904
```

Verified usable context window is now `W=64` on the int16 path.

## Qwen2.5-0.5B W=32 Setup

After the user explicitly requested continuing T4 with Qwen despite the SmolLM2
`pcq` quality blocker, started the Qwen2.5-0.5B-Instruct path.

Verified downloaded Hugging Face files:

```text
work/models/qwen25-0.5b-instruct/config.json                  659 bytes
work/models/qwen25-0.5b-instruct/generation_config.json       242 bytes
work/models/qwen25-0.5b-instruct/model.safetensors      988,097,824 bytes
work/models/qwen25-0.5b-instruct/tokenizer.json         7,031,645 bytes
work/models/qwen25-0.5b-instruct/tokenizer_config.json       7,305 bytes
```

Verified Qwen config:

```text
model_type=qwen2
hidden_size=896
intermediate_size=4864
num_hidden_layers=24
num_attention_heads=14
num_key_value_heads=2
head_dim=64
vocab_size=151936
rope_theta=1000000.0
rms_norm_eps=1e-6
tie_word_embeddings=true
```

Updated the fixed-window real-LM ONNX generator for Qwen:

- q/k/v projection biases are now represented in the ONNX graph.
- Missing q/k/v biases remain supported as zero tensors for SmolLM2.
- The tied `lm_head` is now represented as `Transpose(token_embed)` instead of
  duplicating the embedding initializer; this kept the Qwen ONNX below the
  protobuf-size cliff.

Added Qwen host helpers:

```text
scripts/host/qwen2_tokenizer.py
scripts/host/make_qwen2_calibration.py
```

Verified diagnostic one-layer Qwen W=32 ONNX generation:

```text
work/generated/qwen25_05b_w32_layer1/real_llm.onnx  604,219,825 bytes
```

Verified full Qwen2.5-0.5B-Instruct W=32 ONNX generation:

```text
work/generated/qwen25_05b_w32/real_llm.onnx  1,976,297,294 bytes
layers=24
output=last-token logits
lm_head=transpose(model.embed_tokens.weight)
```

Verified Qwen W=32 calibration dataset:

```text
work/generated/qwen25_05b_w32_calib/dataset.txt  12 calibration windows
pad_token_id=151643
```

Started ACUITY `pcq` conversion for the full Qwen W=32 graph:

```text
logs/host/t4-qwen25-05b-w32-pcq-convert.log
logs/host/t4-qwen25-05b-w32-pcq-convert.err.log
container=tender_buck
```

Full `pcq` observed state before stopping the container:

```text
ONNX import: SUCCESS
qwen25_05b_w32.data: 2,521,343,935 bytes
entropy.txt: 67,137 bytes
last log line: End quantization / Dump net quantize tensor table
container status: still running before manual stop
active process: python3 pegasus.py quantize
elapsed in quantize: over 80 minutes at final observation time
CPU: about 100%
RSS: about 5.0 GiB
qwen25_05b_w32_pcq.quantize: 0 bytes at observation time
IO counters: unchanged across repeated samples
```

Interpretation: the full Qwen `pcq` path did not crash, but it stalled inside
ACUITY quantize after printing `End quantization`. The container `tender_buck`
was stopped after repeated zero-IO checks. No full Qwen `pcq` NBG was exported.

Verified Qwen-shaped diagnostic `pcq` export with one real decoder layer:

```text
logs/host/t4-qwen25-05b-w32-layer1-pcq-convert.log
logs/host/t4-qwen25-05b-w32-layer1-pcq-convert.err.log
ONNX import: SUCCESS
quantization: SUCCESS
final NBG export: Error(0),Warning(0)
network_binary.nb: 274,904,704 bytes
output: int8 asymmetric affine, shape 1x1x151936
ACUITY export simulator: create network 1.894s, verify 15.907s, one run 4.547s
```

This confirms the Qwen graph pattern and large-vocabulary sliced-logits output
are accepted by ACUITY on a smaller Qwen diagnostic graph; the full `pcq`
blocker is scale-specific to the 24-layer graph.

Verified full Qwen2.5-0.5B-Instruct W=32 `int16` control export:

```text
logs/host/t4-qwen25-05b-w32-int16-convert.log
logs/host/t4-qwen25-05b-w32-int16-convert.err.log
ONNX import: SUCCESS
quantization: SUCCESS
final NBG export: Error(0),Warning(0)
network_binary.nb: 1,064,540,800 bytes
output: int16 dynamic fixed point, fl=11, shape 1x1x151936
ACUITY export simulator: create network 14.018s, verify 64.928s, one run 49.430s
```

Board run is blocked at this checkpoint by host-to-board network access from
the current environment:

```text
python/paramiko: WinError 10013
ping 192.168.31.76: General failure
ssh radxa@192.168.31.76: Permission denied while connecting to port 22
```

This is a connectivity/permission blocker from the current host environment,
not a Qwen NBG export blocker and not a reason to request board power cycling.

After network access recovered, uploaded the full Qwen W=32 `int16` package to
the Radxa:

```text
board path: /home/radxa/a733_npu_driver/models/qwen25_05b_w32_int16
network_binary.nb: 1,064,540,800 bytes
board free space before upload: 23G
```

Verified full Qwen W=32 `int16` board smoke is blocked by board RAM:

```text
logs/board/qwen25_05b_w32_int16_smoke-run.log
logs/board/qwen25_05b_w32_int16_smoke-rss.env
runner status: 137
peak_rss_kb: 641,340
run.log: empty
board memory after kill: 959Mi total, 641Mi available, 2.3Gi swap available
```

Interpretation: the 1.016GiB Qwen `int16` NBG does not fit on the current
1GiB Radxa board configuration. The process was killed before the runner could
print VIPLite metadata or execute the graph.

Uploaded and ran the Qwen W=32 one-layer `pcq` diagnostic package on the A733:

```text
board path: /home/radxa/a733_npu_driver/models/qwen25_05b_w32_layer1_pcq
logs/board/qwen25_05b_w32_layer1_pcq_smoke-run.log
logs/board/qwen25_05b_w32_layer1_pcq_smoke-rss.env
network_binary.nb: 274,904,704 bytes
VIPLite: 2.0.3.2-AW-2024-08-30
cid: 0x1000003b
input: int32 1x32
output: int8 asymmetric affine 1x1x151936
memory_pool_bytes: 214,016
nbg_loaded_once: 1
status: 0
```

Verified Qwen layer1 `pcq` persistent-runner timing:

```text
create_network_us=583531
prepare_network_us=744
first_step_wall_us=46465
first_step_profile_us=19217
mean_wall_us=45572.250
mean_profile_us=19196.750
mean_tok_s=21.943
peak_rss_kb=270220
```

Generated layer1 diagnostic tokens:

```text
56446 56446 56446 732
decoded: forgettableforgettableforgettable im
```

This is not a coherence result because the graph contains only one decoder
layer. It is a hardware control proving that a Qwen-shaped fixed-window graph
with Qwen tokenizer windows, q/k/v bias, RoPE, GQA, large-vocabulary sliced
logits, and `pcq` output can execute on the A733 NPU through the persistent
runner.

## Qwen2.5-0.5B Layer Bisection

Verified Qwen W=32 four-layer `pcq` diagnostic export:

```text
work/generated/qwen25_05b_w32_layer4/real_llm.onnx  783,186,037 bytes
logs/host/t4-qwen25-05b-w32-layer4-pcq-convert.log
logs/host/t4-qwen25-05b-w32-layer4-pcq-convert.err.log
ONNX import: SUCCESS
quantization: SUCCESS
inference: completed
final NBG export: Error(0),Warning(0)
network_binary.nb: 316,117,184 bytes
output: int8 asymmetric affine, shape 1x1x151936
ACUITY export simulator: create network 1.726s, verify 19.259s, one run 6.279s
```

Note: the local wrapper exited `2` after ACUITY export because the currently
dirty `scripts/host/convert_onnx_to_nbg.sh` T5 edits have a post-export
packaging syntax issue. The NBG itself was already exported successfully, so
the package was assembled manually from ACUITY's `_nbg_unify` directory for the
board smoke.

Uploaded and ran the Qwen W=32 four-layer `pcq` diagnostic package on the A733:

```text
board path: /home/radxa/a733_npu_driver/models/qwen25_05b_w32_layer4_pcq
logs/board/qwen25_05b_w32_layer4_pcq_smoke-run.log
logs/board/qwen25_05b_w32_layer4_pcq_smoke-rss.env
network_binary.nb: 316,117,184 bytes
VIPLite: 2.0.3.2-AW-2024-08-30
cid: 0x1000003b
input: int32 1x32
output: int8 asymmetric affine 1x1x151936
memory_pool_bytes: 214,016
nbg_loaded_once: 1
status: 0
```

Verified Qwen layer4 `pcq` persistent-runner timing:

```text
create_network_us=416151
prepare_network_us=1181
first_step_wall_us=47515
first_step_profile_us=26347
mean_wall_us=42211.500
mean_profile_us=26254.750
mean_tok_s=23.690
peak_rss_kb=309920
```

Generated layer4 diagnostic tokens:

```text
0 52643 120889 100091
decoded: !ascus棰主义
```

This is still a partial-model diagnostic, not a coherence result. It narrows
the Qwen `pcq` blocker: 4 real decoder layers export and run; 24 layers stall
inside ACUITY quantize-table serialization/rebuild.

## Result

Verified: SmolLM2-135M-Instruct passed the NPU-only coherent-text gate with
`int16` at both `W=32` and `W=64`. Verified: the requested `pcq` int8 path
converts and executes but fails coherence at `W=32`.

Qwen2.5-0.5B-Instruct has now reached full W=32 ONNX generation. ACUITY full
`pcq` quantization stalls after `End quantization`, but a one-layer Qwen `pcq`
diagnostic export passes and the full 24-layer `int16` control export passes.
On the A733 board, the full Qwen W=32 `int16` NBG is blocked by RAM, while
one-layer and four-layer Qwen `pcq` diagnostic NBGs run successfully on the
NPU.

## Next

For a full Qwen board run on this 1GiB Radxa, the viable path is a full `pcq`
or smaller-than-int16 package. The current blocker is ACUITY full-Qwen `pcq`
quantize-table serialization/rebuild. If that cannot be cleared, continue with
layer-count bisection above four layers or a smaller model/graph; full Qwen
`int16` is too large for the board RAM.
