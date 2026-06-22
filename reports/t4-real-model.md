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

## Next

Run ACUITY `pcq` conversion for `work/generated/smollm2_135m_w32/real_llm.onnx`.
If ACUITY rejects the graph or hits a size/resource limit, save the full log
under `logs/host/` and record the exact blocker here before moving to T6.
