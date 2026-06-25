# V2d-vlm-npu-mmproj-glue: PASSED — Hybrid VLM with NPU Vision Offload

Date: 2026-06-25 | Status: **SUCCESS GATE PASSED** ✅

## Summary

SmolVLM-256M-Instruct image chat runs end-to-end on Orange Pi Zero 3W with the
SigLIP vision encoder on the NPU and the LLM on CPU. NPU-produced embeddings
(1×64×576 float32) are injected into llama.cpp via the V2c mtmd patch
(`A733_NPU_EMBEDDINGS` env var). All 3 V1 test images produce **accurate**
answers matching V1 CPU-only quality.

## How It Works

1. **Image preprocessing**: Resize 512×512, normalize mean=0.5 std=0.5 → int16
   DFP fl=15 input (1×3×512×512, 1.5 MB).
2. **NPU vision encode**: `vpm_run` on SmolVLM SigLIP NBG (271 MB, Conv→MatMul
   rewrite, int16). Output: dequantized float32 embeddings (1×64×576, 36864
   values). Profile time 5,958 ms. Verified: `cid=0x1000003b`, `vpm run ret=0`.
3. **NPU embedding injection**: llama-cli with `--mmproj` and `--image` loads
   the SmolVLM model. The V2c mtmd patch reads float32 embeddings from file
   when `A733_NPU_EMBEDDINGS` is set, replacing the CPU vision encoder output.
4. **LLM decode**: llama-cli generates the answer at ~46.5 tok/s on 2×A76
   cores. The `<image>` marker in the raw prompt text (not chat template) tells
   mtmd where to place the image token embeddings.

## Key Discovery: llama-cli Media Marker Bug

The SmolVLM chat template (`--chat-template smolvlm`) does not include `<image>`
in the generated prompt text. When `--image` is used with `--chat-template
smolvlm`, mtmd_tokenize finds 0 media markers but 1 bitmap → `Failed to
tokenize prompt`. This is a bug in llama.cpp `be4a6a6`.

**Workaround**: Use raw `<image>` in the prompt with `--simple-io --no-perf
--log-disable` (no `--chat-template`). PiSpeed `/exit` through stdin to make
llama-cli exit after generation.

## Verified Embedding Quality

| Metric | Value |
|--------|-------|
| ACUITY host int16 vs PyTorch FP32 cosine | **0.9945** |
| Board NPU int16 vs ONNX Runtime FP32 (dog.jpg) | **0.9972** |
| Per-token cosine (min/mean/max) | 0.9944 / 0.9975 / 0.9998 |
| **Quality gate (>0.95)** | **PASSED** ✅ |

## E2E Accuracy (verified on Orange Pi)

### dog.jpg — "What animal is in this image?"
> The image features a white fluffy dog with a thick coat of fur. The dog has a
> calm and attentive demeanor, with its ears perked up and its mouth slightly
> open... The background of the image is a lush green lawn...

**ACCURATE** — matches V1 CPU-only description ("white fluffy dog sitting on a
lush green grassy area").

### cat.jpg — "What animal is in this image?"
> The animal in this image is a cat.

**ACCURATE** — identifies the cat correctly.

### test-1.jpeg (moon-landing newspaper) — "Describe this image."
> The image is a newspaper clipping of a copy of the New York Times... The
> headline on the clipping reads: "MEN WALK ON MOON." The headline is written
> in bold, capital letters...

**ACCURATE** — correctly reads "MEN WALK ON MOON" from the newspaper clipping.

## Final Measurements

| | CPU VLM (V1) | NPU Vision (V2d) |
|---|---|---|
| Vision latency | ~1-2 sec (estimated) | **5.94 sec** (measured) |
| Vision create network | — | 237 ms |
| Vision prepare network | — | 12.4 s (first-time NBG load) |
| Embedding quality (cosine vs FP32) | 1.000 | **0.9972** |
| A76 cores used for vision | 2 (fully loaded) | **0** (NPU only) |
| Prompt throughput | 174 t/s | 174 t/s (same) |
| Generation throughput | 47.8 t/s (dog) | 46.5 t/s (avg) |
| A76 free for ROS2 | 0 | **2** |
| Vision memory | ~634 MB (full VLM RAM) | **21 MB** (NPU pool) |
| Vision disk | 266 MB (text+mmproj GGUF) | **271 MB** (NBG) |
| Answer accuracy | accurate | **accurate (verified)** ✅ |

## Known Limitations

1. **NPU slower than CPU for vision** (5.94 sec vs ~1-2 sec estimated). But NPU
   frees both A76 cores entirely.
2. **llama-cli non-interactive mode**: `llama-cli` rejects `--no-conversation`
   and `llama-completion` lacks `--mmproj`. Workaround: pipe `/exit` through
   stdin.
3. **Chat template bug**: SmolVLM chat template in llama.cpp `be4a6a6` does
   not insert `<image>` marker. Must use raw prompt format.
4. **NBG prepare time**: First-time load of 271 MB NBG takes 12.4 sec.
   Subsequent runs on same VIPLite session would be faster.

## Files

- NBG package: `work/model-packages/smolvlm_256m_vision_v2d/int16/`
- Board NBG: `/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16/`
- ONNX export: `scripts/host/export_smolvlm_vision_v2b.py`
- ONNX fix: `scripts/host/fix_onnx_v2d.py`
- Injector (alternate): `scripts/board/npu_vlm_injector.c`
- Mtmd patch (V2c, already on board): `tools/mtmd/mtmd.cpp` in llama.cpp
- E2E runner: `scripts/board/run-v2d-e2e.sh`
- Calibration prep: `scripts/host/prepare_v2d.py`
- Board logs: `/home/orangepi/a733_npu_driver/logs/v2d/`

## Conclusion

**SUCCESS GATE: PASSED.** SmolVLM-256M hybrid VLM runs end-to-end on Orange Pi
Zero 3W with vision on NPU (5.94 sec, 0 CPU, 21 MB pool) and LLM on CPU
(~46.5 tok/s, 2×A76). All 3 test images produce accurate answers matching V1
quality. CPU offload quantified: 2 A76 cores freed for ROS2. This is a
**proven deliverable**.
