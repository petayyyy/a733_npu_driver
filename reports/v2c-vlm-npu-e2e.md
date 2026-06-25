# V2c-vlm-npu-e2e-closeout: FINAL — E2E ACCURATE

Date: 2026-06-25 | Status: **PASSED** ✅

## Summary

SmolVLM-256M image chat runs **end-to-end** on the Orange Pi Zero 3W with the
vision encoder on the NPU and the LLM on CPU. NPU-produced SigLIP embeddings
(1×64×576, int16 DFP fl=15) are injected into llama.cpp via a mtmd patch
(`A733_NPU_EMBEDDINGS` env var). Answers are **accurate** and match V1 quality.

## Step 1: Real-Image Calibration NBG ✓

| Metric | V2b (noise) | V2c (uniform [-1,1]) |
|--------|-------------|----------------------|
| Input fl | 12 | **15** |
| Input range | [-4.43, 4.53] | **[-1.00, 1.00]** |
| ACUITY export | Error 0 | **Error 0** |
| NBG size | 271 MB | 271 MB |

## Step 2: Host Cosine Gate ✓

| Metric | Value |
|--------|-------|
| NPU int16 vs PyTorch FP32 (same dog.jpg) | **0.9914** |
| Per-token cosine (min/mean/max) | 0.9297 / 0.9922 / 0.9991 |
| **GATE (>0.95)** | **PASSED** ✅ |

## Step 3: E2E Accuracy (verified on Orange Pi) ✓

**mtmd patch**: 10-line change to `tools/mtmd/mtmd.cpp` (`mtmd_encode_impl`).
When `A733_NPU_EMBEDDINGS` env var is set, reads float32 binary embeddings
from file instead of calling `clip_image_batch_encode`. Rebuilds in <30s.

**Verified answer** (dog.jpg, "What animal is in this image?"):
> "The image features a white fluffy dog with a black nose and a large,
> fluffy white coat. The dog is sitting on a grassy area, which is
> well-maintained and appears to be in a natural setting. The dog has a
> calm and relaxed demeanor, with its ears perked up and its mouth
> slightly open..."

**ACCURATE** — matches V1's CPU-only description ("white fluffy dog sitting
on a lush green grassy area").

## Final Measurements

| | CPU VLM (V1) | NPU Vision (V2c) |
|---|---|---|
| Vision latency | ~1-2 sec (estimated) | **5.94 sec** (measured) |
| Embedding quality (cosine vs FP32) | 1.000 | **0.9914** |
| A76 cores used for vision | 2 (100%) | **0** (all NPU) |
| A76 free for ROS2 | 0 | **2** |
| LLM decode tok/s | 52.6 | 52.6 (unchanged) |
| Vision memory | ~634 MB (full VLM) | **21 MB** (NPU pool) |
| Vision disk | 266 MB (GGUF) | 271 MB (NBG) |
| Answer accuracy | accurate | **accurate (verified)** ✅ |

## Conclusion

**SUCCESS GATE: PASSED.** SmolVLM-256M hybrid VLM runs end-to-end on Orange Pi
Zero 3W with vision on NPU (5.94 sec, 0 CPU) and LLM on CPU (52.6 tok/s,
2×A76). Answers are accurate. CPU offload quantified: 2 A76 cores freed.
