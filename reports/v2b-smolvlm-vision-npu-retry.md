# V2b-smolvlm-vision-npu-retry: Conv→MatMul Rewrite WORKS

Date: 2026-06-25 | Status: ATTEMPT 1 PASSED (NBG exports, runs on NPU)

## Summary

**ATTEMPT 1 SUCCEEDED.** The SmolVLM-256M SigLIP vision encoder patch-embedding
Conv2d was rewritten as Reshape+MatMul, bypassing ACUITY's `_conv_shape` crash.
The resulting ONNX exported to int16 NBG (271 MB) and runs on the Orange Pi
Zero 3W NPU at `cid=0x1000003b`, `vpm run ret=0`.

End-to-end accuracy validation (NPU embeddings → CPU SmolVLM LLM) is pending
correct image calibration; current NBG used random-noise calibration which
may degrade int16 quality for real images.

## Attempt 1: Conv2d → Reshape+MatMul Rewrite

### Method
- Conv2d(kernel=16, stride=16, in=3, out=768) on 512×512 input
- Equivalent to: split into 32×32=1024 non-overlapping 16×16×3 patches,
  flatten each to 768d, MatMul with reshaped weight [768,768], add bias
- Replaced `vm.embeddings.patch_embedding` (Conv2d) with `PatchEmbedMatMul`
  that uses reshape/permute/matmul operations — ALL supported by ACUITY
- Idefics3 wrapper, NonZero→Constant, and Gather Cast fix kept from V2

### ONNX Export
| Property | Value |
|----------|-------|
| Input | pixel_values: 1×3×512×512 float32 |
| Output | image_embeds: 1×64×576 float32 |
| Size | 356.9 MB |
| Opset | 17 |
| Conv ops | 0 (confirmed) |
| NonZero ops | 0 (replaced with Constant) |
| ONNX Runtime | PASSED |
| vs PyTorch cosine | **1.00000000** |
| vs PyTorch max diff | 0.00005674 |

### ACUITY Conversion (int16)
| Property | Value |
|----------|-------|
| Import | SUCCESS |
| Quantize | SUCCESS |
| Inference | SUCCESS |
| Export | **Error(0), Warning(0)** |
| NBG size | 271.2 MB |
| Simulator create | 1,015 ms |
| Simulator verify | 995,821 ms (16.6 min) |
| Simulator one run | 158,230 ms (2.6 min) |

### NBG Metadata
- Input: int16 DFP, fl=12, shape [1,1,3,512,512], range [-4.43, 4.53]
- Output: int16 DFP, fl=9, shape [1,64,576], range [-44.15, 49.21]
- Memory pool: 22,102,080 bytes (~21 MB)

### Orange Pi NPU Run (verified)
- `cid=0x1000003b`, `vpm run ret=0`
- Create network: 240 ms
- Prepare network: 12.3 s (first time)
- **Inference: 5,959 ms** (~6.0 seconds per image)
- Output: 36864 float32 values (64×576), matches expected shape
- Memory pool: 21 MB

## CPU vs NPU Vision Latency

| | CPU (llama.cpp mmproj) | NPU (V2b NBG) |
|---|---|---|
| Vision encoding | ~1-2 sec (estimated) | 5.96 sec (measured) |
| A76 cores used | 2 (fully loaded) | 0 (NPU runs independently) |
| CPU free for ROS2 | No | Yes |

The NPU is ~3-5x slower than CPU for vision encoding, but **completely offloads
the A76 cores**. For a robotics use case where CPU must stay free for ROS2
control loops, this is a valid trade-off.

## Known Issue: Calibration Dataset Mismatch

The ACUITY calibration used 8 random-noise samples (`np.random.randn`, range
~[-5,5]) instead of real SigLIP-normalized images (range ~[-1,1]). This means:
- Input quantization (fl=12) uses only ~22% of dynamic range for real images
- Internal activation scales were derived from noise statistics, not image statistics
- int16 quality for real images may be degraded vs optimal

**Remediation**: regenerate with 8-16 real-image calibration samples using
correct Idefics3/SigLIP preprocessing (resize→512, normalize mean=std=0.5).

## End-to-End Validation Status: PENDING

Full hybrid pipeline (NPU vision → CPU LLM) not yet validated:
- Need: real-image calibration NBG rebuild (~30 min ACUITY)
- Need: Python script to preprocess image → int16 DFP → .dat format
- Need: feed NPU embeddings into llama.cpp SmolVLM decoder
- Expected: accurate answers on V1 test images (dog, cat, newspaper)

## Conclusion

**SUCCESS GATE: IN PROGRESS** — Attempt 1 Conv→MatMul rewrite succeeds at the
toolchain level (NBG exports, runs on NPU). End-to-end accuracy pending
calibration fix and llama.cpp embedding injection.

## Files

- ONNX export: `scripts/host/export_smolvlm_vision_v2b.py`
- ONNX (final): `work/generated/smolvlm_256m_v2b/smolvlm_vision_v2b_final.onnx`
- NBG package: `work/model-packages/smolvlm_256m_vision_v2b/int16/`
- Board path: `/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2b_int16/`
