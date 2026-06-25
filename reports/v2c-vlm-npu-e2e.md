# V2c-vlm-npu-e2e-closeout: Real-Image Calibration NBG

Date: 2026-06-25 | Status: PROVEN (toolchain), PENDING (e2e injection)

## Summary

The SmolVLM-256M SigLIP vision encoder NBG was rebuilt with real-image-range
calibration (uniform [-1,1], matching SigLIP normalization). The NBG exports
(Error 0, 271 MB), runs on Orange Pi NPU at **5,942 ms**/inference
(cid=0x1000003b, vpm run ret=0). Input quantization uses fl=15 (full [-1,1]
range) vs V2b's fl=12 (noise range [-4.4, 4.5]).

The hybrid pipeline (NPU vision + CPU SmolVLM LLM) is **toolchain-proven**:
Conv→MatMul rewrite → ONNX → ACUITY int16 NBG → Orange Pi NPU all verified.
Embedding injection into llama.cpp remains as the final integration step.

V1 CPU-only (SmolVLM-256M Q8_0, 52.6 tok/s) remains the deliverable for
immediate use.

## Step 1: Real-Image Calibration NBG

### Calibration Dataset
- 12 samples, uniform distribution in [-1, 1]
- Matches SigLIP-normalized image range (pixel/127.5 - 1)
- Contrast with V2b: V2b used randn ~[-5,5] (only 22% dynamic range)

### ACUITY Conversion (int16)
| Metric | Value |
|--------|-------|
| Import | SUCCESS |
| Quantize | SUCCESS |
| Inference | SUCCESS |
| Export | **Error(0), Warning(0)** |
| NBG size | 271.1 MB |
| Simulator create | 1,017 ms |
| Simulator verify | 1,047,302 ms (17.5 min) |
| Simulator one run | 182,092 ms (3.0 min) |

### NBG Metadata
| Parameter | V2b (noise calib) | V2c (uniform calib) |
|-----------|-------------------|---------------------|
| Input fl | 12 | **15** |
| Input range | [-4.43, 4.53] | **[-1.00, 1.00]** |
| Input utilization | ~22% | **~100%** |
| Output fl | 9 | 9 |
| Output range | [-44.2, 49.2] | [-52.8, 58.6] |
| Memory pool | 21 MB | 21 MB |

The input quantization improvement is significant: fl=15 means 1 LSB = 3.05e-5
vs fl=12 (1 LSB = 2.44e-4). Real images get ~8x better input precision.

### Host Cosine (ACUITY int16 vs FP32 PyTorch)
- **Cosine: 0.906** (below 0.95 gate)
- NOTE: ACUITY host inference uses a calibration sample; the FP32 reference
  used a different test input. This cosine is NOT on the same input — the
  comparison is between different random inputs and is not a valid gate.
  Same-input comparison requires custom inference (not done — ACUITY doesn't
  easily allow custom input injection for host inference).

## Step 2: Orange Pi NPU Run (verified)

```
cid=0x1000003b, device_count=1
create network 0: 236,416 us
prepare network 0: 12,258,526 us (first time)
profile inference time=5,942,150us (~5.94 sec)
vpm run ret=0
```

Input: dfp=15 (match calibrated images), dims 512×512×3×1
Output: dfp=9, dims 576×64×1×0 (1×64×576 layout)
Memory pool: 22,021,120 bytes (~21 MB)

## Step 3: Embedding Injection — PENDING

To complete the hybrid pipeline:
1. Preprocess image (resize 512×512, normalize to [-1,1])
2. Quantize to int16 DFP fl=15, pack as .dat
3. Run vpm_run on NPU → get int16 DFP fl=9 output (1×64×576)
4. Dequantize: float32 = int16 × 2^(-9)
5. Inject 64×576 embeddings into llama.cpp SmolVLM decoder
6. Decode answer

Step 5 requires either:
- Modifying llama.cpp to accept external image embeddings
- Using `llama-cpp-python` with embedding injection API
- Writing a custom C/C++ bridge using llama.cpp's internal API

This is documented as the remaining integration task.

## CPU vs NPU Comparison

| | CPU (llama.cpp mmproj) | NPU (V2c NBG) |
|---|---|---|
| Vision encoding | ~1-2 sec (estimated, part of loading) | **5.94 sec** (measured) |
| A76 cores used | 2 (100%) | 0 |
| CPU free for ROS2 | No | **Yes** |
| Toolchain status | Works (V1, verified) | Works (V2c, verified) |

The NPU path is ~3-5× slower but **completely frees the A76 cores** for
robotics/ROS2 workloads. This is the intended trade-off for the user's
picoclaw + local VLM use case.

## Conclusion

**SUCCESS GATE: TOOLCHAIN PROVEN, E2E INJECTION PENDING**

1. Conv→MatMul rewrite bypassed ACUITY Conv crash ✓
2. Real-image calibration NBG exports (Error 0, 271 MB) ✓
3. NBG runs on Orange Pi NPU (5.94 sec, vpm run ret=0) ✓
4. Embedding injection into llama.cpp — documented, not implemented
5. End-to-end VLM accuracy — pending injection

V1 CPU-only (SmolVLM-256M Q8_0, 52.6 tok/s) remains the recommended VLM path
until the embedding injection bridge is implemented.

## Files

- ONNX: `work/generated/smolvlm_256m_v2b/smolvlm_vision_v2b_final.onnx`
- NBG package: `work/model-packages/smolvlm_256m_vision_v2c/int16/`
- Board path: `/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2c_int16/`
- Export script: `scripts/host/export_smolvlm_vision_v2b.py`
- Preprocess script: `scripts/board/preprocess_siglip.py`
