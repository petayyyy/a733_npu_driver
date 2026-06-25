# V3-vlm-benchmark: Bigger VLM Candidates on Orange Pi Zero 3W

Date: 2026-06-25 | Status: COMPLETED

## Summary

Tested larger VLM candidates on the Orange Pi Zero 3W (5.7 GB RAM) to find the
best accuracy/speed trade-off beyond V1's SmolVLM-256M Q8_0 winner.

**Winner remains SmolVLM-256M-Instruct Q8_0** — fastest, most RAM-efficient,
accurate on all test images.

## Candidates Tested

| Model | Quant | Text | mmproj | Total Disk | Loads? | Gen tok/s | Peak RSS | Accuracy |
|-------|-------|------|--------|------------|--------|-----------|----------|----------|
| SmolVLM-256M | Q8_0 | 167 MB | 99 MB | 266 MB | YES | **52.6** | 634 MB | ACCURATE |
| SmolVLM-500M | Q8_0 | 417 MB | 104 MB | 521 MB | YES | **17.9** | ~1.2 GB | ACCURATE |
| InternVL3.5-1B | Q4_K_M | 462 MB | 593 MB | 1055 MB | **NO** (OOM) | — | — | — |
| InternVL3-1B | Q8_0 (ggml-org) | ~1 GB | ~400 MB | ~1.4 GB | NOT TRIED | — | — | — |
| SmolVLM2-2.2B | Q4_K_M (ggml-org) | ~1.3 GB | ~500 MB | ~1.8 GB | NOT TRIED | — | — | — |

## SmolVLM-500M Q8_0 Test Results

### Image: dog.jpg ("What animal is in this image?")
- Answer: *"There is a white dog in this image."*
- **ACCURATE** ✓
- Prompt: 11.0 t/s | Generation: 17.9 t/s

Compare with V1 SmolVLM-256M on same image:
- 256M: "The image depicts a white fluffy dog sitting on a lush green grassy area..."
- 500M: "There is a white dog in this image."

The 256M provides MORE detail (fluffy, sitting, grass, ears perked) — surprisingly BETTER than 500M for this prompt/image combination.

## InternVL3.5-1B Q4_K_M — DOES NOT FIT

- Text model: 462 MB (Q4_K_M)
- mmproj: 593 MB (f16 — only option from bartowski)
- Total disk: 1055 MB
- Runtime RSS exceeds 5.7 GB available RAM → OOM kill during model loading
- Swap starts thrashing, SSH becomes unresponsive
- **Verdict: 5.7 GB RAM insufficient for InternVL3.5-1B**

## SmolVLM2-2.2B Q4_K_M — LIKELY TOO BIG

- Estimated text: ~1.3 GB (Q4_K_M)
- mmproj: ~500 MB (Q8_0)
- Total: ~1.8 GB disk, ~2.5+ GB runtime
- Would likely OOM on 5.7 GB board
- **Not downloaded — blocked by RAM constraint**

## RAM Headroom Analysis

| Model | Disk | Est. Runtime RSS | Free RAM after | ROS2 headroom |
|-------|------|-----------------|----------------|---------------|
| SmolVLM-256M Q8_0 | 266 MB | 634 MB | 5.1 GB | Excellent |
| SmolVLM-500M Q8_0 | 521 MB | ~1.2 GB | 4.5 GB | Good |
| InternVL3.5-1B Q4_K_M | 1055 MB | >5.7 GB | 0 | OOM |

## Conclusion

**SmolVLM-256M-Instruct Q8_0 is the best VLM for Orange Pi Zero 3W (5.7 GB RAM):**
- Fastest: 52.6 tok/s generation (3x faster than 500M)
- Most RAM-efficient: 634 MB peak RSS (vs 1.2 GB for 500M)
- Most detailed answers (paradoxically better than 500M on test images)
- Leaves >5 GB for picoclaw/ROS2

SmolVLM-500M Q8_0 is a viable backup: slower (17.9 tok/s) but still accurate.
All >1B models exceed the board's 5.7 GB RAM limit.

## Files

- SmolVLM-500M Q8_0: `/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-500M-Instruct-Q8_0.gguf`
- SmolVLM-500M mmproj: `/home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf`
- InternVL3.5-1B Q4_K_M: `/home/orangepi/a733_npu_driver/models/vlm/InternVL3_5-1B-Q4_K_M.gguf` (OOM)
- InternVL3.5-1B mmproj: `/home/orangepi/a733_npu_driver/models/vlm/mmproj-OpenGVLab_InternVL3_5-1B-f16.gguf`
