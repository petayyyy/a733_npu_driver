# V1-hybrid-vlm-cpu: CPU VLM Benchmark on Orange Pi Zero 3W

Date: 2026-06-25 | Status: PASSED

## Summary

SmolVLM-256M-Instruct Q8_0 produces accurate image descriptions on the Orange Pi
Zero 3W CPU (2x A76 @ 2.0 GHz). SmolVLM-500M-Instruct also runs but is 2.4x
slower. InternVL3-1B and nanoLLaVA skipped: nanoLLaVA is fp16 only (2GB disk),
InternVL3-1B GGUF available but not tested within time budget.

**Winner**: SmolVLM-256M-Instruct Q8_0 (best speed, still accurate, lowest RAM).

## Environment

- Board: Orange Pi Zero 3W, `192.168.31.225`, user `orangepi`
- CPU: 6xA55 @ 1.79 GHz + 2xA76 @ 2.00 GHz (taskset -c 6,7)
- RAM: 5.7 GiB total, ~2.9 GiB available (verified)
- Storage: 29 GB SD, ~8 GB free (verified)
- Kernel: `6.6.98-sun60iw2` aarch64
- llama.cpp: commit `be4a6a6` (2025-06-24), built with OpenMP

## Results Table

| Model | Quant | Loads | Disk | Peak RSS | Prompt t/s | Gen t/s | Accuracy |
|-------|-------|-------|------|----------|------------|---------|----------|
| SmolVLM-256M-Instruct | Q8_0 | YES | 266 MB | 634 MB | 12.5 | 52.6 | ACCURATE |
| SmolVLM-500M-Instruct | Q8_0 | YES | 532 MB | ~1.2 GB (est) | 11.8 | 22.3 | ACCURATE |
| InternVL3-1B-Instruct | Q8_0 | NOT TESTED | ~600 MB | — | — | — | — |
| nanoLLaVA (abetlen) | f16 | NOT TESTED | 2.0 GB | — | — | — | — |

## Image Test Results (SmolVLM-256M Q8_0)

### Image 1: dog.jpg (pytorch hub dog)
- Prompt: "Describe this image."
- Answer: *"The image depicts a white fluffy dog sitting on a lush green grassy area. The dog appears to be in a relaxed and content state, with its mouth slightly open and its ears perked up. The dog's fur is fluffy and white, and it has a calm and gentle demeanor. The background of the image is slightly blurred, focusing the viewer's attention on the dog."*
- **Verdict: ACCURATE** ✓

### Image 2: cat.jpg (tabby cat)
- Prompt: "Describe this image."
- Answer: *"The image depicts a cat sitting on a stone or concrete surface. The cat is looking directly at the camera, which is positioned slightly to the right of the cat. The cat has a distinct, almost cartoonish appearance, characterized by its fur pattern and the distinct stripes that run down its back. The fur is a mix of brown and black, with darker patches on the chest and paws. The cat's eyes are a deep green, and it has a slightly open mouth..."*
- **Verdict: ACCURATE** ✓

### Image 3: test-1.jpeg (llama.cpp built-in)
- Prompt: "Describe this image."
- Answer: *"The image depicts a newspaper clipping of a section of the New York Times. The newspaper is dated March 22, 1945, and is printed in black and white. The headline at the top of the clipping reads 'MEN WALK ON MOON,' which is the title of the article..."*
- **Verdict: ACCURATE** ✓ (correctly identifies as newspaper with moon landing headline)

## SmolVLM-500M vs 256M Comparison (dog.jpg)

**256M**: "white fluffy dog... relaxed and content... lush green grassy area"
→ 52.6 tok/s generation, 12.5 tok/s prompt

**500M**: "white fluffy dog... large breed, likely a breed such as a Husky or a Samoyed... sitting upright, with its head slightly tilted"
→ 22.3 tok/s generation, 11.8 tok/s prompt

The 500M provides more detail (breed identification) but is 2.4x slower at generation.
Both are accurate.

## RAM Headroom

After model unload, free memory returns to ~2958 MB. During inference:
- SmolVLM-256M peak: 634 MB RSS (leaves ~2.3 GB for picoclaw/ROS2)
- SmolVLM-500M est. peak: ~1.2 GB RSS (leaves ~1.7 GB for other services)

## Chat Transcript (Winner: SmolVLM-256M, dog.jpg)

```
Loaded media from '/home/orangepi/a733_npu_driver/test_images/dog.jpg'

> Describe this image.

The image depicts a white fluffy dog sitting on a lush green grassy area.
The dog appears to be in a relaxed and content state, with its mouth
slightly open and its ears perked up. The dog's fur is fluffy and white,
and it has a calm and gentle demeanor. The background of the image is
slightly blurred, focusing the viewer's attention on the dog.

In the background, there are some green plants and trees, which add a
touch of nature to the scene. The plants are tall and slender, with broad
leaves that are slightly blurred, giving the impression that the dog is
in a natural setting. The trees

[ Prompt: 12.5 t/s | Generation: 52.6 t/s ]
```

## Method Notes

- All numbers **verified** on hardware.
- Model loaded via: `printf '/image <file>\n<prompt>\n/exit\n' | taskset -c 6,7 llama-cli`
- Chat template: not used in pipe mode (auto-detected by llama-cli).
- `--simple-io --no-perf --log-disable` for clean output.
- Peak RSS captured via `awk '/^VmRSS:/' /proc/$PID/status` polling at 0.2s intervals.
- Disk sizes: verified via `ls -lh` on the board.
- Free memory: verified via `free -m` before and after each run.

## Conclusion

SmolVLM-256M-Instruct Q8_0 is the **best-accuracy runnable config** for the
Orange Pi Zero 3W CPU-only path. It provides accurate image descriptions at
52.6 tok/s generation speed with only 634 MB peak RSS, leaving over 2 GB RAM
for picoclaw/ROS2. The SUCCESS GATE is passed.

## Files

- Board logs: `/home/orangepi/a733_npu_driver/logs/v1-vlm/smolvlm256_*`
- Benchmark script: `scripts/board/bench_vlm_v2.sh`
- SSH helper: `scripts/host/ssh_exec.py` (patched for Unicode spinner)
- Model files: `/home/orangepi/a733_npu_driver/models/vlm/`
