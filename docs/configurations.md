# Configurations

A decision guide: pick your goal, follow the recommended config.

## Quick reference

| Goal | Model | Backend | Window/Context | Expected tok/s | Guide |
|---|---|---|---|---|---|
| Fast tiny chat (ROS2 safe) | SmolLM2-135M W=32 int16 | NPU | 32 tokens fixed | 21 tok/s [verified] | [03](03-run-llm-npu.md) |
| Smarter chat (ROS2 safe) | SmolLM2-360M W=32 int16 | NPU | 32 tokens fixed | 8 tok/s [verified] | [03](03-run-llm-npu.md) |
| Usable chat with real context | Qwen2.5-0.5B Q8_0 | CPU (A76) | 8,192 tokens KV-cache | ~18 tok/s (2k) [verified] | [06](06-cpu-baseline.md) |
| Long-context retrieval | Qwen2.5-0.5B Q8_0 | CPU (A76) | 16,384 tokens KV-cache | 2.2 tok/s real [verified] | [06](06-cpu-baseline.md) |
| VLM vision offload | MobileCLIP-S0 | NPU | 1×256×256 image | 22.6 ms/frame [verified] | [05](05-run-vlm-npu.md) |
| Hybrid assistant (recommended) | MobileCLIP (NPU) + Qwen Q8_0 (CPU) | NPU+CPU | Vision: NPU, Text: 8k | 22.6 ms/vision + 18 tok/s [verified] | [05](05-run-vlm-npu.md) + [06](06-cpu-baseline.md) |
| NPU-only with more context | SmolLM2-135M W=64 int16 | NPU | 64 tokens fixed | 14 tok/s [verified] | [03](03-run-llm-npu.md) |

## How to choose

### Use NPU when:
- ROS2 or other real-time workloads need the A76 CPU cores
- Short, fast responses are acceptable (fixed window 32-64 tokens)
- SmolLM2-class quality is sufficient
- Vision encoding needs low latency (<25 ms)

### Use CPU when:
- ROS2 is paused/frozen and A76 cores are free
- You need real context (>100 tokens, up to thousands)
- You need higher model quality (Qwen > SmolLM2 at same size class)
- Long first-token waits at 16k context are acceptable

### Use hybrid when:
- You need both real-time vision AND real-language context
- Run MobileCLIP on NPU for images
- Run Qwen on CPU for text
- Keep NPU available for robotics during normal operation

## NPU configs that work

| Model | Window | NBG size | RSS peak | Decode | Coherent | Notes |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 32 | 281 MB | 272 MB | 20.7 tok/s | Yes | Best NPU config |
| SmolLM2-135M | 64 | 282 MB | 274 MB | 14.0 tok/s | Yes (weak) | Slightly more context |
| SmolLM2-135M | 128 | 287 MB | 282 MB | 6.0 tok/s | No | Coherence breaks |
| SmolLM2-135M | 256 | 337 MB | 375 MB | 1.2 tok/s | No | Impractical |
| SmolLM2-360M | 32 | 673 MB | 646 MB | 8.4 tok/s | Yes | Smarter, slower |
| SmolLM2-360M | 64 | 675 MB | 649 MB | 4.9 tok/s | Yes | Best 360M config |
| SmolLM2-360M | 128 | 693 MB | 681 MB | 2.0 tok/s | No | Below usable |
| SmolLM2-360M | 256 | 709 MB | 711 MB | 1.2 tok/s | No | Impractical |

## NPU configs that FAIL at export

| Model | Failure |
|---|---|
| SmolLM2-1.7B | `gen_nbg` segfault, 0-byte NBG from 6.85 GB ONNX |
| Qwen2.5-0.5B (all quants) | See [08-known-limits-and-blockers.md](08-known-limits-and-blockers.md) |

## CPU configs that work

| Model | Quant | Context | Prefill tok/s | Decode tok/s | First-token | RSS peak |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Q8_0 | 2,048 | 47.8 | 11.7 (tg64) / 18.4 (real chat) | ~3 s (chat) | 1.2 GiB |
| Qwen2.5-0.5B | Q8_0 | 8,192 | 22.1 | 11.5 (tg64) | ~6 min (est.) | 1.2 GiB |
| Qwen2.5-0.5B | Q8_0 | 16,384 | 13.3 | 12.1 (tg64) / 2.2 (real chat) | ~18 min (chat) | 1.3 GiB |
| Qwen2.5-0.5B | Q4_K_M | 2,048 | 18.0 | 10.9 (tg64) / 19.3 (real chat) | ~3 s (chat) | 734 MiB |
| Qwen2.5-0.5B | Q4_K_M | 8,192 | 12.6 | 10.7 (tg64) | ~11 min (est.) | 737 MiB |
| Qwen2.5-0.5B | Q4_K_M | 16,384 | 9.2 | 11.0 (tg64) | ~30 min (est.) | 838 MiB |

**Note**: Q8_0 is recommended over Q4_K_M for Qwen on this board — it is
both faster and higher quality, despite higher RSS.

## VLM configs

| Component | Input | Latency | RSS | NBG size |
|---|---|---|---|---|
| MobileCLIP-S0 | 1×256×256 | 22.6 ms | 14 MB | 19 MB |
| Tiny VLM bridge | embed + 4 tokens | 0.063 ms | 2 MB | 94 KB |

The tiny bridge is a proof-of-concept only. A real VLM would need a larger
decoder and projector — see [05-run-vlm-npu.md](05-run-vlm-npu.md) for details.

## Commands reference

### NPU: SmolLM2-135M chat
```bash
cd /home/orangepi/a733_npu_driver
python3 scripts/board/chat_shell.py \
  --model models/smollm2_135m_w32_int16/network_binary.nb \
  --tokenizer work/models/smollm2-135m-instruct \
  --runner build/npu_lm_runner \
  --vip-lib /home/orangepi/lib \
  --window 32 --greedy
```

### CPU: Qwen Q8_0 8k context
```bash
taskset -c 6,7 llama-completion \
  -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -c 8192 -t 2 -ngl 0 --no-warmup --temp 0
```

### NPU: MobileCLIP vision
```bash
cd ~/a733_npu_driver
bash scripts/board/run-b3-vpm-package.sh \
  --model-dir models/b3_mobileclip_s0_vision_int16 \
  --log-dir logs/board/b3-mobileclip \
  --vpm-run /opt/vpm_run/vpm_run \
  --vip-lib /home/orangepi/lib
```
