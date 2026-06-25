# Configurations

Pick your goal, follow the recommended config. All numbers verified on Orange Pi
Zero 3W (6 GB RAM).

## Recommended production path: Hybrid

**NPU for vision + small LLMs, CPU for Qwen-class models with real context.**

- MobileCLIP-S0 vision encoder → NPU (22.6 ms/frame)
- SmolLM2-135M short chat → NPU (21 tok/s, CPU cores free)
- Qwen2.5-0.5B chat with real context → CPU (18 tok/s on 2×A76, 6×A55 free)
- SmolVLM-256M image chat → CPU (53 tok/s, 634 MB RSS)
- **SmolVLM-256M hybrid** → NPU vision (5.94s, 0 CPU) + CPU LLM (46.5 tok/s, 2×A76), proven V2d

## Quick reference

| Goal | Model | Backend | Context | Tok/s | Guide |
|---|---|---|---|---|---|
| Fast NPU chat (ROS2 safe) | SmolLM2-135M W=32 int16 | NPU | 32 fixed | 21 | [03](03-run-llm-npu.md) |
| Smarter NPU chat (ROS2 safe) | SmolLM2-360M W=32 int16 | NPU | 32 fixed | 8 | [03](03-run-llm-npu.md) |
| Chat with real context | Qwen2.5-0.5B Q8_0 | CPU (2×A76) | 8k KV-cache | 18 decode | [06](06-cpu-baseline.md) |
| Long-context retrieval | Qwen2.5-0.5B Q8_0 | CPU (2×A76) | 16k KV-cache | 2.2 decode | [06](06-cpu-baseline.md) |
| **Image chat (CLI app)** | **SmolVLM-256M Q8_0** | CPU or NPU+CPU | auto | **52** | [09](09-cli-tools.md) |
| **Text chat (CLI app)** | **Qwen2.5-1.5B Q4_K_M** | CPU (2xA76) | 8k | **8.5** | [09](09-cli-tools.md) |
| Image chat (manual) | SmolVLM-256M Q8_0 | CPU (2×A76) | auto | 53 | [06](06-cpu-baseline.md#smolvlm-image-chat-on-cpu) |
| Higher-detail image chat | SmolVLM-500M Q8_0 | CPU (2×A76) | auto | 22 | [06](06-cpu-baseline.md#smolvlm-image-chat-on-cpu) |
| **Image chat + NPU vision (recommended)** | **SmolVLM-256M hybrid** | NPU vision + CPU LLM | auto | **46.5 tok/s** | [09](09-cli-tools.md) |
| VLM vision offload (SmolVLM) | SmolVLM SigLIP int16 | NPU only | 1×512×512 | 5,959 ms | [05](05-run-vlm-npu.md) |
| VLM vision offload (MobileCLIP) | MobileCLIP-S0 | NPU only | 1×256×256 | 22.6 ms | [05](05-run-vlm-npu.md) |
| NPU-only more context | SmolLM2-135M W=64 int16 | NPU | 64 fixed | 14 | [03](03-run-llm-npu.md) |

## NPU configs (all verified)

| Model | W | NBG size | RSS peak | Decode tok/s | Coherent |
|---|---|---|---|---|---|
| SmolLM2-135M | 32 | 281 MB | 272 MB | 20.7 | Yes |
| SmolLM2-135M | 64 | 282 MB | 274 MB | 14.0 | Weak |
| SmolLM2-360M | 32 | 673 MB | 646 MB | 8.4 | Yes |
| SmolLM2-360M | 64 | 675 MB | 649 MB | 4.9 | Yes |
| MobileCLIP-S0 | — | 19 MB | 14 MB | 22.6 ms | 0.99996 cosine |

W≥128 exports but is incoherent for both models. See coherence cliff.

## CPU configs (all verified, 2×A76 threads=2)

| Model | Quant | Context | Decode tok/s | RSS peak | Notes |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Q8_0 | 2,048 | 18.0 | 1,109 MB | Best decode; 6 A55 free |
| Qwen2.5-0.5B | Q8_0 | 8,192 | 11.5 (tg64) | 1,201 MB | Good default |
| Qwen2.5-0.5B | Q8_0 | 16,384 | 2.2 (real chat) | 1,306 MB | ~18 min first token |
| Qwen2.5-0.5B | Q4_K_M | 2,048 | 10.9 | 734 MB | Slower than Q8_0 |
| SmolVLM-256M | Q8_0 | auto | 52.6 | 634 MB | Recommended VLM |
| SmolVLM-500M | Q8_0 | auto | 22.3 | ~1.2 GB | More detail, slower |
| Qwen2.5-3B | Q4_K_M | 2,048 | 4.0 | 3,857 MB | **Experimental, batch-only** — not for interactive chat |

**Q8_0 beats Q4_K_M** on this board — always use Q8_0 for Qwen.
> **Qwen2.5-3B loads but is not suitable for interactive chat (4 tok/s).**
> Available as `--model qwen-3b` in `app/llm_chat.py` for experiments,
> but recommended only for batch/offline use.

## NPU configs that FAIL

| Model | Failure | Report |
|---|---|---|
| SmolLM2-1.7B | gen_nbg segfault, 0-byte NBG | [b1b](../reports/b1b-benchmark-matrix.md) |
| Qwen2.5-0.5B int16 monolithic | Cosine 0.236 | [t8](../reports/t8-qwen-int16-port.md) |
| Qwen2.5-0.5B BF16 | vnn_VerifyGraph -3 / 64768 | [t9](../reports/t9-qwen-bf16.md) |
| Qwen2.5-0.5B W8A16 | Cosine 0.079 | [q1](../reports/q1-qwen-int8-gates.md) |
| Qwen2.5-0.5B block-chain 26 NBG | Degenerate at 6.6 tok/s | [q2](../reports/q2-qwen-block-nbg.md) |
| ~~SmolVLM SigLIP on NPU~~ | Resolved: Conv→MatMul rewrite + NPU→llama.cpp injection (V2d) | [v2d](../reports/v2d-vlm-npu-mmproj-glue.md) |

## Coherence cliff

NPU fixed-window coherence: W=32 yes, W=64 weak, W≥128 no.
No KV-cache → full recompute per token → O(W²) throughput.

## Commands

### VLM: Image chat (CLI tool)
```bash
cd ~/a733_npu_driver

# CPU mode (default, recommended)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "Describe this image."

# NPU-vision-offload mode
python3 app/vlm_chat.py --image test_images/dog.jpg -q "What animal?" --backend npu

# Interactive REPL
python3 app/vlm_chat.py --image test_images/dog.jpg
```

### LLM: Text chat (CLI tool)
```bash
cd ~/a733_npu_driver

# Interactive REPL (qwen-1.5b default)
python3 app/llm_chat.py

# One-shot
python3 app/llm_chat.py -q "Explain quantum computing."

# Faster model
python3 app/llm_chat.py --model qwen-0.5b

# Experimental 3B (batch-only, ~4 tok/s)
python3 app/llm_chat.py --model qwen-3b
```

### Hybrid: SmolVLM with NPU vision offload
```bash
cd ~/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16
# 1. Run NPU vision on pre-made input
printf '[network]\n./network_binary.nb\n[input]\n./dog_input.dat\n' > s.txt
LD_LIBRARY_PATH=/home/orangepi/lib /opt/vpm_run/vpm_run -s s.txt -b 0 --save_txt 1
# 2. Convert to binary embeddings
python3 -c "import struct;f=open('output_0.txt');v=[float(l.strip())for l in f if l.strip()];f.close();open('/tmp/e.bin','wb').write(b''.join(struct.pack('<f',x)for x in v))"
# 3. Run LLM with NPU embeddings
A733_NPU_EMBEDDINGS=/tmp/e.bin LD_LIBRARY_PATH=/home/orangepi/llama.cpp/build/bin \
printf '/exit\n' | ~/llama.cpp/build/bin/llama-cli \
  -m ~/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj ~/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --image ~/a733_npu_driver/test_images/dog.jpg \
  -p '<image>What animal is in this image?' -n 80 -t 2 --temp 0.0 --simple-io --no-perf --log-disable
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
