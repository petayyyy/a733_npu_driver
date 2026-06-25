# CLI Tools — Install & Usage Guide

Step-by-step from a fresh Orange Pi Zero 3W to running VLM image chat and LLM text chat
in the terminal. No web server, no browser — pure CLI.

## Hardware Requirements

| Model | RAM needed | Speed | Notes |
|---|---|---|---|
| SmolVLM-256M (VLM) | ~634 MB RSS | ~52 tok/s | Default; leaves >5 GB for ROS2/picoclaw |
| SmolVLM-500M (VLM) | ~1.2 GB RSS | ~22 tok/s | More detail, 2.4x slower |
| Qwen2.5-0.5B Q8_0 (LLM) | ~1.1 GB RSS | ~18 tok/s | Fast, low RAM |
| Qwen2.5-1.5B Q4_K_M (LLM) | ~2.0 GB RSS | ~8.5 tok/s | **Recommended default** |
| Qwen2.5-3B Q4_K_M (LLM) | ~3.7 GB RSS | ~4 tok/s | Experimental, tight on 6 GB |

**What NOT to run:** >1B VLMs (InternVL3-1B loads but starves RAM at <5 tok/s),
>3B LLMs on this 6 GB board. Use CPU path (any aarch64 + enough RAM); NPU path
needs `/dev/vipcore` + VIPLite 2.0.3.2.

## Step 1: System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip cmake build-essential git wget
pip3 install Pillow --break-system-packages   # needed for NPU vision mode only
```

## Step 2: Build llama.cpp (with multimodal support)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout be4a6a6   # proven commit for Orange Pi Zero 3W
cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2a -DGGML_OPENMP=OFF
cmake --build build --target llama-cli -- -j4
```

> Tip: If you already have llama.cpp built, just verify `llama-cli` accepts
> `--mmproj` and `--chat-template chatml`. The commit `be4a6a6` is validated.

## Step 3: Clone the Repo & Download Models

```bash
cd ~
git clone https://github.com/your-org/a733_npu_driver   # or copy from USB/network
cd a733_npu_driver
```

### VLM models (SmolVLM)

```bash
mkdir -p models/vlm

# SmolVLM-256M (default, recommended)
wget -P models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf
wget -P models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf

# SmolVLM-500M (optional, more detail)
wget -P models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf
wget -P models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

### LLM models (Qwen2.5)

```bash
# Qwen2.5-0.5B Q8_0 (~18 tok/s, ~1.1 GB)
wget -P models \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Qwen2.5-1.5B Q4_K_M (~8.5 tok/s, ~2.0 GB) — RECOMMENDED
wget -P models \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Qwen2.5-3B Q4_K_M (experimental, ~4 tok/s, ~3.7 GB)
wget -P models \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

## Step 4: Run the Tools

### VLM Image Chat

```bash
cd ~/a733_npu_driver

# Interactive REPL — image stays loaded
python3 app/vlm_chat.py --image test_images/dog.jpg

# One-shot (exits after answer)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "Describe this image."

# Use NPU vision offload (requires NPU setup from docs/02-board-bringup.md)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "What animal?" --backend npu

# Larger 500M model
python3 app/vlm_chat.py --image test_images/cat.jpg --model smolvlm-500m
```

### LLM Text Chat

```bash
cd ~/a733_npu_driver

# Interactive REPL (qwen-1.5b default)
python3 app/llm_chat.py

# One-shot
python3 app/llm_chat.py -q "Explain quantum computing in 2 sentences."

# Faster 0.5B model
python3 app/llm_chat.py --model qwen-0.5b

# Experimental 3B model
python3 app/llm_chat.py --model qwen-3b
```

## Sample Sessions

### VLM one-shot (CPU)

```
$ python3 app/vlm_chat.py --image test_images/dog.jpg -q "Describe this image."
VLM Chat -- SmolVLM-256M CPU | ~52 tok/s | ~634 MB
Image: dog.jpg | Q: Describe this image.

A white fluffy dog is sitting on a lush green grassy area. The dog appears to
be a large breed, likely a Husky or similar. The background is blurred but
shows some greenery and what appears to be a fence.

-- SmolVLM-256M CPU | wall 8.6s | prompt 11 t/s | gen 61 t/s | RAM 306/5853 MB used | 2xA76
```

### VLM one-shot (NPU offload)

```
$ python3 app/vlm_chat.py --image test_images/dog.jpg -q "What animal?" --backend npu
VLM Chat -- SmolVLM-256M NPU | ~52 tok/s | ~634 MB
Image: dog.jpg | Q: What animal?

[NPU vision: 5959ms]
A white dog is sitting on the grass.

-- SmolVLM-256M NPU-offload | wall 26.4s (vision 5959ms + LLM 1.7s) | gen 65 t/s | NPU vision, ROS2 safe
```

### LLM one-shot

```
$ python3 app/llm_chat.py -q "What is the capital of France?"
LLM Chat -- Qwen2.5-1.5B Q4_K_M | ~8.5 tok/s | ~2.0 GB
Q: What is the capital of France?

The capital of France is Paris.

-- Qwen2.5-1.5B Q4_K_M | wall 16.1s | prompt 20 t/s | gen 10 t/s | RAM 241/5853 MB used | cores 6-7
```

### VLM interactive

```
$ python3 app/vlm_chat.py --image test_images/dog.jpg
VLM Chat -- SmolVLM-256M CPU | ~52 tok/s | ~634 MB
Image: dog.jpg | Type questions, /exit to quit, /image <path> to change

A white fluffy dog sitting on green grass. There are plants and a wooden fence
in the background.

> What color is the dog?
The dog is white.

> /image test_images/cat.jpg
[Image changed to: cat.jpg]

> What animal is this?
A cat sitting on a stone wall with bare trees in the background.

> /exit
-- SmolVLM-256M CPU | gen 55 t/s | RAM 240/5853 MB used | 2xA76
```

### LLM interactive

```
$ python3 app/llm_chat.py
LLM Chat -- Qwen2.5-1.5B Q4_K_M | ~8.5 tok/s | ~2.0 GB
Cores: 6-7 | Context: 8192 | RAM 241/5853 MB used
Type /exit to quit, /reset to clear history.

[Loading model...]
[Ready]

Hello! I'm a helpful AI assistant running on Orange Pi Zero 3W. How can I help?

> What is Python?
Python is a high-level, interpreted programming language known for its
simplicity and readability...

> /reset
[History cleared]

> Tell me a joke.
Why don't scientists trust atoms? Because they make up everything!

> /exit
-- Qwen2.5-1.5B Q4_K_M | session 62s | 3 turns | gen 10 t/s | cores 6-7
```

## Troubleshooting

### "Model not found" / GGUF missing
Verify GGUF/mmproj files are in the correct directories. Run Step 3 again.

### OOM / process killed
Check `free -h` before running. Close other apps. Pick a smaller model:
- VLM: use smolvlm-256m (634 MB) instead of smolvlm-500m (1.2 GB)
- LLM: use qwen-0.5b (1.1 GB) instead of qwen-1.5b (2.0 GB) or qwen-3b (3.7 GB)

### "PIL not installed" (NPU vision mode)
```bash
pip3 install Pillow --break-system-packages
```
CPU mode doesn't need Pillow.

### NPU mode: "vpm_run not found" or "NBG not found"
NPU requires VIPLite setup and NBG package. Follow:
1. [docs/02-board-bringup.md](02-board-bringup.md) for `/dev/vipcore` bring-up
2. [docs/05-run-vlm-npu.md](05-run-vlm-npu.md) for SmolVLM vision NBG

Use `--cpu-only` or omit `--backend npu` as fallback.

### NPU mode: /dev/vipcore not found
The board's NPU driver is not loaded. Follow [docs/02-board-bringup.md](02-board-bringup.md).
Falls back to CPU automatically.

### Wrong embedding alignment (NPU mode)
If the NPU vision output shape doesn't match what llama.cpp expects, you'll see
garbage answers. This means the NBG was built with a different model version.
Rebuild the NBG following [docs/05-run-vlm-npu.md](05-run-vlm-npu.md).

### Slow answers
- First run loads the model from SD card (~5-15s cold load). Subsequent runs
  reuse the disk cache and are faster.
- NPU mode adds ~6s for vision encoding (but frees both A76 cores).
- If SD card is slow, try a faster card or a USB SSD.

### Disk space (90%+ used)
The models and repo can fill the 32GB SD card. Check `df -h /`:
```bash
# Remove unused models to free space
rm models/qwen2.5-3b-instruct-q4_k_m.gguf   # 2 GB
rm models/vlm/SmolVLM-500M-Instruct-Q8_0.gguf   # 437 MB
rm models/vlm/InternVL3_5-1B-Q4_K_M.gguf   # 484 MB
```

## For Other A733 Boards

The CPU path is board-agnostic — any aarch64 board with ≥4 GB RAM works. Just build
llama.cpp (Step 2) and download models (Step 3).

The NPU path needs:
- `/dev/vipcore` present (A733 SoC with NPU driver loaded)
- VIPLite 2.0.3.2-AW-2024-08-30 `.so` files, glibc-matched to the board's OS
- SmolVLM vision encoder NBG package (host-built via ACUITY Docker)

For porting VIPLite to a different A733 board, see
[docs/07-porting-radxa-to-orangepi.md](07-porting-radxa-to-orangepi.md).
