# CLI Chat Tools

Terminal-only tools for VLM image chat and LLM text chat on Orange Pi Zero 3W.
No web server, no browser — just SSH in and run.

## VLM — Image Chat

```bash
# Interactive REPL (image stays loaded, ask multiple questions)
python3 app/vlm_chat.py --image test_images/dog.jpg

# One-shot question
python3 app/vlm_chat.py --image test_images/dog.jpg -q "Describe this image."

# NPU vision offload (frees CPU for ROS2, ~6s vision encode on NPU)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "What animal?" --backend npu

# Larger 500M model for more detail
python3 app/vlm_chat.py --image test_images/dog.jpg --model smolvlm-500m

# Force CPU even if NPU is present
python3 app/vlm_chat.py --image test_images/dog.jpg --cpu-only
```

## LLM — Text Chat

```bash
# Interactive REPL (qwen-1.5b default, balanced speed/quality)
python3 app/llm_chat.py

# One-shot question
python3 app/llm_chat.py -q "Explain quantum computing."

# Faster model (~18 tok/s, less RAM)
python3 app/llm_chat.py --model qwen-0.5b

# Experimental 3B model (~4 tok/s, needs ~3.7 GB RAM)
python3 app/llm_chat.py --model qwen-3b

# Override CPU cores (default: A76 cores 6-7)
python3 app/llm_chat.py --cores 0-3
```

## Options (VLM)

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to image file |
| `-q`, `--question` | (none = interactive) | One-shot question |
| `--model` | smolvlm-256m | smolvlm-256m or smolvlm-500m |
| `--backend` | cpu | cpu or npu |
| `--max-tokens` | 256 | Max answer tokens |
| `--temp` | 0.0 | Temperature (0 = deterministic) |
| `--cpu-only` | false | Force CPU mode |

## Options (LLM)

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--question` | (none = interactive) | One-shot question |
| `--model` | qwen-1.5b | qwen-0.5b, qwen-1.5b, or qwen-3b |
| `--max-tokens` | 256 | Max answer tokens |
| `--temp` | 0.7 | Temperature |
| `--cores` | 6-7 | CPU core range (A76 only by default) |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/exit` | Quit the chat |
| `/reset` | Clear conversation history |
| `/image <path>` | Switch to a different image (VLM only) |

## Hardware Requirements

| Model | RAM needed | Speed | Notes |
|-------|-----------|-------|-------|
| SmolVLM-256M (VLM) | ~634 MB | ~52 tok/s | Default, leaves >5 GB for ROS2 |
| SmolVLM-500M (VLM) | ~1.2 GB | ~22 tok/s | More detail, slower |
| Qwen2.5-0.5B Q8_0 (LLM) | ~1.1 GB | ~18 tok/s | Fast, low RAM |
| Qwen2.5-1.5B Q4_K_M (LLM) | ~2.0 GB | ~8.5 tok/s | Default, balanced |
| Qwen2.5-3B Q4_K_M (LLM) | ~3.7 GB | ~4 tok/s | Experimental |

Orange Pi Zero 3W: 2xA76 + 6xA55, 6 GB RAM. CPU mode requires aarch64 + enough RAM.
NPU mode requires `/dev/vipcore` + VIPLite 2.0.3.2 + NBG package.

## Requirements

- Orange Pi Zero 3W (6 GB RAM) or any A733 board with >=4 GB RAM
- llama.cpp built with multimodal (mmproj) support (for VLM)
- Models in `models/` directory (see [docs/09-cli-tools.md](docs/09-cli-tools.md))
- NPU mode: `/dev/vipcore` + VIPLite 2.0.3.2-AW-2024-08-30 + NBG package + Pillow
