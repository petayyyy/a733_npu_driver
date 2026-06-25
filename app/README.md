# V4 Chat Apps

CLI tools for image+text (VLM) and text-only (LLM) conversations on A733 / Orange Pi Zero 3W.

## VLM — Image Chat

```bash
# CPU mode (default, fast, recommended, ~52 tok/s)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image."

# NPU-vision-offload mode (frees CPU cores, 6s vision on NPU)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image." --backend npu

# Use larger 500M model
python3 app/vlm_chat.py --image test_images/dog.jpg --model SmolVLM-500M
```

## LLM — Text Chat

```bash
# NPU SmolLM2-135M (fast, 0 CPU, ~21 tok/s)
python3 app/llm_chat.py -p "What is the capital of France?"

# CPU Qwen2.5-0.5B (real KV-cache, ~18 tok/s)
python3 app/llm_chat.py --cpu-only -p "Explain quantum computing."

# NPU SmolLM2-360M (smarter, slower, ~8 tok/s)
python3 app/llm_chat.py --model SmolLM2-360M -p "Tell me a joke."
```

## Options (VLM)

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to image file |
| `-p` | "Describe this image." | Question |
| `--model` | SmolVLM-256M | SmolVLM-256M or SmolVLM-500M |
| `--backend` | cpu | cpu or npu |
| `--max-tokens` | 128 | Max answer tokens |
| `--temp` | 0.0 | Temperature |
| `--cpu-only` | false | Force CPU |

## Options (LLM)

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | (required) | Your question |
| `--model` | SmolLM2-135M | SmolLM2-135M, SmolLM2-360M, or Qwen2.5-0.5B |
| `--backend` | auto | npu or cpu |
| `--max-tokens` | 128 | Max answer tokens |
| `--temp` | 0.0 | Temperature |
| `--cpu-only` | false | Force CPU (Qwen2.5-0.5B) |

## Requirements

- Orange Pi Zero 3W (6 GB RAM) or any A733 board with >=4 GB RAM
- llama.cpp built with multimodal (mmproj) support (for VLM)
- Models in `models/` directory
- NPU mode: `/dev/vipcore` + VIPLite 2.0.3.2
