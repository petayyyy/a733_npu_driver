# V4 VLM Chat CLI

Image+text conversation tool for A733 / Orange Pi Zero 3W.

## Quick Start

```bash
# CPU mode (default, fast, recommended)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image."

# NPU-vision-offload mode (frees CPU cores, slower vision)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image." --backend npu

# Use larger 500M model (more detail, slower, ~1.2 GB RAM)
python3 app/vlm_chat.py --image test_images/dog.jpg --model SmolVLM-500M

# CPU-only mode (hides NPU option)
python3 app/vlm_chat.py --cpu-only --image test_images/dog.jpg
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to image file |
| `-p`, `--prompt` | "Describe this image." | Question about the image |
| `--model` | SmolVLM-256M | SmolVLM-256M or SmolVLM-500M |
| `--backend` | cpu | cpu (llama.cpp) or npu (NPU vision + CPU LLM) |
| `--max-tokens` | 128 | Max answer tokens |
| `--temp` | 0.0 | Temperature (0 = deterministic) |
| `--cpu-only` | false | Force CPU, hide NPU option |

## Requirements

- Orange Pi Zero 3W (6 GB RAM) or any A733 board with ≥4 GB RAM
- llama.cpp built with multimodal (mmproj) support
- SmolVLM GGUF + mmproj in `models/vlm/`
- NPU mode: `/dev/vipcore` + VIPLite 2.0.3.2 + `models/smolvlm_256m_vision_v2d_int16/`
