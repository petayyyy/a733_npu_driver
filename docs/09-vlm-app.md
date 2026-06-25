# VLM Image Chat — Install & Run Guide

Step-by-step from a fresh Orange Pi Zero 3W to running image+text chat.

## Hardware Requirements

| Model | RAM needed | Speed | Notes |
|---|---|---|---|
| SmolVLM-256M (default) | ~634 MB RSS | ~52 tok/s | Leaves >5 GB for ROS2 |
| SmolVLM-500M | ~1.2 GB RSS | ~22 tok/s | More detail, slower |
| >1B models | >2 GB | <5 tok/s | Not supported on 6 GB |

Orange Pi Zero 3W: 2xA76 + 6xA55, 6 GB RAM. SmolVLM-256M leaves >5 GB
free for ROS2/picoclaw. CPU mode requires only aarch64 + enough RAM —
works on any A733 board. NPU mode requires `/dev/vipcore` + VIPLite 2.0.3.2.

## Step 1: System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip cmake build-essential git
pip3 install Pillow --break-system-packages
```

## Step 2: Build llama.cpp (with multimodal support)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout be4a6a6   # proven commit for this board
cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2a -DGGML_OPENMP=OFF
cmake --build build --target llama-cli -- -j4
```

> If you need the NPU-offload mode, first bring up the NPU
> ([docs/02-board-bringup.md](02-board-bringup.md)), then apply the V2c mtmd
> patch before building llama.cpp. The patch is pre-applied on the reference board.

## Step 3: Download Models

```bash
mkdir -p ~/a733_npu_driver/models/vlm

# SmolVLM-256M (default, recommended)
wget -P ~/a733_npu_driver/models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf
wget -P ~/a733_npu_driver/models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf

# SmolVLM-500M (optional, more detail)
wget -P ~/a733_npu_driver/models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf
wget -P ~/a733_npu_driver/models/vlm \
  https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

## Step 4: Run the App

```bash
cd ~/a733_npu_driver

# CPU mode (default)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image."

# NPU-vision-offload mode (requires NPU setup from docs/02-board-bringup.md)
python3 app/vlm_chat.py --image test_images/dog.jpg -p "Describe this image." --backend npu

# 500M model
python3 app/vlm_chat.py --image test_images/cat.jpg --model SmolVLM-500M
```

## Sample Output

```
V4 VLM Chat -- SmolVLM-256M | ~52 tok/s | ~634 MB
Image: dog.jpg | Prompt: Describe this image.
Backend: CPU

In the foreground of the picture there is a white dog sitting on the grass.
Behind the dog there are plants.

-- SmolVLM-256M CPU | wall 8.6s | prompt 12 t/s | gen 52 t/s | RAM 251/5853 MB used (5602 MB free) | 2xA76 used, ROS2 safe
```

## Troubleshooting

**"Model not found"** — verify GGUF/mmproj files are in `models/vlm/`. Run Step 3.

**OOM / killed** — the 500M model needs ~1.2 GB. Check `free -h` before running.
Close other apps. The 256M model only needs ~634 MB and should always fit.

**"PIL not installed"** — `pip3 install Pillow --break-system-packages`. NPU mode
needs PIL for image preprocessing; CPU mode doesn't.

**NPU mode: "vpm_run not found"** — NPU requires VIPLite setup and NBG package.
Follow [docs/02-board-bringup.md](02-board-bringup.md) for `/dev/vipcore` bring-up,
then [docs/05-run-vlm-npu.md](05-run-vlm-npu.md) for the SmolVLM vision NBG.
Use `--cpu-only` as fallback.

**NPU mode: "NBG not found"** — the SmolVLM vision encoder NBG package is
missing. Requires ACUITY host build (not covered here). Use `--cpu-only`.

**Port conflict (if web UI added later)** — default port not used in CLI mode.

**Slow answers** — first run loads the model (~5-10s). Subsequent runs reuse
the model from disk cache. NPU mode adds ~6s for vision encoding.

## For Other A733 Boards

The CPU path works on any A733/aarch64 board with ≥4 GB RAM — just build
llama.cpp (Step 2) and download models (Step 3). The NPU-offload path needs
`/dev/vipcore` present + glibc-matched VIPLite .so files. See
[docs/07-porting-radxa-to-orangepi.md](07-porting-radxa-to-orangepi.md) for
porting guidance.
