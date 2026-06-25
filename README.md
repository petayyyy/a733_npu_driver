# A733 NPU Driver — LLM/VLM on Vivante VIP9000

**Running real LLMs and a VLM vision pipeline NPU-only on the Allwinner A733
(Vivante VIP9000, ~3 TOPS).** Validated on Radxa Cubie A7Z (development) and
Orange Pi Zero 3W (final target). This is a working proof-of-concept with a
full reproducible toolchain — not a polished product.

## What works / what doesn't

| Status | What |
|---|---|
| Works | SmolLM2-135M int16 NPU chat at 21 tok/s (W=32, coherent) |
| Works | SmolLM2-360M int16 NPU chat at 8 tok/s (W=32, coherent) |
| Works | MobileCLIP-S0 vision encoder on NPU (22.6 ms/frame) |
| Works | Qwen2.5-0.5B CPU chat via llama.cpp with real KV-cache (~18 tok/s) |
| Export FAILS | SmolLM2-1.7B (gen_nbg segfault, 6.85 GB ONNX) |
| Export FAILS | Qwen2.5-0.5B on NPU (BF16 blocked by vnn_VerifyGraph -3) |
| N/A | No KV-cache (fixed-window static NBG only) |
| N/A | No working int4/int8 LLM (pcq quality fails) |

Full numbers and analysis: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Hardware

| | Radxa Cubie A7Z (dev) | Orange Pi Zero 3W (final) |
|---|---|---|
| SoC | Allwinner A733 | Allwinner A733 |
| NPU | Vivante VIP9000, cid `0x1000003b`, 1 core | Vivante VIP9000, cid `0x1000003b`, 1 core |
| Kernel | 5.15.147-21-a733 | 6.6.98-sun60iw2 |
| RAM | ~1 GiB | 5.7 GiB |
| OS | Debian 11 | Orange Pi Debian (bookworm) |
| VIPLite | 2.0.3.2-AW-2024-08-30 | 2.0.3.2-AW-2024-08-30 |
| `/dev/vipcore` | major:minor 199,0 | 199,0 |

NBG files are binary-compatible across both boards (same silicon, same
VIPLite `cid`). The persistent runner must be rebuilt against each board's
glibc-matched VIPLite `.so`.

## Start here

1. **Set up the host** — [docs/01-setup-host.md](docs/01-setup-host.md)
2. **Bring up the board NPU** — [docs/02-board-bringup.md](docs/02-board-bringup.md)
3. **Pick a configuration** — [docs/configurations.md](docs/configurations.md)

Then follow the run guide for your use case:

| Goal | Guide |
|---|---|
| Chat with SmolLM2 on NPU | [docs/03-run-llm-npu.md](docs/03-run-llm-npu.md) |
| Interactive chat shell on board | [docs/04-chat-shell.md](docs/04-chat-shell.md) |
| Run MobileCLIP vision encoder + VLM bridge on NPU | [docs/05-run-vlm-npu.md](docs/05-run-vlm-npu.md) |
| Run Qwen on CPU with real KV-cache | [docs/06-cpu-baseline.md](docs/06-cpu-baseline.md) |
| Port from Radxa to Orange Pi | [docs/07-porting-radxa-to-orangepi.md](docs/07-porting-radxa-to-orangepi.md) |
| Understand limits and blockers | [docs/08-known-limits-and-blockers.md](docs/08-known-limits-and-blockers.md) |

## Repository layout

```text
docs/               — Task-oriented guides and reference docs
scripts/
  host/             — x86 host tools (ONNX gen, ACUITY convert, compare, SSH)
  board/            — Board scripts (smoke tests, runner build, chat shell, benchmarks)
reports/            — Chronological research log (gates, tasks, benchmarks)
logs/               — Ignored; board and host run logs
work/               — Ignored; generated ONNX, NBG packages, model files, SDK checkout
```

## Dependencies

- **Host**: Docker with `ubuntu-npu:v2.0.10.1` (ACUITY 6.30.22), Python 3
- **Board**: VIPLite 2.0.3.2 userspace libraries, C compiler, Python 3
- **Models** (downloaded by scripts): SmolLM2-135M/360M from Hugging Face, Qwen2.5-0.5B GGUF
- **SDK**: [ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk) for VIPLite headers/libs

## Quick host setup

```powershell
# From PowerShell on the x86 host
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1
```

```bash
# Convert an ONNX model to A733 NBG (inside Docker)
scripts/host/convert_onnx_to_nbg.sh \
  --name my_model \
  --onnx path/to/model.onnx \
  --dataset path/to/dataset.txt \
  --quant int16 \
  --inputs token_ids \
  --input-size-list 32 \
  --outputs logits
```

## Quick board instructions

See [docs/02-board-bringup.md](docs/02-board-bringup.md) for full details.

```bash
# Build the persistent runner on the board
cd ~/a733_npu_driver
bash scripts/board/build-npu-lm-runner.sh

# Run the chat shell
python3 scripts/board/chat_shell.py \
  --model models/smollm2_135m_w32_int16/network_binary.nb \
  --tokenizer work/models/smollm2-135m-instruct \
  --runner build/npu_lm_runner \
  --vip-lib /home/orangepi/lib \
  --window 32
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the report format convention
(verified/assumption tags, blocker format) and how to add new findings.

## Acknowledgements

- **Radxa** — A733 NPU documentation, Cubie A7Z images, and the VIPLite 2.0 SDK
- **VeriSilicon** — TIM-VX/ACUITY toolchain (`ubuntu-npu` Docker images)
- **[ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk)** — A733-compatible VIPLite SDK examples and build scripts
- **[Rabs9/A733-kernel](https://github.com/Rabs9/a733-kernel)** — A733 kernel with device-tree NPU (`sunxi_npu`) support
- **Rockchip RKLLM** — Reference architecture for LLM with KV-cache on edge NPUs
- **ACUITY dimension limit** — The 65536 per-dimension limit identified in T11 is documented in [docs/08-known-limits-and-blockers.md](docs/08-known-limits-and-blockers.md)

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
