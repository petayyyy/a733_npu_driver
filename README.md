# A733 NPU Driver — LLM/VLM on Vivante VIP9000

**This NPU is a vision/CNN accelerator that also runs tiny LLMs. It is NOT an
LLM accelerator for Qwen-class models. The productive path is hybrid: NPU for
vision + small LLMs, CPU (llama.cpp) for Qwen-class models with real context.**

## Project Summary

We attempted to run real LLM/VLM inference NPU-only on the Allwinner A733
(Vivante VIP9000, ~3 TOPS). We succeeded at proof-of-concept scale: SmolLM2-135M
and 360M run coherently on the NPU at usable speeds (21 and 8 tok/s), a
MobileCLIP-S0 vision encoder works on the NPU (22.6 ms/frame), and we built a
fully reproducible toolchain (ONNX → ACUITY → NBG → VIPLite → board) with
interactive chat support.

**What's blocked is Qwen2.5-0.5B and SmolLM2-1.7B on the NPU** — every
configuration was tried and verified to fail. int16 collapses on Qwen's
activation outliers; BF16 fixes quality but won't compile (`vnn_VerifyGraph -3`);
INT8 quality is 0.079 cosine; per-block NBG chaining degrades to garbage at full
depth. These are genuine toolchain/vendor limits, not missing experiments.

**The way forward is hybrid**: NPU for vision encoding (MobileCLIP-S0,
22.6 ms/frame) and tiny LLM chat (SmolLM2-135M, 21 tok/s), CPU for smarter
models (Qwen2.5-0.5B Q8_0, 18 tok/s on 2 A76 cores leaving 6 A55 cores free
for ROS2) and low-end VLM (SmolVLM-256M Q8_0, 52.6 tok/s, 634 MB RSS).

## What works / what doesn't

| Status | What |
|---|---|
| Works (NPU) | SmolLM2-135M int16 W=32, 20.7 tok/s, coherent |
| Works (NPU) | SmolLM2-360M int16 W=32, 8.4 tok/s, coherent |
| Works (NPU) | MobileCLIP-S0 vision encoder, 22.6 ms, cosine 0.99996 |
| Works (CPU) | Qwen2.5-0.5B Q8_0, 18.0 tok/s, 2×A76 = 25% CPU, 6 cores free |
| Works (CPU) | SmolVLM-256M-Instruct Q8_0 image chat, 52.6 tok/s, 634 MB RSS |
| Works (CPU) | SmolVLM-500M-Instruct Q8_0 image chat, 22.3 tok/s |
| Works (CPU,batch) | Qwen2.5-3B Q4_K_M, 4.0 tok/s — not for interactive chat, batch/offline only |
| Export FAILS | Qwen2.5-0.5B on NPU — int16 cosine 0.236; FP16 0.541; BF16 vnn_VerifyGraph -3; W8A16 0.079; block-chained degenerates at 6.6 tok/s |
| Export FAILS | SmolLM2-1.7B on NPU — gen_nbg segfault, 0-byte NBG |
| **Works (hybrid)** | **SmolVLM SigLIP on NPU → CPU LLM** — Conv→MatMul rewrite, NBG exports, e2e accurate on 3 test images (V2d) |
| N/A | No KV-cache (static-shape NBG); short fixed window W≤64 only |
| N/A | No working LLM int8/int4 (pcq quality fails, per-channel int16 not in ACUITY) |

Full numbers and analysis: **[docs/RESULTS.md](docs/RESULTS.md)**.
All blockers: **[docs/blockers.md](docs/blockers.md)**.
Vendor tickets: **[docs/vendor-tickets.md](docs/vendor-tickets.md)**.

## Hardware

| | Radxa Cubie A7Z (dev) | Orange Pi Zero 3W (final) |
|---|---|---|
| SoC | Allwinner A733 | Allwinner A733 |
| NPU | Vivante VIP9000, cid `0x1000003b`, 1 core, ~1.0 GHz | same |
| TOPS | ~3 TOPS INT8 | same |
| Kernel | 5.15.147-21-a733 | 6.6.98-sun60iw2 |
| RAM | ~1 GiB | 6 GB |
| CPU | 8 cores | 2×A76 + 6×A55 |
| OS | Debian 11 | Orange Pi Debian (bookworm) |
| VIPLite | 2.0.3.2-AW-2024-08-30 | 2.0.3.2-AW-2024-08-30 |
| `/dev/vipcore` | 199,0 | 199,0 |

## Continue this project / hand-off

> **[docs/import_chat.md](docs/import_chat.md)** — Import this into a new
> Claude / Claude Code session to continue the project with full context.
> Self-contained knowledge base with all verified numbers, blockers, and
> building blocks.

## Start here

1. **Set up the host** — [docs/01-setup-host.md](docs/01-setup-host.md)
2. **Bring up the board NPU** — [docs/02-board-bringup.md](docs/02-board-bringup.md)
3. **Pick a configuration** — [docs/configurations.md](docs/configurations.md)
4. **Run CLI chat tools** — [docs/09-cli-tools.md](docs/09-cli-tools.md)
5. **Read the honest limits** — [docs/08-known-limits-and-blockers.md](docs/08-known-limits-and-blockers.md)
6. **File vendor tickets** — [docs/vendor-tickets.md](docs/vendor-tickets.md)

Then follow the run guide for your use case:

| Goal | Guide |
|---|---|
| **VLM image chat (CLI)** | [app/README.md](app/README.md) — `python3 app/vlm_chat.py --image dog.jpg` |
| **LLM text chat (CLI)** | [app/README.md](app/README.md) — `python3 app/llm_chat.py` |
| Install everything from scratch | [docs/09-cli-tools.md](docs/09-cli-tools.md) |
| Chat with SmolLM2 on NPU | [docs/03-run-llm-npu.md](docs/03-run-llm-npu.md) |
| Interactive chat shell on board | [docs/04-chat-shell.md](docs/04-chat-shell.md) |
| Run MobileCLIP vision encoder on NPU | [docs/05-run-vlm-npu.md](docs/05-run-vlm-npu.md) |
| Run Qwen on CPU with real KV-cache | [docs/06-cpu-baseline.md](docs/06-cpu-baseline.md) |
| Run SmolVLM image chat on CPU | [docs/06-cpu-baseline.md](docs/06-cpu-baseline.md#smolvlm-image-chat-on-cpu) |
| Port from Radxa to Orange Pi | [docs/07-porting-radxa-to-orangepi.md](docs/07-porting-radxa-to-orangepi.md) |
| Understand limits and blockers | [docs/08-known-limits-and-blockers.md](docs/08-known-limits-and-blockers.md) |

## Repository layout

```text
app/                — CLI chat tools (VLM image chat, LLM text chat)
docs/               — Task-oriented guides, reference, configurations, blocker list
scripts/
  host/             — x86 tools (ONNX gen, ACUITY convert, compare, SSH)
  board/            — Board scripts (smoke tests, runner build, chat shell, benchmarks)
reports/            — Chronological research log (gates, tasks, benchmarks)
logs/               — Ignored; board and host run logs
work/               — Ignored; generated ONNX, NBG packages, model files, SDK checkout
```

## Dependencies

- **Host**: Docker with `ubuntu-npu:v2.0.10.1` (ACUITY 6.30.22), Python 3
- **Board**: VIPLite 2.0.3.2 userspace libs, C compiler, Python 3, llama.cpp (for CPU path)
- **Models**: SmolLM2 from Hugging Face, Qwen2.5 GGUF, SmolVLM GGUF
- **SDK**: [ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk) — VIPLite headers/libs

## Quick host setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\host\prepare-workspace.ps1
```

```bash
# Convert ONNX to A733 NBG
DOCKER_RUN_ARGS="--cpus 10 --memory 24g" \
  scripts/host/convert_onnx_to_nbg.sh \
    --name smollm2_135m_w32 \
    --onnx work/generated/smollm2_135m_w32/real_llm.onnx \
    --dataset work/generated/smollm2_135m_w32/dataset.txt \
    --quant int16 \
    --inputs token_ids \
    --input-size-list 32 \
    --outputs logits
```

## Quick board: VLM image chat

```bash
cd ~/a733_npu_driver

# CPU mode (default, recommended)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "Describe this image."

# NPU-vision-offload mode (frees CPU cores)
python3 app/vlm_chat.py --image test_images/dog.jpg -q "What animal?" --backend npu
```

## Quick board: LLM text chat

```bash
cd ~/a733_npu_driver

# Interactive REPL (qwen-1.5b default)
python3 app/llm_chat.py

# One-shot
python3 app/llm_chat.py -q "Explain quantum computing."
```

## Project Conclusions

1. **The A733 VIP9000 is a capable vision/CNN accelerator.** MobileCLIP-S0 runs
   at 22.6 ms/frame with pixel-perfect accuracy (cosine 0.99996 vs host). This is
   the NPU's strength — and the foundation for hybrid vision+LLM systems.

2. **It can run tiny LLMs NPU-only.** SmolLM2-135M at 21 tok/s is genuinely
   useful for short chat responses, and keeps the CPU cores free for other work.
   SmolLM2-360M at 8 tok/s is the smartest model that runs entirely on the NPU.

3. **It cannot run Qwen-class LLMs NPU-only.** Qwen2.5-0.5B's activation outliers
   defeat every quantization path: int16 DFP (cosine 0.236), FP16 (0.541), W8A16
   (0.079), per-channel int16 (chip-gated, pegasus rejects it). BF16 is the only
   format that preserves quality (cosine 0.991) but won't export (vnn_VerifyGraph
   -3 / 64768). Per-block NBG chaining compiles and runs mechanically (6.6 tok/s)
   but produces garbage due to int16 depth collapse. This is a genuine toolchain
   limit requiring vendor support.

4. **The A733 is not an RK3588 competitor for LLM/VLM.** The RK3588 (6 TOPS,
   RKLLM with KV-cache + int4/int8) runs full LLMs/VLMs natively. The A733
   (3 TOPS, no KV-cache, no working LLM int8 BF16 export) cannot match this.

5. **The hybrid NPU-vision + CPU-LLM path is the recommended architecture.**
   Offload vision to the NPU (MobileCLIP-S0), run the language model on CPU
   (Qwen Q8_0 at 18 tok/s on 2 A76 cores, leaving 6 A55 cores free for ROS2),
   and optionally keep SmolLM2-135M on NPU for fast short responses. This is
   the path for a real local assistant on A733 hardware.

## Licence

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- **Radxa** — A733 NPU docs, Cubie A7Z images, VIPLite 2.0 SDK
- **VeriSilicon** — TIM-VX/ACUITY toolchain (`ubuntu-npu` Docker images)
- **[ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk)** — A733 VIPLite SDK examples
- **[Rabs9/A733-kernel](https://github.com/Rabs9/a733-kernel)** — A733 kernel with NPU devicetree support
- **Rockchip RKLLM** — Reference architecture for LLM with KV-cache on edge NPUs
- **ACUITY dimension limit (65536)** — documented from ST Edge AI forum and vLLM-Ascend
