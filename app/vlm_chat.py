#!/usr/bin/env python3
"""
V4 VLM Chat — CLI tool for image+text conversations on A733 / Orange Pi Zero 3W.

Modes:
  cpu  — SmolVLM on CPU via llama.cpp (fast, recommended, ~52 tok/s)
  npu  — NPU vision encode + CPU LLM decode (frees CPU cores, 5.9s vision)

Usage:
  python3 vlm_chat.py --image dog.jpg -p "Describe this image."
  python3 vlm_chat.py --image dog.jpg -p "What animal?" --backend npu
  python3 vlm_chat.py --image dog.jpg -p "Describe." --model SmolVLM-500M
  python3 vlm_chat.py --cpu-only                          # hide NPU option
"""
import argparse, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

# ── paths (auto‑detected from script location) ──────────────────────────
HOME      = Path.home()
REPO      = HOME / "a733_npu_driver"
VLM_DIR   = REPO / "models/vlm"
NBG_DIR   = REPO / "models/smolvlm_256m_vision_v2d_int16"
IMG_DIR   = REPO / "test_images"
LLAMA_CLI = HOME / "llama.cpp/build/bin/llama-cli"
VIPM_RUN  = Path("/opt/vpm_run/vpm_run")
VIP_LIB   = HOME / "lib"
LLAMA_LIB = HOME / "llama.cpp/build/bin"

MODELS = {
    "SmolVLM-256M": {
        "gguf":   VLM_DIR / "SmolVLM-256M-Instruct-Q8_0.gguf",
        "mmproj": VLM_DIR / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        "speed":  "~52 tok/s",
        "rss":    "~634 MB",
    },
    "SmolVLM-500M": {
        "gguf":   VLM_DIR / "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj": VLM_DIR / "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "speed":  "~22 tok/s",
        "rss":    "~1.2 GB",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════

def _run(cmd: list[str], timeout: int = 600,
         env: dict | None = None, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr)."""
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})}, cwd=cwd,
    )
    return p.returncode, p.stdout, p.stderr


def _stream(cmd: list[str], timeout: int = 600,
            env: dict | None = None, cwd: str | None = None):
    """Run command, yield (stream, line). stream='out'|'err'."""
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={**os.environ, **(env or {})}, cwd=cwd,
        bufsize=1,
    )
    import select
    try:
        while True:
            r, _, _ = select.select([p.stdout, p.stderr], [], [], 0.1)
            for fd in r:
                line = fd.readline()
                if fd is p.stdout:
                    yield ("out", line)
                else:
                    yield ("err", line)
            if p.poll() is not None:
                break
    finally:
        p.stdout.close()
        p.stderr.close()
        p.wait()


def _parse_timing(stderr: str) -> tuple[float | None, float | None]:
    """Extract prompt t/s and generation t/s from llama-cli stderr."""
    import re
    prompt_tps = gen_tps = None
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", stderr)
    if m: prompt_tps = float(m.group(1))
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", stderr)
    if m: gen_tps = float(m.group(1))
    return prompt_tps, gen_tps


def _ram_info() -> str:
    """One‑line RAM usage string."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = int(lines[0].split()[1]) // 1024
        avail = int([l for l in lines if "Available" in l][0].split()[1]) // 1024
        used = total - avail
        return f"RAM {used}/{total} MB used ({avail} MB free)"
    except Exception:
        return "RAM: unknown"


# ═══════════════════════════════════════════════════════════════════════
#  CPU backend (V1‑proven)
# ═══════════════════════════════════════════════════════════════════════

def chat_cpu(image: Path, prompt: str, model_name: str,
             max_tokens: int, temp: float) -> str:
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"Model not found: {cfg['gguf']}")
    if not cfg["mmproj"].exists():
        raise FileNotFoundError(f"Mmproj not found: {cfg['mmproj']}")

    cmd = [
        str(LLAMA_CLI),
        "-m", str(cfg["gguf"]),
        "--mmproj", str(cfg["mmproj"]),
        "--image", str(image),
        "-p", f"<image>{prompt}",
        "-n", str(max_tokens),
        "-t", "2",
        "--temp", str(temp),
        "--simple-io",
        "--no-perf",
        "--log-disable",
    ]

    env = {"LD_LIBRARY_PATH": str(LLAMA_LIB)}

    # pipe /exit so llama‑cli exits after first response
    t0 = time.time()
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env={**os.environ, **env},
    )
    p.stdin.write("/exit\n")
    p.stdin.close()

    answer_lines = []
    stderr_lines = []
    in_answer = False
    for line in p.stdout:
        line = line.rstrip("\n\r")
        stderr_lines.append(line)

        if "<image>" in line:
            in_answer = True
            continue
        if "[ Prompt:" in line or "[ Generation:" in line:
            continue
        if "Exiting" in line:
            continue
        if in_answer and line.strip() and ">" not in line[:2]:
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    t1 = time.time()

    # read stderr
    for line in p.stderr:
        stderr_lines.append(line.rstrip("\n\r"))

    stderr_text = "\n".join(stderr_lines)
    prompt_tps, gen_tps = _parse_timing(stderr_text)

    elapsed = t1 - t0
    print(f"\n-- {model_name} CPU | wall {elapsed:.1f}s", end="")
    if prompt_tps: print(f" | prompt {prompt_tps:.0f} t/s", end="")
    if gen_tps: print(f" | gen {gen_tps:.0f} t/s", end="")
    print(f" | {_ram_info()} | 2xA76 used, ROS2 safe")
    return "\n".join(answer_lines)


# ═══════════════════════════════════════════════════════════════════════
#  NPU‑vision‑offload backend (V2d‑proven)
# ═══════════════════════════════════════════════════════════════════════

def chat_npu(image: Path, prompt: str, model_name: str,
             max_tokens: int, temp: float) -> str:
    cfg = MODELS[model_name]

    # 2.1 — preprocess image to int16 DFP (fl=15) if PIL available
    tag = image.stem.replace(".", "_")[:10]
    input_dat = NBG_DIR / f"_{tag}_input.dat"

    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image).convert("RGB")
        img = img.resize((512, 512), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1).reshape(1, 3, 512, 512)
        int16_arr = np.clip(np.round(arr * (2**15)), -32768, 32767).astype(np.int16)
        int16_arr.tofile(str(input_dat))
    except ImportError:
        raise RuntimeError("PIL not installed; NPU mode requires: pip3 install Pillow")

    # 2.2 — run vpm_run
    sample_txt = NBG_DIR / "_v4_sample.txt"
    sample_txt.write_text(
        f"[network]\n./network_binary.nb\n[input]\n./{input_dat.name}\n"
    )

    t0 = time.time()
    rc, out, err = _run(
        [str(VIPM_RUN), "-s", str(sample_txt), "-b", "0", "--save_txt", "1"],
        env={"LD_LIBRARY_PATH": str(VIP_LIB)}, cwd=str(NBG_DIR), timeout=300,
    )
    if rc != 0:
        raise RuntimeError(f"NPU run failed (rc={rc}):\n{out}\n{err}")

    import re
    profile_ms = None
    m = re.search(r"profile inference time=(\d+)us", out)
    if m: profile_ms = int(m.group(1)) / 1000

    print(f"[NPU vision: {profile_ms:.0f}ms]")

    # 2.3 — convert output_0.txt → float32 binary
    emb_bin = NBG_DIR / "_v4_embeddings.bin"
    vals = []
    with open(NBG_DIR / "output_0.txt") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: vals.append(float(line))
            except: pass
    with open(emb_bin, "wb") as f:
        for v in vals:
            f.write(struct.pack("<f", float(v)))

    # 2.4 — run llama‑cli with NPU embeddings
    cmd = [
        str(LLAMA_CLI),
        "-m", str(cfg["gguf"]),
        "--mmproj", str(cfg["mmproj"]),
        "--image", str(image),
        "-p", f"<image>{prompt}",
        "-n", str(max_tokens),
        "-t", "2",
        "--temp", str(temp),
        "--simple-io",
        "--no-perf",
        "--log-disable",
    ]
    env = {
        "LD_LIBRARY_PATH": str(LLAMA_LIB),
        "A733_NPU_EMBEDDINGS": str(emb_bin),
    }

    t_llm = time.time()
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env={**os.environ, **env},
    )
    p.stdin.write("/exit\n")
    p.stdin.close()

    answer_lines = []
    stderr_lines = []
    in_answer = False
    for line in p.stdout:
        line = line.rstrip("\n\r")
        stderr_lines.append(line)
        if "<image>" in line:
            in_answer = True
            continue
        if "[ Prompt:" in line or "[ Generation:" in line:
            continue
        if "Exiting" in line:
            continue
        if in_answer and line.strip() and ">" not in line[:2]:
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    t2 = time.time()

    for line in p.stderr:
        stderr_lines.append(line.rstrip("\n\r"))

    stderr_text = "\n".join(stderr_lines)
    prompt_tps, gen_tps = _parse_timing(stderr_text)

    elapsed = t2 - t0
    llm_time = t2 - t_llm
    print(f"\n-- {model_name} NPU-offload | wall {elapsed:.1f}s", end="")
    print(f" (vision {profile_ms:.0f}ms + LLM {llm_time:.1f}s)", end="")
    if prompt_tps: print(f" | prompt {prompt_tps:.0f} t/s", end="")
    if gen_tps: print(f" | gen {gen_tps:.0f} t/s", end="")
    print(f" | {_ram_info()} | 0 CPU for vision, ROS2 safe")

    # cleanup
    sample_txt.unlink(missing_ok=True)
    emb_bin.unlink(missing_ok=True)
    return "\n".join(answer_lines)


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V4 VLM Chat — image+text conversation on A733 / Orange Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 vlm_chat.py --image dog.jpg -p "Describe this image."
  python3 vlm_chat.py --image cat.jpg -p "What animal?" --backend npu
  python3 vlm_chat.py --image test-1.jpeg --model SmolVLM-500M
  python3 vlm_chat.py --cpu-only --image dog.jpg -p "What is this?"
""",
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("-p", "--prompt", default="Describe this image.", help="Question")
    parser.add_argument("--model", choices=["SmolVLM-256M", "SmolVLM-500M"],
                        default="SmolVLM-256M", help="Model (256M default)")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu",
                        help="cpu=llama.cpp CPU only | npu=NPU vision + CPU LLM")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max answer tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU mode, hide NPU option")
    args = parser.parse_args()

    if args.cpu_only:
        args.backend = "cpu"

    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")

    model_name = args.model
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model: {model_name}")

    cfg = MODELS[model_name]
    print(f"V4 VLM Chat -- {model_name} | {cfg['speed']} | {cfg['rss']}")
    print(f"Image: {image.name} | Prompt: {args.prompt}")
    print(f"Backend: {args.backend.upper()}")
    print()

    if args.backend == "npu":
        if not VIPM_RUN.exists():
            raise SystemExit(f"vpm_run not found at {VIPM_RUN}. Use --cpu-only")
        if not NBG_DIR.exists():
            raise SystemExit(f"NBG not found at {NBG_DIR}. Use --cpu-only")
        chat_npu(image, args.prompt, model_name, args.max_tokens, args.temp)
    else:
        chat_cpu(image, args.prompt, model_name, args.max_tokens, args.temp)


if __name__ == "__main__":
    main()
