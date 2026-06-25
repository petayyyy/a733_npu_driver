#!/usr/bin/env python3
"""
VLM Image Chat — CLI tool for image+text conversations on Orange Pi Zero 3W.
Pure terminal, no web server.

Backends:
  cpu  — SmolVLM on CPU via llama.cpp (fast, recommended, ~52 tok/s)
  npu  — NPU vision offload (V2d) + CPU LLM (frees A76 cores, ~6s vision)

Usage:
  python3 vlm_chat.py --image dog.jpg                           # interactive REPL
  python3 vlm_chat.py --image dog.jpg --question "What animal?" # one-shot
  python3 vlm_chat.py --image dog.jpg --backend npu             # NPU vision
  python3 vlm_chat.py --image dog.jpg --model smolvlm-500m      # larger model
"""
import argparse
import os
import re
import select
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / "a733_npu_driver"
VLM_DIR = REPO / "models/vlm"
NBG_DIR = REPO / "models/smolvlm_256m_vision_v2d_int16"
LLAMA_BIN = HOME / "llama.cpp/build/bin"
LLAMA_CLI = LLAMA_BIN / "llama-cli"
VIPM_RUN = Path("/opt/vpm_run/vpm_run")
VIP_LIB = HOME / "lib"

MODELS = {
    "smolvlm-256m": {
        "gguf": VLM_DIR / "SmolVLM-256M-Instruct-Q8_0.gguf",
        "mmproj": VLM_DIR / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        "speed": "~52 tok/s",
        "rss": "~634 MB",
        "label": "SmolVLM-256M",
    },
    "smolvlm-500m": {
        "gguf": VLM_DIR / "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj": VLM_DIR / "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "speed": "~22 tok/s",
        "rss": "~1.2 GB",
        "label": "SmolVLM-500M",
    },
}


def _ram_info():
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = int(lines[0].split()[1]) // 1024
        avail = int([l for l in lines if "Available" in l][0].split()[1]) // 1024
        used = total - avail
        return f"RAM {used}/{total} MB used ({avail} MB free)"
    except Exception:
        return "RAM: unknown"


def _parse_timing(stderr_text):
    prompt_tps = gen_tps = None
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", stderr_text)
    if m:
        prompt_tps = float(m.group(1))
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", stderr_text)
    if m:
        gen_tps = float(m.group(1))
    return prompt_tps, gen_tps


def _print_stats(label, backend, elapsed, gen_tps, prompt_tps):
    parts = [f"-- {label} {backend} | wall {elapsed:.1f}s"]
    if prompt_tps:
        parts.append(f"prompt {prompt_tps:.0f} t/s")
    if gen_tps:
        parts.append(f"gen {gen_tps:.0f} t/s")
    parts.append(_ram_info())
    parts.append("2xA76" if backend == "CPU" else "NPU vision + 2xA76 LLM")
    print(" | ".join(parts))


# ═══════════════════════════════════════════════════════════════════════
#  CPU backend
# ═══════════════════════════════════════════════════════════════════════

def _run_llama_cli(cfg, args_list, env_extra=None):
    cmd = [
        str(LLAMA_CLI),
        "-m", str(cfg["gguf"]),
        "--mmproj", str(cfg["mmproj"]),
        "-t", "2",
        "--cpu-range", "6-7",
        "--simple-io",
        "--log-disable",
    ] + args_list

    env = {"LD_LIBRARY_PATH": str(LLAMA_BIN)}
    if env_extra:
        env.update(env_extra)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
    )


def chat_cpu_one_shot(image, question, model_name, max_tokens, temp):
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"GGUF not found: {cfg['gguf']}")
    if not cfg["mmproj"].exists():
        raise FileNotFoundError(f"mmproj not found: {cfg['mmproj']}")

    t0 = time.time()
    p = _run_llama_cli(cfg, [
        "--image", str(image),
        "-p", f"<image>{question}",
        "-n", str(max_tokens),
        "--temp", str(temp),
        "--single-turn",
    ])

    p.stdin.write("/exit\n")
    p.stdin.close()

    answer_lines = []
    timing_lines = []
    in_answer = False
    for line in p.stdout:
        line = line.rstrip("\n\r")
        if "<image>" in line:
            in_answer = True
            continue
        if "Exiting" in line or "available commands" in line:
            continue
        if line.startswith("[ Prompt:") or line.startswith("[ Generation:"):
            timing_lines.append(line)
            continue
        if in_answer and line.strip():
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    for line in p.stderr:
        timing_lines.append(line.rstrip("\n\r"))

    t1 = time.time()
    prompt_tps, gen_tps = _parse_timing("\n".join(timing_lines))
    _print_stats(cfg["label"], "CPU", t1 - t0, gen_tps, prompt_tps)
    return "\n".join(answer_lines)


def chat_cpu_interactive(image, model_name, max_tokens, temp):
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"GGUF not found: {cfg['gguf']}")
    if not cfg["mmproj"].exists():
        raise FileNotFoundError(f"mmproj not found: {cfg['mmproj']}")

    print(f"VLM Chat — {cfg['label']} CPU | {cfg['speed']} | {cfg['rss']}")
    print(f"Image: {image.name} | Type questions, /exit to quit, /image <path> to change")
    print()

    p = _run_llama_cli(cfg, [
        "--image", str(image),
        "-n", str(max_tokens),
        "--temp", str(temp),
        "--conversation",
    ])

    stderr_collector = []

    def _read_stderr():
        while True:
            r, _, _ = select.select([p.stderr], [], [], 0.05)
            if not r:
                break
            line = p.stderr.readline()
            if line:
                stderr_collector.append(line.rstrip("\n\r"))

    def _read_until_idle(timeout=2.0):
        buf = []
        last_read = time.time()
        while True:
            _read_stderr()
            r, _, _ = select.select([p.stdout], [], [], 0.1)
            if r:
                line = p.stdout.readline()
                if line:
                    line = line.rstrip("\n\r")
                    if line.startswith("> ") or "available commands" in line:
                        continue
                    if line.startswith("build ") or line.startswith("model "):
                        continue
                    if line.startswith("modalities"):
                        continue
                    buf.append(line)
                    if line.strip():
                        print(line, flush=True)
                    last_read = time.time()
                    continue
            if time.time() - last_read > timeout:
                break
            if p.poll() is not None:
                break
        return buf

    # wait for model loading and initial output to settle
    _read_until_idle(timeout=1.0)
    print()

    # send the initial question with <image> tag for vision context
    first_question = "<image>Describe this image."
    p.stdin.write(f"{first_question}\n")
    p.stdin.flush()
    _read_until_idle(timeout=5.0)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/exiting...")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input.startswith("/image "):
            new_img = user_input[7:].strip()
            if Path(new_img).exists():
                image = Path(new_img)
                p.stdin.write(f"/image {new_img}\n")
                p.stdin.flush()
                print(f"[Image changed to: {image.name}]")
            else:
                print(f"[Image not found: {new_img}]")
            continue

        p.stdin.write(f"{user_input}\n")
        p.stdin.flush()
        _read_until_idle(timeout=3.0)

    p.stdin.close()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

    # final stats from stderr
    stderr_text = "\n".join(stderr_collector)
    prompt_tps, gen_tps = _parse_timing(stderr_text)
    print()
    _print_stats(cfg["label"], "CPU", 0, gen_tps, prompt_tps)


# ═══════════════════════════════════════════════════════════════════════
#  NPU vision-offload backend
# ═══════════════════════════════════════════════════════════════════════

def _npu_vision_encode(image, model_name):
    cfg = MODELS[model_name]
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

    sample_txt = NBG_DIR / "_v4_sample.txt"
    sample_txt.write_text(
        f"[network]\n./network_binary.nb\n[input]\n./{input_dat.name}\n"
    )

    t0 = time.time()
    p = subprocess.run(
        [str(VIPM_RUN), "-s", str(sample_txt), "-b", "0", "--save_txt", "1"],
        capture_output=True, text=True, timeout=300,
        env={"LD_LIBRARY_PATH": str(VIP_LIB)},
        cwd=str(NBG_DIR),
    )
    if p.returncode != 0:
        raise RuntimeError(f"NPU run failed (rc={p.returncode}):\n{p.stdout}\n{p.stderr}")

    profile_ms = None
    m = re.search(r"profile inference time=(\d+)us", p.stdout)
    if m:
        profile_ms = int(m.group(1)) / 1000

    print(f"[NPU vision: {profile_ms:.0f}ms]")

    emb_bin = NBG_DIR / "_v4_embeddings.bin"
    vals = []
    with open(NBG_DIR / "output_0.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                pass
    with open(emb_bin, "wb") as f:
        for v in vals:
            f.write(struct.pack("<f", float(v)))

    sample_txt.unlink(missing_ok=True)
    return emb_bin, profile_ms


def chat_npu_one_shot(image, question, model_name, max_tokens, temp):
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"GGUF not found: {cfg['gguf']}")
    if not cfg["mmproj"].exists():
        raise FileNotFoundError(f"mmproj not found: {cfg['mmproj']}")

    t0 = time.time()
    emb_bin, profile_ms = _npu_vision_encode(image, model_name)

    t_llm = time.time()
    p = _run_llama_cli(cfg, [
        "--image", str(image),
        "-p", f"<image>{question}",
        "-n", str(max_tokens),
        "--temp", str(temp),
        "--single-turn",
    ], env_extra={"A733_NPU_EMBEDDINGS": str(emb_bin)})

    p.stdin.write("/exit\n")
    p.stdin.close()

    answer_lines = []
    timing_lines = []
    in_answer = False
    for line in p.stdout:
        line = line.rstrip("\n\r")
        if "<image>" in line:
            in_answer = True
            continue
        if "Exiting" in line or "available commands" in line:
            continue
        if line.startswith("[ Prompt:") or line.startswith("[ Generation:"):
            timing_lines.append(line)
            continue
        if in_answer and line.strip():
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    for line in p.stderr:
        timing_lines.append(line.rstrip("\n\r"))

    t2 = time.time()
    prompt_tps, gen_tps = _parse_timing("\n".join(timing_lines))
    elapsed = t2 - t0
    llm_time = t2 - t_llm
    print()
    print(f"-- {cfg['label']} NPU-offload | wall {elapsed:.1f}s", end="")
    print(f" (vision {profile_ms:.0f}ms + LLM {llm_time:.1f}s)", end="")
    if prompt_tps:
        print(f" | prompt {prompt_tps:.0f} t/s", end="")
    if gen_tps:
        print(f" | gen {gen_tps:.0f} t/s", end="")
    print(f" | {_ram_info()} | NPU vision, ROS2 safe")

    emb_bin.unlink(missing_ok=True)
    return "\n".join(answer_lines)


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM Image Chat — CLI image+text conversation on Orange Pi Zero 3W",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 vlm_chat.py --image dog.jpg
  python3 vlm_chat.py --image dog.jpg --question "Describe this image."
  python3 vlm_chat.py --image dog.jpg --backend npu
  python3 vlm_chat.py --image dog.jpg --model smolvlm-500m
""",
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--question", "-q", default=None,
                        help="One-shot question (without this, enters interactive REPL)")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="smolvlm-256m", help="Model (smolvlm-256m default)")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu",
                        help="cpu=llama.cpp CPU only | npu=NPU vision + CPU LLM")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max answer tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU mode even if NPU is available")
    args = parser.parse_args()

    if args.cpu_only:
        args.backend = "cpu"

    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")

    cfg = MODELS[args.model]

    # NPU availability check
    if args.backend == "npu":
        if not VIPM_RUN.exists():
            print(f"[WARNING] vpm_run not found at {VIPM_RUN}. Falling back to CPU.", file=sys.stderr)
            args.backend = "cpu"
        elif not NBG_DIR.exists() or not (NBG_DIR / "network_binary.nb").exists():
            print(f"[WARNING] NBG not found at {NBG_DIR}. Falling back to CPU.", file=sys.stderr)
            args.backend = "cpu"
        elif not Path("/dev/vipcore").exists():
            print(f"[WARNING] /dev/vipcore not found. Falling back to CPU.", file=sys.stderr)
            args.backend = "cpu"

    if args.question:
        # one-shot
        print(f"VLM Chat — {cfg['label']} {args.backend.upper()} | {cfg['speed']} | {cfg['rss']}")
        print(f"Image: {image.name} | Q: {args.question}")
        print()

        if args.backend == "npu":
            chat_npu_one_shot(image, args.question, args.model, args.max_tokens, args.temp)
        else:
            chat_cpu_one_shot(image, args.question, args.model, args.max_tokens, args.temp)
    else:
        # interactive REPL
        if args.backend == "npu":
            print("Interactive mode with NPU backend is not yet supported. Use --question for one-shot.", file=sys.stderr)
            print("Falling back to CPU interactive mode.", file=sys.stderr)
            args.backend = "cpu"

        chat_cpu_interactive(image, args.model, args.max_tokens, args.temp)


if __name__ == "__main__":
    main()
