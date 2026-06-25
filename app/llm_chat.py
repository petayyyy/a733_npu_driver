#!/usr/bin/env python3
"""
LLM Text Chat -- CLI conversation on Orange Pi Zero 3W.
Pure terminal, no web server. Uses llama.cpp with Qwen2.5 models.

Usage:
  python3 llm_chat.py                              # interactive REPL (qwen-1.5b default)
  python3 llm_chat.py --question "What is Python?"  # one-shot
  python3 llm_chat.py --model qwen-0.5b             # faster model
  python3 llm_chat.py --model qwen-3b               # experimental, slow
  python3 llm_chat.py --cores 0-3                   # override A76 pinning
"""
import argparse
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / "a733_npu_driver"
LLAMA_BIN = HOME / "llama.cpp/build/bin"
LLAMA_CLI = LLAMA_BIN / "llama-cli"

MODELS = {
    "qwen-0.5b": {
        "gguf": REPO / "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        "speed": "~18 tok/s",
        "rss": "~1.1 GB",
        "label": "Qwen2.5-0.5B Q8_0",
        "ctx": 8192,
    },
    "qwen-1.5b": {
        "gguf": REPO / "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "speed": "~8.5 tok/s",
        "rss": "~2.0 GB",
        "label": "Qwen2.5-1.5B Q4_K_M",
        "ctx": 8192,
    },
    "qwen-3b": {
        "gguf": REPO / "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "speed": "~4 tok/s",
        "rss": "~3.7 GB",
        "label": "Qwen2.5-3B Q4_K_M (EXPERIMENTAL)",
        "ctx": 4096,
    },
}

SYSTEM_PROMPT = "You are a helpful AI assistant running on Orange Pi Zero 3W. Keep answers concise."


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


def _count_tokens(text):
    return len(text.split())


def _print_startup(cfg, cores):
    print(f"LLM Chat -- {cfg['label']} | {cfg['speed']} | {cfg['rss']}")
    print(f"Cores: {cores} | Context: {cfg['ctx']} | {_ram_info()}")
    print("Type /exit to quit, /reset to clear history.")
    print()


def chat_one_shot(question, model_name, max_tokens, temp, cores):
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"GGUF not found: {cfg['gguf']}")

    print(f"LLM Chat -- {cfg['label']} | {cfg['speed']} | {cfg['rss']}")
    print(f"Q: {question}")
    print()

    cmd = [
        str(LLAMA_CLI),
        "-m", str(cfg["gguf"]),
        "-p", question,
        "-n", str(max_tokens),
        "-t", "2",
        "--cpu-range", cores,
        "--temp", str(temp),
        "-c", str(cfg["ctx"]),
        "--simple-io",
        "--log-disable",
        "--single-turn",
        "--chat-template", "chatml",
    ]

    env = {"LD_LIBRARY_PATH": str(LLAMA_BIN)}
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
    )
    p.stdin.write("/exit\n")
    p.stdin.close()

    answer_lines = []
    timing_lines = []
    in_answer = False
    for line in p.stdout:
        line = line.rstrip("\n\r")
        if "Exiting" in line or "available commands" in line:
            continue
        if line.startswith("> "):
            in_answer = True
            continue
        if in_answer and line.strip():
            if line.startswith("[ Prompt:") or line.startswith("[ Generation:"):
                timing_lines.append(line)
                continue
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    for line in p.stderr:
        timing_lines.append(line.rstrip("\n\r"))

    t1 = time.time()
    prompt_tps, gen_tps = _parse_timing("\n".join(timing_lines))
    print()
    parts = [f"-- {cfg['label']} | wall {t1 - t0:.1f}s"]
    if prompt_tps:
        parts.append(f"prompt {prompt_tps:.0f} t/s")
    if gen_tps:
        parts.append(f"gen {gen_tps:.0f} t/s")
    parts.append(_ram_info())
    parts.append(f"cores {cores}")
    print(" | ".join(parts))
    return "\n".join(answer_lines)


def chat_interactive(model_name, max_tokens, temp, cores):
    cfg = MODELS[model_name]
    if not cfg["gguf"].exists():
        raise FileNotFoundError(f"GGUF not found: {cfg['gguf']}")

    _print_startup(cfg, cores)

    cmd = [
        str(LLAMA_CLI),
        "-m", str(cfg["gguf"]),
        "-n", str(max_tokens),
        "-t", "2",
        "--cpu-range", cores,
        "--temp", str(temp),
        "-c", str(cfg["ctx"]),
        "--simple-io",
        "--no-perf",
        "--log-disable",
        "--conversation",
        "--chat-template", "chatml",
        "--system-prompt", SYSTEM_PROMPT,
    ]

    env = {"LD_LIBRARY_PATH": str(LLAMA_BIN)}
    t_start = time.time()
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
    )

    stderr_collector = []
    token_count = 0
    turn = 0

    def _read_stderr():
        while True:
            r, _, _ = select.select([p.stderr], [], [], 0.05)
            if not r:
                break
            line = p.stderr.readline()
            if line:
                stderr_collector.append(line.rstrip("\n\r"))

    def _read_until_idle(timeout=3.0):
        nonlocal token_count
        last_read = time.time()
        generated = 0
        while True:
            _read_stderr()
            r, _, _ = select.select([p.stdout], [], [], 0.1)
            if r:
                line = p.stdout.readline()
                if not line:
                    if p.poll() is not None:
                        break
                    continue
                line = line.rstrip("\n\r")
                if line.startswith("[ Prompt:") or line.startswith("[ Generation:"):
                    continue
                if line.startswith("> "):
                    continue
                if line.strip():
                    print(line, flush=True)
                    generated += 1
                    token_count += _count_tokens(line)
                    last_read = time.time()
                continue
            if time.time() - last_read > timeout:
                break
            if p.poll() is not None:
                break
        return generated

    # wait for model to load
    sys.stderr.write("[Loading model...]\n")
    sys.stderr.flush()
    _read_until_idle(timeout=0.5)
    sys.stderr.write("[Ready]\n\n")
    sys.stderr.flush()

    # initial greeting
    p.stdin.write("Hello! Introduce yourself briefly.\n")
    p.stdin.flush()
    _read_until_idle(timeout=3.0)
    turn += 1

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
        if user_input == "/reset":
            p.stdin.write("/clear\n")
            p.stdin.flush()
            token_count = 0
            turn = 0
            print("[History cleared]")
            continue

        p.stdin.write(f"{user_input}\n")
        p.stdin.flush()
        turn += 1
        _read_until_idle(timeout=4.0)

    p.stdin.close()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

    elapsed = time.time() - t_start
    stderr_text = "\n".join(stderr_collector)
    prompt_tps, gen_tps = _parse_timing(stderr_text)

    print()
    parts = [f"-- {cfg['label']} | session {elapsed:.0f}s | {turn} turns"]
    if gen_tps:
        parts.append(f"gen {gen_tps:.0f} t/s")
    parts.append(_ram_info())
    parts.append(f"cores {cores}")
    print(" | ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="LLM Text Chat -- CLI conversation on Orange Pi Zero 3W",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 llm_chat.py
  python3 llm_chat.py --question "What is the capital of France?"
  python3 llm_chat.py --model qwen-0.5b
  python3 llm_chat.py --model qwen-3b
  python3 llm_chat.py --cores 0-3
""",
    )
    parser.add_argument("--question", "-q", default=None,
                        help="One-shot question (without this, enters interactive REPL)")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="qwen-1.5b", help="Model (qwen-1.5b default)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max answer tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--cores", default="6-7",
                        help="CPU core range for affinity (default: 6-7 = A76 only)")
    args = parser.parse_args()

    cfg = MODELS[args.model]

    if args.model == "qwen-3b":
        print("[WARNING] qwen-3b is experimental. ~4 tok/s, ~3.7 GB RAM.", file=sys.stderr)
        print("[WARNING] First load from SD card may take 30-60 seconds.", file=sys.stderr)

    if args.question:
        chat_one_shot(args.question, args.model, args.max_tokens, args.temp, args.cores)
    else:
        chat_interactive(args.model, args.max_tokens, args.temp, args.cores)


if __name__ == "__main__":
    main()
