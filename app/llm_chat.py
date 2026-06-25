#!/usr/bin/env python3
"""
V4 LLM Chat — text conversation on A733 / Orange Pi Zero 3W.

Backends:
  npu — SmolLM2-135M / 360M on NPU via npu_lm_runner (protocol mode)
  cpu — Qwen2.5-0.5B on CPU via llama.cpp (real KV-cache, 8K context)

Usage:
  python3 llm_chat.py -p "Explain quantum computing in one sentence."
  python3 llm_chat.py --backend cpu --model Qwen2.5-0.5B -p "Hello"
  python3 llm_chat.py --cpu-only -p "Tell me a joke"
"""
import argparse, os, re, struct, subprocess, sys, time, json, math
from pathlib import Path
from typing import Iterator

# ── paths ──────────────────────────────────────────────────────────────
HOME      = Path.home()
REPO      = HOME / "a733_npu_driver"
BUILD     = REPO / "build"
LLAMA_BIN = HOME / "llama.cpp/build/bin"
VIP_LIB   = HOME / "lib"

MODELS = {
    "SmolLM2-135M": {
        "backend":  "npu",
        "nbg":      REPO / "models/smollm2_135m_w32_int16/network_binary.nb",
        "tokenizer": REPO / "work/models/smollm2-135m-instruct/tokenizer.json",
        "vocab": 49152, "window": 32,
        "speed": "~21 tok/s", "rss": "~272 MB",
        "desc": "NPU, fast, coherent short answers",
    },
    "SmolLM2-360M": {
        "backend":  "npu",
        "nbg":      REPO / "models/smollm2_360m_w32_int16/network_binary.nb",
        "tokenizer": REPO / "work/models/smollm2-360m-instruct/tokenizer.json",
        "vocab": 49152, "window": 32,
        "speed": "~8 tok/s", "rss": "~646 MB",
        "desc": "NPU, smarter, slower",
    },
    "Qwen2.5-0.5B": {
        "backend":  "cpu",
        "gguf":     REPO / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "gguf_q8":  REPO / "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        "context": 8192,
        "speed": "~18 tok/s (Q8_0)", "rss": "~1.1 GB",
        "desc": "CPU, Qwen2.5, real KV-cache",
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  tokenizer (minimal, no external deps)
# ═══════════════════════════════════════════════════════════════════════

# ── tokenizer ──────────────────────────────────────────────────────────

try:
    from tokenizers import Tokenizer as HfTokenizer
    _HAS_HF_TOKENIZER = True
except ImportError:
    _HAS_HF_TOKENIZER = False


class _SmolTokenizer:
    """Thin wrapper: HF tokenizers if available, else minimal fallback."""

    def __init__(self, path: Path):
        if _HAS_HF_TOKENIZER:
            self._hf = HfTokenizer.from_file(str(path))
            self._vocab_size = self._hf.get_vocab_size()
        else:
            self._hf = None
            self._vocab_size = 0

    def encode(self, text: str) -> list[int]:
        if self._hf:
            return self._hf.encode(text).ids
        return _fallback_encode(text)

    def decode(self, ids: list[int]) -> str:
        if self._hf:
            return self._hf.decode(ids)
        return _fallback_decode(ids)


def _fallback_encode(text: str) -> list[int]:
    """Minimal fallback using llama-cli tokenization."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        f.flush()
        tmp = f.name
    try:
        p = subprocess.run(
            [str(HOME / "llama.cpp/build/bin/llama-cli"),
             "-m", str(list(MODELS.values())[0].get("nbg", "")) or "/dev/null",
             "--tokenize", tmp],
            capture_output=True, text=True, timeout=30, cwd=str(HOME),
        )
        # llama-cli --tokenize outputs tokens
        tokens = []
        for line in p.stdout.strip().split("\n"):
            try: tokens.append(int(line.strip()))
            except: pass
        return tokens if tokens else []
    except Exception:
        return []
    finally:
        os.unlink(tmp)


def _fallback_decode(ids: list[int]) -> str:
    return " ".join(str(i) for i in ids)


def _ram_info() -> str:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = int(lines[0].split()[1]) // 1024
        avail = int([l for l in lines if "Available" in l][0].split()[1]) // 1024
        return f"RAM {total - avail}/{total} MB used ({avail} MB free)"
    except Exception:
        return "RAM: unknown"


# ═══════════════════════════════════════════════════════════════════════
#  NPU backend (SmolLM2 via npu_lm_runner --protocol)
# ═══════════════════════════════════════════════════════════════════════

def chat_npu(model_name: str, prompt: str, max_tokens: int, temp: float) -> str:
    cfg = MODELS[model_name]
    runner_bin = BUILD / "npu_lm_runner"
    if not runner_bin.exists():
        raise FileNotFoundError(f"Runner not found: {runner_bin}")

    # build runner command
    runner_cmd = [
        str(runner_bin),
        "--nbg", str(cfg["nbg"]),
        "--seq-len", str(cfg["window"]),
        "--vocab", str(cfg["vocab"]),
        "--temperature", str(temp),
        "--protocol",
    ]

    env = {"LD_LIBRARY_PATH": str(VIP_LIB)}

    p = subprocess.Popen(
        runner_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env={**os.environ, **env},
    )

    # wait for READY
    ready = False
    for line in p.stdout:
        if "READY" in line:
            ready = True
            break
        # print stderr startup info
    if not ready:
        p.kill()
        raise RuntimeError("Runner failed to start")

    # load tokenizer
    tok_path = cfg["tokenizer"]
    if not tok_path.exists():
        p.kill()
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tok = _SmolTokenizer(tok_path)

    pad_id = 2     # SmolLM2 pad
    eos_tokens = {0, 2, 49152, 49153}  # <|endoftext|>, <|im_end|>, and common stops

    # build prompt with chat template
    chat_prompt = (
        "<|im_start|>system\n"
        "You are a helpful AI assistant. Keep answers short.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_ids = tok.encode(chat_prompt)
    window = cfg["window"]

    # sliding window: keep last N tokens
    if len(prompt_ids) > window:
        prompt_ids = prompt_ids[-window:]
    elif len(prompt_ids) < window:
        # left-pad with pad token
        prompt_ids = [pad_id] * (window - len(prompt_ids)) + prompt_ids

    # run generation
    t0 = time.time()
    answer_pieces: list[str] = []
    gen_count = 0

    for step in range(max_tokens):
        run_cmd = "RUN " + " ".join(str(t) for t in prompt_ids) + "\n"
        p.stdin.write(run_cmd)
        p.stdin.flush()

        resp = p.stdout.readline().strip()
        if not resp:
            break

        if resp.startswith("ERROR"):
            break

        # parse TOKEN response
        m = re.search(r"TOKEN id=(\d+)", resp)
        if not m:
            break
        next_id = int(m.group(1))

        if next_id in eos_tokens and gen_count > 3:
            break

        piece = tok.decode([next_id])
        answer_pieces.append(piece)
        sys.stdout.write(piece)
        sys.stdout.flush()
        gen_count += 1

        # slide window
        prompt_ids = prompt_ids[1:] + [next_id]

    # shutdown
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=5)
    t1 = time.time()

    elapsed = t1 - t0
    tok_s = gen_count / elapsed if elapsed > 0 else 0
    print(f"\n-- {model_name} NPU | {gen_count} tokens | wall {elapsed:.1f}s | "
          f"{tok_s:.0f} tok/s | window {window} | {_ram_info()} | 0 CPU, ROS2 safe")
    return "".join(answer_pieces)


# ═══════════════════════════════════════════════════════════════════════
#  CPU backend (Qwen2.5 via llama-completion)
# ═══════════════════════════════════════════════════════════════════════

def chat_cpu(model_name: str, prompt: str, max_tokens: int, temp: float) -> str:
    cfg = MODELS[model_name]

    # prefer Q8_0 if available, fall back to Q4_K_M
    gguf_q8 = cfg.get("gguf_q8")
    gguf = Path(gguf_q8) if gguf_q8 and Path(gguf_q8).exists() else Path(cfg["gguf"])
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf} (tried Q8 and Q4)")

    llama_bin = LLAMA_BIN / "llama-completion"
    if not llama_bin.exists():
        raise FileNotFoundError(f"llama-completion not found: {llama_bin}")

    cmd = [
        str(llama_bin),
        "-m", str(gguf),
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", "2",
        "--temp", str(temp),
        "-c", str(cfg["context"]),
        "-no-cnv",
    ]

    env = {"LD_LIBRARY_PATH": str(LLAMA_BIN)}
    t0 = time.time()
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={**os.environ, **env},
    )

    answer_lines: list[str] = []
    for line in p.stdout:
        line = line.rstrip("\n\r")
        if "[end of text]" in line:
            continue
        if line.strip():
            answer_lines.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    p.wait()
    t1 = time.time()

    # parse timing from stderr
    stderr_text = p.stderr.read()
    prompt_tps = gen_tps = None
    m = re.search(r"prompt eval time.*?(\d+\.?\d*) tokens per second", stderr_text)
    if m: prompt_tps = float(m.group(1))
    m = re.search(r"eval time.*?(\d+\.?\d*) tokens per second", stderr_text)
    if m: gen_tps = float(m.group(1))

    elapsed = t1 - t0
    print(f"\n-- {model_name} CPU | wall {elapsed:.1f}s", end="")
    if prompt_tps: print(f" | prompt {prompt_tps:.0f} t/s", end="")
    if gen_tps: print(f" | gen {gen_tps:.0f} t/s", end="")
    print(f" | ctx {cfg['context']} | {_ram_info()} | 2xA76, 6 cores free")
    return "\n".join(answer_lines)


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V4 LLM Chat — text conversation on A733 / Orange Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 llm_chat.py -p "Explain quantum computing in one sentence."
  python3 llm_chat.py --backend cpu --model Qwen2.5-0.5B -p "Hello"
  python3 llm_chat.py --model SmolLM2-360M -p "Write a haiku about AI"
  python3 llm_chat.py --cpu-only -p "Tell me a joke"
""",
    )
    parser.add_argument("-p", "--prompt", required=True, help="Your question")
    parser.add_argument("--model", default="SmolLM2-135M",
                        choices=list(MODELS.keys()),
                        help="Model (SmolLM2-135M default)")
    parser.add_argument("--backend", choices=["cpu", "npu"], default=None,
                        help="Override auto-detected backend")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max answer tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only show CPU models, force CPU backend")
    args = parser.parse_args()

    if args.cpu_only:
        args.model = "Qwen2.5-0.5B"

    model_name = args.model
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model: {model_name}")

    cfg = MODELS[model_name]
    backend = args.backend or cfg["backend"]

    # validate
    if backend == "npu":
        if not cfg.get("nbg") or not Path(cfg["nbg"]).exists():
            raise SystemExit(f"NBG not found: {cfg.get('nbg')}. Use --cpu-only or pick cpu model.")

    print(f"V4 LLM Chat -- {model_name} | {cfg['speed']} | {cfg['rss']}")
    print(f"Prompt: {args.prompt}")
    print(f"Backend: {backend.upper()} | {cfg['desc']}")
    print()

    if backend == "npu":
        chat_npu(model_name, args.prompt, args.max_tokens, args.temp)
    else:
        chat_cpu(model_name, args.prompt, args.max_tokens, args.temp)


if __name__ == "__main__":
    main()
