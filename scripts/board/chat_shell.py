#!/usr/bin/env python3
"""Interactive fixed-window chat shell for the A733 NPU LM runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Iterable, Optional

from smollm2_chat import SmolTokenizer


DEFAULT_BASE = Path("/home/orangepi/a733_npu_driver")
DEFAULT_MODEL = DEFAULT_BASE / "models" / "smollm2_135m_w32_int16" / "network_binary.nb"
DEFAULT_TOKENIZER = DEFAULT_BASE / "work" / "models" / "smollm2-135m-instruct"
DEFAULT_RUNNER = DEFAULT_BASE / "build" / "npu_lm_runner"
DEFAULT_VIP_LIB = Path("/home/orangepi/lib")
DEFAULT_DEVICE = Path("/dev/vipcore")
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


class Tokenizer:
    def __init__(self, path: Path) -> None:
        self.root = path
        self.json_path = path / "tokenizer.json" if path.is_dir() else path
        self.config_path = self.json_path.parent / "tokenizer_config.json"
        if not self.json_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {self.json_path}")

        self.config: dict[str, object] = {}
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.backend_name = "HF tokenizer.json built-in"
        self._hf = None
        try:
            from tokenizers import Tokenizer as HfTokenizer  # type: ignore

            self._hf = HfTokenizer.from_file(str(self.json_path))
            self.backend_name = "Hugging Face tokenizers"
        except Exception:
            self._fallback = SmolTokenizer(self.json_path)
        else:
            self._fallback = None

    def encode(self, text: str) -> list[int]:
        if self._hf is not None:
            return list(self._hf.encode(text, add_special_tokens=False).ids)
        return self._fallback.encode(text)  # type: ignore[union-attr]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        ids_list = list(ids)
        if self._hf is not None:
            return self._hf.decode(ids_list, skip_special_tokens=skip_special)
        return self._fallback.decode(ids_list, skip_special=skip_special)  # type: ignore[union-attr]

    def token_to_id(self, token: str) -> Optional[int]:
        if self._hf is not None:
            value = self._hf.token_to_id(token)
            return int(value) if value is not None else None
        return self._fallback.specials.get(token)  # type: ignore[union-attr]

    def vocab_size(self) -> int:
        if self._hf is not None:
            return int(self._hf.get_vocab_size(with_added_tokens=True))
        return len(self._fallback.vocab)  # type: ignore[union-attr]

    def eos_token_id(self) -> Optional[int]:
        eos = self.config.get("eos_token")
        if isinstance(eos, str):
            return self.token_to_id(eos)
        return None


class RunnerClient:
    def __init__(self, args: argparse.Namespace, vocab: int) -> None:
        self.args = args
        self.vocab = vocab
        self.proc: Optional[subprocess.Popen[str]] = None
        self._lines: queue.Queue[Optional[str]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self.startup_lines: list[str] = []
        self.ready_line = ""
        self.start()

    def command(self) -> list[str]:
        command = [
            str(self.args.runner),
            "--protocol",
            "--seq-len",
            str(self.args.window),
            "--vocab",
            str(self.vocab),
            "--temperature",
            f"{self.args.temperature:.8g}",
            "--seed",
            str(self.args.seed),
        ]
        if self.args.model.is_dir():
            command.extend(["--model-dir", str(self.args.model)])
        else:
            command.extend(["--nbg", str(self.args.model)])
        return command

    def start(self) -> None:
        env = os.environ.copy()
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{self.args.vip_lib}:{current_ld}" if current_ld else str(self.args.vip_lib)
        )
        self.proc = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        try:
            deadline = time.monotonic() + self.args.runner_start_timeout
            while True:
                line = self._readline(max(0.1, deadline - time.monotonic()))
                self.startup_lines.append(line.rstrip("\n"))
                if line.startswith("READY "):
                    self.ready_line = line.rstrip("\n")
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError("runner did not become ready")
        except Exception:
            self.close()
            raise

    def _reader_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _readline(self, timeout: float) -> str:
        if self.proc is None:
            raise RuntimeError("runner is not started")
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            if self.proc.poll() is not None:
                raise RuntimeError(f"runner exited with code {self.proc.returncode}")
            raise TimeoutError("timed out waiting for runner output")
        if line is None:
            raise RuntimeError(f"runner exited with code {self.proc.poll()}")
        return line

    def run_window(self, window: list[int]) -> dict[str, str]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("runner is not started")
        self.proc.stdin.write("RUN " + " ".join(str(token) for token in window) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self._readline(self.args.runner_token_timeout).strip()
            if line.startswith("TOKEN "):
                return parse_key_values(line.split(" ", 1)[1])
            if line.startswith("ERROR "):
                raise RuntimeError(f"runner returned {line}")

    def discard_pending_token(self) -> None:
        if self.proc is None:
            return
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                return
            if line.startswith("TOKEN ") or line.startswith("ERROR "):
                return

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=3)


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in text.split():
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


def render_chat(messages: list[tuple[str, str]], system: Optional[str]) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for role, content in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def model_nbg_path(model: Path) -> Path:
    return model / "network_binary.nb" if model.is_dir() else model


def format_size(path: Path) -> str:
    size = path.stat().st_size
    return f"{size:,} bytes"


def make_window(tokens: list[int], window: int, pad_token: int) -> tuple[list[int], int, bool]:
    if len(tokens) >= window:
        return tokens[-window:], window, len(tokens) > window
    used = len(tokens)
    return [pad_token] * (window - used) + tokens, used, False


def print_counter(used: int, window: int) -> None:
    text = f"[window {used}/{window}]"
    if sys.stderr.isatty():
        print("\r" + text, end="", file=sys.stderr, flush=True)
    else:
        print(text, file=sys.stderr, flush=True)


def stop_ids_from_args(tokenizer: Tokenizer, args: argparse.Namespace) -> set[int]:
    ids = {int(token) for token in args.stop_token}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            ids.add(token_id)
    eos = tokenizer.eos_token_id()
    if eos is not None:
        ids.add(eos)
    return ids


def generate_reply(
    tokenizer: Tokenizer,
    runner: RunnerClient,
    args: argparse.Namespace,
    messages: list[tuple[str, str]],
    stop_ids: set[int],
) -> tuple[str, bool, int, float]:
    context = tokenizer.encode(render_chat(messages, args.system))
    generated: list[int] = []
    displayed = ""
    warned = False
    start = time.monotonic()

    try:
        for _ in range(args.max_new_tokens):
            window_ids, used, clipped = make_window(context, args.window, args.pad_token)
            if clipped and not warned:
                print(
                    f"\n[context] fixed window is {args.window} tokens; using the last {args.window}",
                    file=sys.stderr,
                    flush=True,
                )
                warned = True
            result = runner.run_window(window_ids)

            token = int(result["id"])
            if token in stop_ids:
                break
            generated.append(token)
            context.append(token)
            rendered = tokenizer.decode(generated, skip_special=True)
            delta = rendered[len(displayed) :]
            if delta:
                print(delta, end="", flush=True)
                displayed = rendered
            print_counter(min(len(context), args.window), args.window)
    except KeyboardInterrupt:
        runner.discard_pending_token()
        print("\n[turn stopped]", file=sys.stderr, flush=True)
        return displayed.strip(), True, len(generated), time.monotonic() - start

    elapsed = time.monotonic() - start
    if sys.stderr.isatty():
        print(file=sys.stderr)
    return displayed.strip(), False, len(generated), elapsed


def validate_args(args: argparse.Namespace) -> None:
    nbg = model_nbg_path(args.model)
    if not nbg.exists():
        raise FileNotFoundError(f"NBG not found: {nbg}")
    if not args.runner.exists():
        raise FileNotFoundError(f"runner not found: {args.runner}")
    if not os.access(args.runner, os.X_OK):
        raise PermissionError(f"runner is not executable: {args.runner}")
    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.greedy:
        args.temperature = 0.0
    if args.temperature < 0.0:
        args.temperature = 0.0
    if args.no_system:
        args.system = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="NBG file or package dir")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--vip-lib", type=Path, default=DEFAULT_VIP_LIB)
    parser.add_argument("--device", type=Path, default=DEFAULT_DEVICE)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true", help="force argmax sampling")
    parser.add_argument("--vocab", type=int, default=0, help="override tokenizer vocab size")
    parser.add_argument("--pad-token", type=int, default=2)
    parser.add_argument("--stop-token", type=int, action="append", default=[0, 2])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--runner-start-timeout", type=float, default=120.0)
    parser.add_argument("--runner-token-timeout", type=float, default=30.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        tokenizer = Tokenizer(args.tokenizer)
        vocab = args.vocab or tokenizer.vocab_size()
        stop_ids = stop_ids_from_args(tokenizer, args)

        nbg = model_nbg_path(args.model)
        print(f"model={nbg}")
        print(f"nbg_size={format_size(nbg)}")
        print(f"device={args.device} present={args.device.exists()}")
        print(f"tokenizer={tokenizer.backend_name} path={tokenizer.json_path}")
        print("NPU-only: model layers on NPU, CPU does tokenize/argmax/loop.")

        runner = RunnerClient(args, vocab)
        for line in runner.startup_lines:
            if line.startswith(
                (
                    "vip_init=",
                    "cid=",
                    "create_network_us=",
                    "prepare_network_us=",
                    "nbg_loaded_once=",
                )
            ):
                print(line)
        print(runner.ready_line)
    except Exception as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2

    messages: list[tuple[str, str]] = []
    try:
        print("Type /reset to clear the window, /exit to quit.")
        while True:
            try:
                user_text = input("\nuser> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("\n[input interrupted]")
                continue

            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                return 0
            if user_text == "/reset":
                messages.clear()
                print("context reset")
                continue

            messages.append(("user", user_text))
            print("assistant> ", end="", flush=True)
            answer, interrupted, token_count, elapsed = generate_reply(
                tokenizer, runner, args, messages, stop_ids
            )
            print()
            tok_s = token_count / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[reply] tokens={token_count} tok_s={tok_s:.3f} window={args.window}",
                file=sys.stderr,
                flush=True,
            )
            if interrupted:
                messages.pop()
                continue
            messages.append(("assistant", answer))
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
