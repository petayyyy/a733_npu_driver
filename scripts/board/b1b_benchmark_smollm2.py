#!/usr/bin/env python3
"""B1b fixed-window SmolLM2 board benchmark.

The script keeps the NBG loaded through npu_lm_runner --protocol, sends one
fixed token window per generated token, samples runner RSS, and writes a JSON
record that can be copied back into the host report. Model-layer compute stays
inside the NBG; Python does tokenization, window management, argmax plumbing,
detokenization, logging, and RSS sampling.
"""

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

from smollm2_chat import DEFAULT_SYSTEM, SmolTokenizer, render_chat


DEFAULT_BASE = Path("/home/orangepi/a733_npu_driver")
DEFAULT_RUNNER = DEFAULT_BASE / "build" / "npu_lm_runner"
DEFAULT_TOKENIZER = DEFAULT_BASE / "work" / "models" / "smollm2-135m-instruct" / "tokenizer.json"
DEFAULT_VIP_LIB = Path("/home/orangepi/lib")
DEFAULT_PROMPT = "The capital of France is"


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in text.split():
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


class RunnerProtocol:
    def __init__(self, args: argparse.Namespace, vocab: int) -> None:
        self.args = args
        self.vocab = vocab
        self.proc: Optional[subprocess.Popen[str]] = None
        self.sampler: Optional[RssSampler] = None
        self.lines: queue.Queue[Optional[str]] = queue.Queue()
        self.startup_lines: list[str] = []
        self.ready_line = ""
        self._reader: Optional[threading.Thread] = None
        self.start()

    def command(self) -> list[str]:
        command = [
            str(self.args.runner),
            "--protocol",
            "--nbg",
            str(self.args.nbg),
            "--seq-len",
            str(self.args.window),
            "--vocab",
            str(self.vocab),
            "--temperature",
            "0",
        ]
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
        self.sampler = RssSampler(self.proc.pid, self.args.rss_interval_s)
        self.sampler.start()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + self.args.start_timeout_s
        while True:
            line = self.readline(max(0.1, deadline - time.monotonic()))
            stripped = line.rstrip("\n")
            self.startup_lines.append(stripped)
            if line.startswith("READY "):
                self.ready_line = stripped
                return
            if time.monotonic() > deadline:
                raise TimeoutError("runner did not become ready")

    def _reader_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def readline(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"runner exited with code {self.proc.returncode}")
            raise TimeoutError("timed out waiting for runner output")
        if line is None:
            code = self.proc.poll() if self.proc is not None else None
            raise RuntimeError(f"runner exited with code {code}")
        return line

    def run_window(self, window: list[int]) -> dict[str, str]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("runner is not started")
        self.proc.stdin.write("RUN " + " ".join(str(token) for token in window) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.readline(self.args.token_timeout_s).strip()
            if line.startswith("TOKEN "):
                return parse_key_values(line.split(" ", 1)[1])
            if line.startswith("ERROR "):
                raise RuntimeError(f"runner returned {line}")

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
        if self.sampler is not None:
            self.sampler.stop()


class RssSampler:
    def __init__(self, pid: int, interval_s: float) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.peak_kb = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        status = Path(f"/proc/{self.pid}/status")
        while not self._stop.is_set():
            try:
                for line in status.read_text(encoding="ascii", errors="ignore").splitlines():
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss = int(parts[1])
                            self.peak_kb = max(self.peak_kb, rss)
                            self.samples += 1
                        break
            except FileNotFoundError:
                return
            time.sleep(self.interval_s)


def make_window(tokens: list[int], window: int, pad_token: int) -> list[int]:
    if len(tokens) >= window:
        return tokens[-window:]
    return [pad_token] * (window - len(tokens)) + tokens


def token_count(tokenizer: SmolTokenizer) -> int:
    return max(tokenizer.id_to_token) + 1


def generated_text(tokenizer: SmolTokenizer, ids: Iterable[int]) -> str:
    return tokenizer.decode(ids, skip_special=True).strip()


def parse_metric(lines: list[str], key: str) -> Optional[int]:
    prefix = key + "="
    for line in lines:
        if line.startswith(prefix):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbg", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--vip-lib", type=Path, default=DEFAULT_VIP_LIB)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--pad-token", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rss-interval-s", type=float, default=0.02)
    parser.add_argument("--start-timeout-s", type=float, default=180.0)
    parser.add_argument("--token-timeout-s", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.no_system:
        args.system = None
    for path, label in ((args.nbg, "NBG"), (args.tokenizer, "tokenizer"), (args.runner, "runner")):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2
    if args.window <= 0 or args.steps <= 0:
        print("--window and --steps must be positive", file=sys.stderr)
        return 2

    tokenizer = SmolTokenizer(args.tokenizer)
    vocab = token_count(tokenizer)
    prompt_text = render_chat([("user", args.prompt)], args.system)
    prompt_tokens = tokenizer.encode(prompt_text)
    initial_window = make_window(prompt_tokens, args.window, args.pad_token)
    tokens = list(initial_window)
    generated: list[int] = []
    step_records: list[dict[str, object]] = []

    runner = RunnerProtocol(args, vocab)
    total_wall_us = 0
    total_profile_us = 0
    try:
        for step in range(args.steps):
            window = tokens[-args.window :]
            result = runner.run_window(window)
            token = int(result["id"])
            wall_us = int(result["wall_us"])
            profile_us = int(result["profile_us"])
            total_wall_us += wall_us
            total_profile_us += profile_us
            generated.append(token)
            tokens.append(token)
            step_records.append(
                {
                    "step": step,
                    "window_tail": window[-8:],
                    "token": token,
                    "profile_us": profile_us,
                    "cycle": int(result["cycle"]),
                    "wall_us": wall_us,
                    "top5": result.get("top5", ""),
                }
            )
    finally:
        runner.close()

    sampler = runner.sampler
    peak_rss_kb = sampler.peak_kb if sampler is not None else 0
    rss_samples = sampler.samples if sampler is not None else 0
    first_wall_us = int(step_records[0]["wall_us"])
    mean_wall_us = total_wall_us / float(args.steps)
    mean_profile_us = total_profile_us / float(args.steps)
    output = {
        "prompt": args.prompt,
        "system": args.system,
        "prompt_token_count": len(prompt_tokens),
        "window": args.window,
        "steps": args.steps,
        "nbg": str(args.nbg),
        "nbg_size_bytes": args.nbg.stat().st_size,
        "startup_lines": runner.startup_lines,
        "create_network_us": parse_metric(runner.startup_lines, "create_network_us"),
        "prepare_network_us": parse_metric(runner.startup_lines, "prepare_network_us"),
        "memory_pool_bytes": parse_metric(runner.startup_lines, "memory_pool_bytes"),
        "initial_window": initial_window,
        "generated_tokens": generated,
        "generated_text": generated_text(tokenizer, generated),
        "final_tokens": tokens,
        "final_decoded": tokenizer.decode(tokens, skip_special=False),
        "first_token_ms": first_wall_us / 1000.0,
        "decode_tok_s": 1000000.0 / mean_wall_us,
        "prefill_tok_s": args.window * 1000000.0 / first_wall_us,
        "mean_wall_us": mean_wall_us,
        "mean_profile_us": mean_profile_us,
        "total_wall_us": total_wall_us,
        "total_profile_us": total_profile_us,
        "peak_rss_kb": peak_rss_kb,
        "rss_samples": rss_samples,
        "records": step_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
