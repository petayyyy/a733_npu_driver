#!/usr/bin/env python3
"""Interactive SmolLM2 chat wrapper for the A733 NPU runner.

This script is intentionally dependency-free so it can run on the Radxa board
without installing Hugging Face tokenizers. It reads tokenizer.json directly,
tokenizes chat messages, launches the persistent VIPLite runner for one answer,
and decodes the generated token ids back to text.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional


DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
DEFAULT_MODEL_DIR = Path("/home/radxa/a733_npu_driver/models/smollm2_135m_w64_int16")
DEFAULT_TOKENIZER = Path("/home/radxa/a733_npu_driver/work/models/smollm2-135m-instruct/tokenizer.json")
DEFAULT_RUNNER = Path("/home/radxa/a733_npu_driver/build/npu_lm_runner")
DEFAULT_VIP_LIB = Path("/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0")


def bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    visible += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    chars = visible[:]
    offset = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + offset)
            offset += 1
    encoder = dict(zip(visible, (chr(char) for char in chars)))
    decoder = {char: byte for byte, char in encoder.items()}
    return encoder, decoder


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def split_special(text: str, specials: dict[str, int]) -> Iterable[tuple[str, Optional[int]]]:
    ordered = sorted(specials.items(), key=lambda item: len(item[0]), reverse=True)
    pos = 0
    pending: list[str] = []
    while pos < len(text):
        match = None
        for content, token_id in ordered:
            if text.startswith(content, pos):
                match = (content, token_id)
                break
        if match is not None:
            if pending:
                yield "".join(pending), None
                pending = []
            yield match[0], match[1]
            pos += len(match[0])
        else:
            pending.append(text[pos])
            pos += 1
    if pending:
        yield "".join(pending), None


def split_digits(text: str) -> Iterable[str]:
    current: list[str] = []
    for char in text:
        if char.isdigit():
            if current:
                yield "".join(current)
                current = []
            yield char
        else:
            current.append(char)
    if current:
        yield "".join(current)


def bytelevel_pieces(text: str) -> Iterable[str]:
    i = 0
    while i < len(text):
        start = i
        char = text[i]

        if char == "'" and i + 1 < len(text):
            lowered = text[i + 1 : i + 3].lower()
            for suffix in ("re", "ve", "ll"):
                if lowered.startswith(suffix):
                    yield text[i : i + 1 + len(suffix)]
                    i += 1 + len(suffix)
                    break
            else:
                lowered_one = text[i + 1 : i + 2].lower()
                if lowered_one in {"s", "t", "m", "d"}:
                    yield text[i : i + 2]
                    i += 2
                else:
                    yield char
                    i += 1
            continue

        if char == " " and i + 1 < len(text) and not text[i + 1].isspace():
            i += 1
            nxt = text[i]
            if nxt.isalpha():
                while i < len(text) and text[i].isalpha():
                    i += 1
                yield text[start:i]
                continue
            if nxt.isdigit():
                while i < len(text) and text[i].isdigit():
                    i += 1
                yield text[start:i]
                continue
            if not nxt.isalnum() and not nxt.isspace():
                while i < len(text) and (not text[i].isalnum()) and (not text[i].isspace()):
                    i += 1
                yield text[start:i]
                continue
            yield " "
            continue

        if char.isalpha():
            while i < len(text) and text[i].isalpha():
                i += 1
            yield text[start:i]
            continue

        if char.isdigit():
            while i < len(text) and text[i].isdigit():
                i += 1
            yield text[start:i]
            continue

        if char.isspace():
            while i < len(text) and text[i].isspace():
                i += 1
            yield text[start:i]
            continue

        while i < len(text) and (not text[i].isalnum()) and (not text[i].isspace()):
            i += 1
        yield text[start:i]


class SmolTokenizer:
    def __init__(self, tokenizer_json: Path) -> None:
        data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        model = data["model"]
        self.vocab: dict[str, int] = {str(token): int(idx) for token, idx in model["vocab"].items()}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.specials = {str(item["content"]): int(item["id"]) for item in data.get("added_tokens", [])}
        self.special_ids = {token_id: content for content, token_id in self.specials.items()}
        self.byte_encoder, self.byte_decoder = bytes_to_unicode()
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            if isinstance(merge, str):
                left, right = merge.split()
            else:
                left, right = merge
            self.bpe_ranks[(left, right)] = rank
        self.cache: dict[str, list[int]] = {}

    def bpe(self, token: str) -> list[int]:
        if token in self.cache:
            return self.cache[token]
        byte_token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
        if byte_token in self.vocab:
            ids = [self.vocab[byte_token]]
            self.cache[token] = ids
            return ids

        word = tuple(byte_token)
        if len(word) <= 1:
            ids = [self.vocab[byte_token]]
            self.cache[token] = ids
            return ids

        while True:
            pairs = get_pairs(word)
            if not pairs:
                break
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, 10**12))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break

        ids = [self.vocab[piece] for piece in word]
        self.cache[token] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk, special_id in split_special(text, self.specials):
            if special_id is not None:
                ids.append(special_id)
                continue
            for digit_chunk in split_digits(chunk):
                for piece in bytelevel_pieces(digit_chunk):
                    if piece:
                        ids.extend(self.bpe(piece))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        byte_buffer: list[int] = []

        def flush_bytes() -> None:
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in self.special_ids:
                if skip_special:
                    continue
                flush_bytes()
                pieces.append(self.special_ids[token_id])
                continue
            token = self.id_to_token.get(token_id)
            if token is None:
                continue
            for char in token:
                byte = self.byte_decoder.get(char)
                if byte is not None:
                    byte_buffer.append(byte)
                else:
                    flush_bytes()
                    pieces.append(char)
        flush_bytes()
        return "".join(pieces)


def render_chat(messages: list[tuple[str, str]], system: Optional[str]) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for role, content in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def parse_token_ids(line: str) -> list[int]:
    return [int(part) for part in line.replace(",", " ").split() if part]


def run_npu(args: argparse.Namespace, prompt_ids: list[int]) -> tuple[list[int], str]:
    if len(prompt_ids) > args.seq_len:
        window = prompt_ids[-args.seq_len :]
    else:
        window = [args.pad_token] * (args.seq_len - len(prompt_ids)) + prompt_ids

    env = os.environ.copy()
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{args.vip_lib}:{current_ld}" if current_ld else str(args.vip_lib)

    command = [
        str(args.runner),
        "--model-dir",
        str(args.model_dir),
        "--steps",
        str(args.steps),
        "--prompt",
        " ".join(str(token) for token in window),
        "--seq-len",
        str(args.seq_len),
        "--vocab",
        str(args.vocab),
    ]
    result = subprocess.run(command, env=env, check=False, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if args.verbose_runner:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"NPU runner failed with exit code {result.returncode}\n{output}")

    final_line = None
    for line in output.splitlines():
        if line.startswith("final_tokens="):
            final_line = line.split("=", 1)[1]
    if final_line is None:
        raise RuntimeError(f"NPU runner did not print final_tokens\n{output}")
    return parse_token_ids(final_line), output


def extract_metrics(runner_output: str) -> list[str]:
    wanted = ("mean_wall_us=", "mean_profile_us=", "mean_tok_s=", "peak_rss_kb=")
    return [line for line in runner_output.splitlines() if line.startswith(wanted)]


def answer_once(tokenizer: SmolTokenizer, args: argparse.Namespace, messages: list[tuple[str, str]]) -> str:
    prompt = render_chat(messages, args.system)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > args.seq_len:
        print(f"[context] prompt has {len(prompt_ids)} tokens; using the last {args.seq_len}", file=sys.stderr)
    final_tokens, runner_output = run_npu(args, prompt_ids)
    generated = final_tokens[args.seq_len :]
    trimmed: list[int] = []
    for token in generated:
        if token in args.stop_token:
            break
        trimmed.append(token)
    answer = tokenizer.decode(trimmed, skip_special=True).strip()
    if args.show_tokens:
        print("prompt_ids:", " ".join(str(token) for token in prompt_ids))
        print("generated_ids:", " ".join(str(token) for token in generated))
    if args.show_metrics:
        for line in extract_metrics(runner_output):
            print(line, file=sys.stderr)
    return answer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="*", help="single user message; omit for interactive mode")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--vip-lib", type=Path, default=DEFAULT_VIP_LIB)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=49152)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--pad-token", type=int, default=2)
    parser.add_argument("--stop-token", type=int, action="append", default=[2, 0])
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--verbose-runner", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_system:
        args.system = None

    for path, label in ((args.tokenizer, "tokenizer"), (args.runner, "runner"), (args.model_dir, "model dir")):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2
    if not os.access(args.runner, os.X_OK):
        print(f"runner is not executable: {args.runner}", file=sys.stderr)
        print("Build it first: bash scripts/board/build-npu-lm-runner.sh", file=sys.stderr)
        return 2

    tokenizer = SmolTokenizer(args.tokenizer)
    messages: list[tuple[str, str]] = []
    if args.message:
        user_text = " ".join(args.message)
        messages.append(("user", user_text))
        print(f"User: {user_text}")
        print("Assistant:", flush=True)
        print(answer_once(tokenizer, args, messages))
        return 0

    print("SmolLM2 A733 NPU chat. Ctrl-D or /exit to quit.")
    while True:
        try:
            user_text = input("\nUser: ").strip()
        except EOFError:
            print()
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        messages.append(("user", user_text))
        print("Assistant:", flush=True)
        answer = answer_once(tokenizer, args, messages)
        print(answer)
        messages.append(("assistant", answer))


if __name__ == "__main__":
    raise SystemExit(main())
