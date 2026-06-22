#!/usr/bin/env python3
"""Small Qwen2.5 tokenizer/chat helper for T4 host-side validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_ids(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def render_chat(user: str, system: str = DEFAULT_SYSTEM, add_generation_prompt: bool = True) -> str:
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("work/models/qwen25-0.5b-instruct"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--chat-user")
    group.add_argument("--decode-ids")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--no-generation-prompt", action="store_true")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    if args.decode_ids:
        print(tokenizer.decode(parse_ids(args.decode_ids), skip_special_tokens=False))
        return

    text = args.text if args.text is not None else render_chat(args.chat_user, args.system, not args.no_generation_prompt)
    ids = tokenizer.encode(text).ids
    print(text)
    print(" ".join(str(token) for token in ids))


if __name__ == "__main__":
    main()
