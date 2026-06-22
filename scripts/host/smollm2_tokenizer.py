#!/usr/bin/env python3
"""Tokenize and detokenize SmolLM2 prompts using the real HF tokenizer.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def render_chat(user: str, system: str | None, add_generation_prompt: bool) -> str:
    parts: list[str] = []
    if system is not None:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def parse_ids(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("work/models/smollm2-135m-instruct"),
        help="directory containing tokenizer.json and tokenizer_config.json",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="raw text to tokenize")
    group.add_argument("--chat-user", help="single user message rendered with the SmolLM2 chat format")
    group.add_argument("--decode", help="space- or comma-separated token ids to decode")
    parser.add_argument(
        "--system",
        default=None,
        help="system message for --chat-user; omit to use no system message",
    )
    parser.add_argument(
        "--default-system",
        action="store_true",
        help="use SmolLM2 tokenizer_config default system text for --chat-user",
    )
    parser.add_argument("--no-generation-prompt", action="store_true")
    parser.add_argument("--window", type=int, help="also print the rightmost N token ids")
    parser.add_argument("--skip-special", action="store_true", help="skip special tokens while decoding")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer_config = json.loads((args.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))

    if args.decode is not None:
        ids = parse_ids(args.decode)
        print(tokenizer.decode(ids, skip_special_tokens=args.skip_special))
        return

    if args.text is not None:
        text = args.text
    else:
        system = DEFAULT_SYSTEM if args.default_system else args.system
        text = render_chat(args.chat_user, system, not args.no_generation_prompt)

    ids = tokenizer.encode(text).ids
    print("text:")
    print(text)
    print("token_count:", len(ids))
    print("ids:")
    print(" ".join(str(token) for token in ids))
    if args.window is not None:
        window_ids = ids[-args.window :]
        print(f"window_{args.window}_count:", len(window_ids))
        print(f"window_{args.window}_ids:")
        print(" ".join(str(token) for token in window_ids))
        print(f"window_{args.window}_decoded:")
        print(tokenizer.decode(window_ids, skip_special_tokens=False))
    print("decoded:")
    print(tokenizer.decode(ids, skip_special_tokens=False))
    print("eos_token_id:", tokenizer_config.get("eos_token_id", 2))


if __name__ == "__main__":
    main()
