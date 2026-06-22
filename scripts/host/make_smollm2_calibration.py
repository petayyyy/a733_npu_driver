#!/usr/bin/env python3
"""Create representative fixed-window token calibration data for SmolLM2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from smollm2_tokenizer import DEFAULT_SYSTEM, render_chat


RAW_PROMPTS = [
    "The capital of France is",
    "Question: What is the capital of France? Answer: The capital of France is",
    "Python is a programming language that",
    "The quick brown fox jumps over the lazy",
    "In a small village near the river,",
    "A neural processing unit accelerates matrix",
]

CHAT_PROMPTS = [
    "What is the capital of France?",
    "Write one short sentence about Paris.",
    "Name two colors in the French flag.",
    "What is 2 plus 2?",
    "Complete this phrase: machine learning is",
    "Answer briefly: why is the sky blue?",
]


def fixed_window(ids: list[int], seq_len: int, pad_token: int) -> np.ndarray:
    if len(ids) < seq_len:
        ids = [pad_token] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]
    return np.asarray([ids], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("work/models/smollm2-135m-instruct"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--pad-token", type=int, default=2)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []
    descriptions: list[str] = []

    for index, text in enumerate(RAW_PROMPTS):
        ids = tokenizer.encode(text).ids
        filename = f"token_ids_raw_{index:02d}.npy"
        np.save(args.output_dir / filename, fixed_window(ids, args.seq_len, args.pad_token))
        entries.append(filename)
        descriptions.append(f"{filename}: raw: {text}")

    for index, user in enumerate(CHAT_PROMPTS):
        text = render_chat(user, DEFAULT_SYSTEM, True)
        ids = tokenizer.encode(text).ids
        filename = f"token_ids_chat_{index:02d}.npy"
        np.save(args.output_dir / filename, fixed_window(ids, args.seq_len, args.pad_token))
        entries.append(filename)
        descriptions.append(f"{filename}: chat: {user}")

    (args.output_dir / "dataset.txt").write_text("\n".join(entries) + "\n", encoding="ascii")
    (args.output_dir / "tokens.txt").write_text(
        "calibration windows are stored in token_ids_*.npy\n",
        encoding="ascii",
    )
    (args.output_dir / "calibration_manifest.txt").write_text(
        "\n".join(descriptions) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} calibration windows to {args.output_dir}")


if __name__ == "__main__":
    main()
