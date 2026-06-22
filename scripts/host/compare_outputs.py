#!/usr/bin/env python3
"""Compare a board ``output_0.txt`` tensor with an ACUITY host golden tensor."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def load_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if ":" in line:
                line = line.split(":", 1)[1]
            values.extend(float(match.group(0)) for match in FLOAT_RE.finditer(line))
    if not values:
        raise SystemExit(f"no numeric values found in {path}")
    return values


def top_indices(values: list[float], k: int) -> list[int]:
    limit = min(k, len(values))
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:limit]


def format_top(values: list[float], indices: list[int]) -> str:
    return ", ".join(f"{index}:{values[index]:.8f}" for index in indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", type=Path, help="ACUITY host golden tensor text file")
    parser.add_argument("board", type=Path, help="board output_0.txt tensor text file")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    golden = load_values(args.golden)
    board = load_values(args.board)
    if len(golden) != len(board):
        raise SystemExit(f"length mismatch: golden={len(golden)} board={len(board)}")

    diffs = [abs(left - right) for left, right in zip(golden, board)]
    max_abs = max(diffs)
    mean_abs = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    dot = sum(left * right for left, right in zip(golden, board))
    norm_golden = math.sqrt(sum(value * value for value in golden))
    norm_board = math.sqrt(sum(value * value for value in board))
    cosine = dot / (norm_golden * norm_board) if norm_golden and norm_board else float("nan")

    golden_top = top_indices(golden, args.top_k)
    board_top = top_indices(board, args.top_k)
    top_match = golden_top == board_top

    print(f"length: {len(golden)}")
    print(f"top-{len(golden_top)} index match: {'yes' if top_match else 'no'}")
    print(f"golden top-{len(golden_top)}: {format_top(golden, golden_top)}")
    print(f"board top-{len(board_top)}: {format_top(board, board_top)}")
    print(f"max abs diff: {max_abs:.9f}")
    print(f"mean abs diff: {mean_abs:.9f}")
    print(f"RMSE: {rmse:.9f}")
    print(f"cosine: {cosine:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
