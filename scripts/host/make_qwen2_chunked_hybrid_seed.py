#!/usr/bin/env python3
"""Create an ACUITY hybrid seed for chunked Qwen BF16/int16 trials."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENTRY_RE = re.compile(r"^  '([^']+)':\n?$")
LAYER_RE = re.compile(r"^  ([^:]+): .+\n?$")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").splitlines(keepends=True)


def find_section_or_inline_empty(lines: list[str], name: str) -> tuple[int, int]:
    header = f"{name}:\n"
    inline = f"{name}: {{}}\n"
    for index, line in enumerate(lines):
        if line == header:
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                if lines[candidate].strip() and not lines[candidate].startswith(" "):
                    end = candidate
                    break
            return index, end
        if line == inline:
            return index, index + 1
    raise SystemExit(f"missing section {name!r}")


def block_map(lines: list[str]) -> dict[str, list[str]]:
    _, end = find_section_or_inline_empty(lines, "quantize_parameters")
    blocks: dict[str, list[str]] = {}
    index = 0
    while index < end:
        match = ENTRY_RE.match(lines[index])
        if not match:
            index += 1
            continue
        key = match.group(1)
        next_index = index + 1
        while next_index < end and not ENTRY_RE.match(lines[next_index]):
            next_index += 1
        blocks[key] = lines[index:next_index]
        index = next_index
    return blocks


def replace_qparams(base_lines: list[str], donor_blocks: dict[str, list[str]], prefixes: set[str]) -> tuple[list[str], int, int]:
    start, end = find_section_or_inline_empty(base_lines, "quantize_parameters")
    output = base_lines[: start + 1]
    index = start + 1
    seen: set[str] = set()
    replaced = 0

    while index < end:
        match = ENTRY_RE.match(base_lines[index])
        if not match:
            output.append(base_lines[index])
            index += 1
            continue
        key = match.group(1)
        seen.add(key)
        next_index = index + 1
        while next_index < end and not ENTRY_RE.match(base_lines[next_index]):
            next_index += 1
        if any(key.startswith(prefix) for prefix in prefixes) and key in donor_blocks:
            output.extend(donor_blocks[key])
            replaced += 1
        else:
            output.extend(base_lines[index:next_index])
        index = next_index

    added = 0
    for key in sorted(donor_blocks):
        if key not in seen and any(key.startswith(prefix) for prefix in prefixes):
            output.extend(donor_blocks[key])
            added += 1
    output.extend(base_lines[end:])
    return output, replaced, added


def rewrite_layers(lines: list[str], bf16_layers: set[str]) -> tuple[list[str], int, int]:
    start, end = find_section_or_inline_empty(lines, "customized_quantize_layers")
    existing: dict[str, str] = {}
    removed = 0
    for line in lines[start + 1 : end]:
        match = LAYER_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name in bf16_layers:
            removed += 1
            continue
        existing[name] = line

    for name in bf16_layers:
        existing[name] = f"  {name}: bfloat16-bfloat16\n"

    output = lines[:start]
    output.append("customized_quantize_layers:\n")
    output.extend(existing[name] for name in sorted(existing))
    output.extend(lines[end:])
    return output, len(bf16_layers), removed


def entropy_layers(path: Path, count: int) -> list[str]:
    pairs: list[tuple[float, str]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if "," not in line:
            continue
        name, value = line.split(",", 1)
        name = name.strip()
        if not name.endswith(":weight"):
            continue
        layer = name[1:].split(":", 1)[0] if name.startswith("@") else name.split(":", 1)[0]
        try:
            pairs.append((float(value.strip()), layer))
        except ValueError:
            continue
    pairs.sort(reverse=True)
    return [layer for _, layer in pairs[:count]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int16", required=True, type=Path)
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--entropy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--entropy-count", type=int, default=8)
    parser.add_argument(
        "--lm-head-layer",
        action="append",
        default=[],
        help="lm_head chunk layer name to force to BF16; repeat for all chunks",
    )
    args = parser.parse_args()

    int16_lines = read_lines(args.int16)
    bf16_blocks = block_map(read_lines(args.bf16))
    selected_layers = set(args.lm_head_layer)
    selected_layers.update(entropy_layers(args.entropy, args.entropy_count))
    prefixes = {f"@{name}:" for name in selected_layers}
    prefixes.update({f"@{name}" for name in selected_layers})

    mixed, replaced, added = replace_qparams(int16_lines, bf16_blocks, prefixes)
    mixed, bf16_layer_count, replaced_layer_count = rewrite_layers(mixed, selected_layers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("".join(mixed))
    summary = {
        "int16": str(args.int16),
        "bf16": str(args.bf16),
        "entropy": str(args.entropy),
        "output": str(args.output),
        "entropy_count": args.entropy_count,
        "selected_layers": sorted(selected_layers),
        "qparam_prefixes": sorted(prefixes),
        "qparams_replaced": replaced,
        "qparams_added": added,
        "bf16_customized_layers": bf16_layer_count,
        "replaced_existing_layers": replaced_layer_count,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
