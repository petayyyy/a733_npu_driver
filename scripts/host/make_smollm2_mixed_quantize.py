#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


ENTRY_RE = re.compile(r"^  '([^']+)':\n?$")
LAYER_RE = re.compile(r"^  ([^:]+): .+\n?$")

CRITICAL_QPARAM_PREFIXES = (
    "@attach_logits/out0_0:",
    "@final_last_token_2:",
    "@final_rms",
    "@reshape_2278:",
    "@fullconnect_2279:",
    "@reshape_2280:",
    "@hidden0_1883:",
    "@token_embed_1920:",
)

CRITICAL_LAYER_PREFIXES = (
    "final_last_token_2",
    "final_rms",
    "reshape_2278",
    "fullconnect_2279",
    "hidden0_1883",
)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").splitlines(keepends=True)


def find_section(lines: list[str], name: str) -> tuple[int, int]:
    header = f"{name}:\n"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise SystemExit(f"missing section {name!r}") from exc

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith(" "):
            end = index
            break
    return start, end


def block_map(lines: list[str]) -> dict[str, list[str]]:
    _, end = find_section(lines, "quantize_parameters")
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


def is_critical_qparam(key: str) -> bool:
    return key.startswith(CRITICAL_QPARAM_PREFIXES)


def is_critical_layer(name: str) -> bool:
    return name.startswith(CRITICAL_LAYER_PREFIXES)


def replace_qparams(pcq_lines: list[str], int16_blocks: dict[str, list[str]]) -> tuple[list[str], int]:
    start, end = find_section(pcq_lines, "quantize_parameters")
    output = pcq_lines[: start + 1]
    index = start + 1
    copied = 0

    while index < end:
        match = ENTRY_RE.match(pcq_lines[index])
        if not match:
            output.append(pcq_lines[index])
            index += 1
            continue

        key = match.group(1)
        next_index = index + 1
        while next_index < end and not ENTRY_RE.match(pcq_lines[next_index]):
            next_index += 1

        if is_critical_qparam(key) and key in int16_blocks:
            output.extend(int16_blocks[key])
            copied += 1
        else:
            output.extend(pcq_lines[index:next_index])
        index = next_index

    output.extend(pcq_lines[end:])
    return output, copied


def layer_map(lines: list[str]) -> dict[str, str]:
    start, end = find_section(lines, "customized_quantize_layers")
    layers: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        match = LAYER_RE.match(line)
        if match:
            layers[match.group(1)] = line
    return layers


def replace_layers(lines: list[str], int16_layers: dict[str, str]) -> tuple[list[str], int]:
    start, end = find_section(lines, "customized_quantize_layers")
    output = lines[: start + 1]
    copied = 0

    for line in lines[start + 1 : end]:
        match = LAYER_RE.match(line)
        if match and is_critical_layer(match.group(1)) and match.group(1) in int16_layers:
            output.append(int16_layers[match.group(1)])
            copied += 1
        else:
            output.append(line)

    output.extend(lines[end:])
    return output, copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SmolLM2 mixed-precision quantize table: pcq for transformer "
            "linear weights, int16 for embedding/final RMSNorm/lm_head/logits."
        )
    )
    parser.add_argument("--pcq", required=True, type=Path, help="source pcq .quantize")
    parser.add_argument("--int16", required=True, type=Path, help="source int16 .quantize")
    parser.add_argument("--output", required=True, type=Path, help="output mixed .quantize")
    args = parser.parse_args()

    pcq_lines = read_lines(args.pcq)
    int16_lines = read_lines(args.int16)

    mixed_lines, qparam_count = replace_qparams(pcq_lines, block_map(int16_lines))
    mixed_lines, layer_count = replace_layers(mixed_lines, layer_map(int16_lines))

    if qparam_count == 0:
        raise SystemExit("no quantize_parameter entries were copied from int16")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as handle:
        handle.write("".join(mixed_lines))
    print(f"wrote {args.output}")
    print(f"copied int16 quantize_parameters: {qparam_count}")
    print(f"copied int16 customized_quantize_layers: {layer_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
