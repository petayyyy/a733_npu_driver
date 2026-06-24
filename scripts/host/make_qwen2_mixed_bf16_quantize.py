#!/usr/bin/env python3
"""Build a Qwen mixed BF16/int16 ACUITY quantize seed.

The seed starts from the BF16 table that passed the T9 host-quality gate, then
copies the embedding and final big-vocab logits/lm_head path back from the
known-exportable int16 table. This is meant to test whether avoiding BF16 on
the 151936-wide projection removes the BF16 VerifyGraph -3 blocker while
preserving BF16 in the outlier-heavy transformer body.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENTRY_RE = re.compile(r"^  '([^']+)':\n?$")
LAYER_RE = re.compile(r"^  ([^:]+): .+\n?$")
OUTLIER_NAME_PARTS = (
    "rms_squared",
    "rms_mean_eps",
    "scores_masked",
    "_scores_",
    "_gated_",
)

DEFAULT_QPARAM_PREFIXES = (
    "@attach_logits/out0_0:",
    "@fullconnect_1973:",
    "@hidden0_1554:",
    "@token_embed_1593:",
)

DEFAULT_LAYER_PREFIXES = (
    "fullconnect_1973",
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


def find_section_or_inline_empty(lines: list[str], name: str) -> tuple[int, int]:
    try:
        return find_section(lines, name)
    except SystemExit:
        inline = f"{name}: {{}}\n"
        try:
            start = lines.index(inline)
        except ValueError as exc:
            raise SystemExit(f"missing section {name!r}") from exc
        return start, start + 1


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


def block_values(block: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in block[1:]:
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def collect_qwen_outlier_prefixes(
    int16_blocks: dict[str, list[str]],
    min_abs: float,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, object]]]:
    qparam_prefixes: set[str] = set()
    layer_prefixes: set[str] = set()
    selected: list[dict[str, object]] = []
    for key, block in int16_blocks.items():
        if not key.endswith(":out0"):
            continue
        name = key[1:].split(":", 1)[0] if key.startswith("@") else key.split(":", 1)[0]
        if not any(part in name for part in OUTLIER_NAME_PARTS):
            continue
        values = block_values(block)
        if values.get("qtype") != "i16":
            continue
        try:
            max_abs = max(abs(float(values.get("max_value", "0"))), abs(float(values.get("min_value", "0"))))
        except ValueError:
            continue
        if max_abs < min_abs:
            continue
        qparam_prefixes.add(f"@{name}")
        layer_prefixes.add(name)
        selected.append({"name": name, "max_abs": max_abs, "fl": values.get("fl")})

    selected.sort(key=lambda item: (-float(item["max_abs"]), str(item["name"])))
    return tuple(sorted(qparam_prefixes)), tuple(sorted(layer_prefixes)), selected


def layer_map(lines: list[str]) -> dict[str, str]:
    start, end = find_section(lines, "customized_quantize_layers")
    layers: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        match = LAYER_RE.match(line)
        if match:
            layers[match.group(1)] = line
    return layers


def is_critical_qparam(key: str, prefixes: tuple[str, ...]) -> bool:
    return key.startswith(prefixes)


def is_critical_layer(name: str, prefixes: tuple[str, ...]) -> bool:
    return name.startswith(prefixes)


def replace_and_add_qparams(
    bf16_lines: list[str],
    int16_blocks: dict[str, list[str]],
    prefixes: tuple[str, ...],
) -> tuple[list[str], int, int]:
    start, end = find_section(bf16_lines, "quantize_parameters")
    output = bf16_lines[: start + 1]
    index = start + 1
    replaced = 0
    seen: set[str] = set()

    while index < end:
        match = ENTRY_RE.match(bf16_lines[index])
        if not match:
            output.append(bf16_lines[index])
            index += 1
            continue

        key = match.group(1)
        seen.add(key)
        next_index = index + 1
        while next_index < end and not ENTRY_RE.match(bf16_lines[next_index]):
            next_index += 1

        if is_critical_qparam(key, prefixes) and key in int16_blocks:
            output.extend(int16_blocks[key])
            replaced += 1
        else:
            output.extend(bf16_lines[index:next_index])
        index = next_index

    added = 0
    for key in sorted(int16_blocks):
        if is_critical_qparam(key, prefixes) and key not in seen:
            output.extend(int16_blocks[key])
            added += 1

    output.extend(bf16_lines[end:])
    return output, replaced, added


def replace_customized_layers(
    lines: list[str],
    int16_layers: dict[str, str],
    prefixes: tuple[str, ...],
) -> tuple[list[str], int]:
    start, end = find_section_or_inline_empty(lines, "customized_quantize_layers")
    critical_lines = [line for name, line in sorted(int16_layers.items()) if is_critical_layer(name, prefixes)]
    output = lines[:start]
    if critical_lines:
        output.append("customized_quantize_layers:\n")
        output.extend(critical_lines)
    else:
        output.extend(lines[start:end])
    output.extend(lines[end:])
    return output, len(critical_lines)


def remove_customized_layers(lines: list[str], prefixes: tuple[str, ...]) -> tuple[list[str], int]:
    if not prefixes:
        return lines, 0
    start, end = find_section_or_inline_empty(lines, "customized_quantize_layers")
    kept: list[str] = []
    removed = 0
    for line in lines[start + 1 : end]:
        match = LAYER_RE.match(line)
        if match and is_critical_layer(match.group(1), prefixes):
            removed += 1
            continue
        kept.append(line)

    output = lines[:start]
    if kept:
        output.append("customized_quantize_layers:\n")
        output.extend(kept)
    else:
        output.append("customized_quantize_layers: {}\n")
    output.extend(lines[end:])
    return output, removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True, type=Path, help="source BF16 .quantize")
    parser.add_argument("--int16", required=True, type=Path, help="source int16 .quantize")
    parser.add_argument("--output", required=True, type=Path, help="output mixed .quantize")
    parser.add_argument(
        "--base",
        choices=["bf16", "int16"],
        default="bf16",
        help="base table to start from; donor qparams are copied from the other table",
    )
    parser.add_argument(
        "--qparam-prefix",
        action="append",
        default=None,
        help="copy qparams whose key starts with this prefix; may be repeated",
    )
    parser.add_argument(
        "--layer-prefix",
        action="append",
        default=None,
        help="copy customized layer entries whose name starts with this prefix; may be repeated",
    )
    parser.add_argument(
        "--drop-layer-prefix",
        action="append",
        default=None,
        help="drop customized layer entries whose name starts with this prefix; useful with --base int16",
    )
    parser.add_argument(
        "--auto-qwen-outliers",
        action="store_true",
        help="with --base int16, copy BF16 qparams for high-range Qwen transformer outlier activations",
    )
    parser.add_argument(
        "--outlier-min-abs",
        type=float,
        default=1000.0,
        help="minimum int16 calibration abs range for --auto-qwen-outliers",
    )
    args = parser.parse_args()

    qparam_prefixes = tuple(args.qparam_prefix or DEFAULT_QPARAM_PREFIXES)
    layer_prefixes = tuple(args.layer_prefix or DEFAULT_LAYER_PREFIXES)
    drop_layer_prefixes = tuple(args.drop_layer_prefix or ())

    bf16_lines = read_lines(args.bf16)
    int16_lines = read_lines(args.int16)
    int16_blocks = block_map(int16_lines)
    selected_outliers: list[dict[str, object]] = []

    if args.auto_qwen_outliers:
        qparam_prefixes, auto_drop_prefixes, selected_outliers = collect_qwen_outlier_prefixes(
            int16_blocks,
            args.outlier_min_abs,
        )
        drop_layer_prefixes = tuple(sorted(set(drop_layer_prefixes).union(auto_drop_prefixes)))

    if args.base == "bf16":
        mixed, replaced, added = replace_and_add_qparams(bf16_lines, int16_blocks, qparam_prefixes)
        mixed, customized_layers = replace_customized_layers(mixed, layer_map(int16_lines), layer_prefixes)
        removed_layers = 0
    else:
        mixed, replaced, added = replace_and_add_qparams(int16_lines, block_map(bf16_lines), qparam_prefixes)
        mixed, removed_layers = remove_customized_layers(mixed, drop_layer_prefixes)
        customized_layers = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as handle:
        handle.write("".join(mixed))

    summary = {
        "bf16": str(args.bf16),
        "int16": str(args.int16),
        "output": str(args.output),
        "base": args.base,
        "qparam_prefixes": list(qparam_prefixes),
        "layer_prefixes": list(layer_prefixes),
        "drop_layer_prefixes": list(drop_layer_prefixes),
        "auto_qwen_outliers": args.auto_qwen_outliers,
        "outlier_min_abs": args.outlier_min_abs,
        "selected_outliers": selected_outliers,
        "critical_qparams_replaced": replaced,
        "critical_qparams_added": added,
        "critical_customized_layers": customized_layers,
        "removed_customized_layers": removed_layers,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
