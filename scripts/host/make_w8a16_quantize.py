#!/usr/bin/env python3
"""Build an ACUITY W8A16 seed quantize table from an int16 table.

The seed keeps activation/output tensors at int16 dynamic fixed point and only
changes transformer fullconnect weights to per-channel int8. The logits
projection is kept int16 by default to match the T7 W8A16 hypothesis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


ENTRY_RE = re.compile(r"^  '([^']+)':\n?$")
REF_RE = re.compile(r"@([^:]+):")


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


def block_values(block: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in block[1:]:
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key] = value.strip()
    return values


def qparam_scale(block: list[str]) -> float:
    values = block_values(block)
    quantizer = values.get("quantizer", "")
    if quantizer == "dynamic_fixed_point":
        return float(2.0 ** (-int(values["fl"])))
    if "scale" in values:
        return float(values["scale"])
    raise SystemExit(f"cannot derive scalar scale from block: {block[0].strip()}")


def consumers_for(layers: dict[str, Any]) -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for name, info in layers.items():
        for item in info.get("inputs", []) or []:
            match = REF_RE.match(item)
            if match:
                consumers.setdefault(match.group(1), []).append(name)
    return consumers


def deep_consumers(name: str, consumers: dict[str, list[str]], depth: int = 4) -> list[str]:
    seen: set[str] = set()
    frontier = [name]
    output: list[str] = []
    for _ in range(depth):
        next_frontier: list[str] = []
        for item in frontier:
            for consumer in consumers.get(item, []):
                if consumer in seen:
                    continue
                seen.add(consumer)
                output.append(consumer)
                next_frontier.append(consumer)
        frontier = next_frontier
    return output


def classify_fullconnect(name: str, consumers: dict[str, list[str]]) -> tuple[str, int | None]:
    downstream = " ".join(deep_consumers(name, consumers))
    if "attach_logits" in downstream:
        return "lm_head", None
    patterns = (
        (r"layer(\d+)_mlp_resid", "down"),
        (r"layer(\d+)_attn_resid", "o"),
        (r"layer(\d+)_gated", "up"),
        (r"layer(\d+)_gate_sigmoid", "gate"),
        (r"layer(\d+)_q_", "q"),
        (r"layer(\d+)_k_", "k"),
        (r"layer(\d+)_v_", "v"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, downstream)
        if match:
            return kind, int(match.group(1))
    raise SystemExit(f"could not classify {name}; downstream={downstream!r}")


def initializer_name(kind: str, layer: int | None) -> str:
    if kind == "lm_head":
        return "lm_head"
    if layer is None:
        raise SystemExit(f"layer is required for {kind}")
    names = {
        "q": "wq",
        "k": "wk",
        "v": "wv",
        "o": "wo",
        "gate": "w_gate",
        "up": "w_up",
        "down": "w_down",
    }
    return f"layer{layer}_{names[kind]}"


def load_initializers(path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(path))
    return {tensor.name: numpy_helper.to_array(tensor).astype(np.float32, copy=False) for tensor in model.graph.initializer}


def weight_scales(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise SystemExit(f"expected 2D fullconnect weight, got shape {value.shape}")
    scales = np.max(np.abs(value), axis=0).astype(np.float64) / 127.0
    scales[~np.isfinite(scales)] = 0.0
    return np.maximum(scales, np.finfo(np.float32).tiny)


def scale_block(name: str, qtype: str, scales: np.ndarray, channel_dim: int) -> list[str]:
    lines = [
        f"  '{name}':\n",
        f"    qtype: {qtype}\n",
        "    quantizer: perchannel_symmetric_affine\n",
        "    rounding: rtne\n",
        "    scale:\n",
    ]
    lines.extend(f"    - {float(scale)}\n" for scale in scales)
    lines.append(f"    channel_dim: {channel_dim}\n")
    return lines


def ref_name(value: str) -> str:
    match = REF_RE.match(value)
    if not match:
        raise SystemExit(f"bad tensor reference: {value}")
    return f"@{match.group(1)}:out0"


def parse_layer_set(value: str | None) -> set[int]:
    if not value:
        return set()
    layers: set[int] = set()
    for item in value.replace(",", " ").split():
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            if start > end:
                raise SystemExit(f"bad layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int16-quantize", required=True, type=Path)
    parser.add_argument("--acuity-json", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force-int16-layers",
        help="comma/space separated layer numbers or ranges to leave at int16, e.g. '0,1,22-23'",
    )
    parser.add_argument("--quantize-lm-head", action="store_true", help="also quantize the tied logits projection")
    args = parser.parse_args()

    lines = read_lines(args.int16_quantize)
    blocks = block_map(lines)
    graph = json.loads(args.acuity_json.read_text(encoding="ascii"))["Layers"]
    consumers = consumers_for(graph)
    initializers = load_initializers(args.onnx)
    force_int16_layers = parse_layer_set(args.force_int16_layers)

    fc_info: dict[str, dict[str, Any]] = {}
    for name, info in graph.items():
        if info.get("op") != "fullconnect":
            continue
        kind, layer = classify_fullconnect(name, consumers)
        should_quantize = kind != "lm_head" or args.quantize_lm_head
        if layer in force_int16_layers:
            should_quantize = False
        init_name = initializer_name(kind, layer)
        if should_quantize and init_name not in initializers:
            raise SystemExit(f"missing ONNX initializer {init_name!r} for {name}")
        fc_info[name] = {
            "kind": kind,
            "layer": layer,
            "quantize": should_quantize,
            "scales": weight_scales(initializers[init_name]) if should_quantize else None,
            "input_key": ref_name(info["inputs"][0]),
        }

    output: list[str] = []
    start, end = find_section(lines, "quantize_parameters")
    output.extend(lines[: start + 1])
    index = start + 1
    replaced_weights = 0
    replaced_biases = 0
    kept_fullconnect_parts = 0

    while index < end:
        match = ENTRY_RE.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        key = match.group(1)
        next_index = index + 1
        while next_index < end and not ENTRY_RE.match(lines[next_index]):
            next_index += 1
        block = lines[index:next_index]

        fullconnect = re.match(r"@(fullconnect_\d+):(weight|bias)$", key)
        if fullconnect:
            name, part = fullconnect.groups()
            info = fc_info.get(name)
            if info is None or not info["quantize"]:
                output.extend(block)
                kept_fullconnect_parts += 1
            elif part == "weight":
                output.extend(scale_block(key, "i8", info["scales"], -1))
                replaced_weights += 1
            else:
                input_block = blocks.get(info["input_key"])
                if input_block is None:
                    raise SystemExit(f"missing input activation qparam for {name}: {info['input_key']}")
                output.extend(scale_block(key, "i32", info["scales"] * qparam_scale(input_block), 0))
                replaced_biases += 1
        else:
            output.extend(block)
        index = next_index

    output.extend(lines[end:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as handle:
        handle.write("".join(output))

    summary = {
        "int16_quantize": str(args.int16_quantize),
        "acuity_json": str(args.acuity_json),
        "onnx": str(args.onnx),
        "output": str(args.output),
        "force_int16_layers": sorted(force_int16_layers),
        "quantize_lm_head": args.quantize_lm_head,
        "fullconnects": len(fc_info),
        "replaced_weights": replaced_weights,
        "replaced_biases": replaced_biases,
        "kept_fullconnect_parts": kept_fullconnect_parts,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
