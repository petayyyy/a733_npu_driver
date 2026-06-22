#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np


ENTRY_RE = re.compile(r"^  '([^']+)':\n?$")
REF_RE = re.compile(r"@([^:]+):")


class SafeTensorReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("rb")
        header_len = struct.unpack("<Q", self.handle.read(8))[0]
        self.header = json.loads(self.handle.read(header_len))
        self.data_start = 8 + header_len

    def close(self) -> None:
        self.handle.close()

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.header:
            raise KeyError(f"missing tensor in {self.path}: {name}")
        info = self.header[name]
        dtype = str(info["dtype"]).upper()
        shape = tuple(int(dim) for dim in info["shape"])
        start, end = (int(value) for value in info["data_offsets"])
        self.handle.seek(self.data_start + start)
        data = self.handle.read(end - start)

        if dtype == "BF16":
            raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            return (raw << 16).view(np.float32).reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(shape)
        if dtype == "F32":
            return np.frombuffer(data, dtype="<f4").reshape(shape).copy()
        raise ValueError(f"unsupported tensor dtype for {name}: {dtype}")


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


def activation_qparams(block: list[str]) -> tuple[float, list[str]]:
    values = block_values(block)
    max_value = float(values.get("max_value", "1.0"))
    min_value = float(values.get("min_value", "0.0"))
    scale = (max_value - min_value) / 255.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(abs(max_value), abs(min_value), 1.0) / 127.0
    zero_point = int(round(-128.0 - min_value / scale))
    zero_point = max(-128, min(127, zero_point))
    key = ENTRY_RE.match(block[0])
    if not key:
        raise SystemExit(f"bad quantize block header: {block[0]!r}")
    name = key.group(1)
    lines = [
        f"  '{name}':\n",
        "    qtype: i8\n",
        "    quantizer: asymmetric_affine\n",
        "    rounding: rtne\n",
        f"    max_value: {max_value}\n",
        f"    min_value: {min_value}\n",
        f"    scale: {scale}\n",
        f"    zero_point: {zero_point}\n",
    ]
    return scale, lines


def consumers_for(layers: dict[str, Any]) -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for name, info in layers.items():
        for item in info.get("inputs", []) or []:
            match = REF_RE.match(item)
            if match:
                consumers.setdefault(match.group(1), []).append(name)
    return consumers


def deep_consumers(name: str, consumers: dict[str, list[str]]) -> list[str]:
    output: list[str] = []
    for first in consumers.get(name, []):
        output.append(first)
        output.extend(consumers.get(first, []))
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


def tensor_for(reader: SafeTensorReader, kind: str, layer: int | None) -> np.ndarray:
    if kind == "lm_head":
        return reader.tensor("model.embed_tokens.weight").T
    if layer is None:
        raise SystemExit(f"layer is required for {kind}")
    prefix = f"model.layers.{layer}"
    names = {
        "q": "self_attn.q_proj.weight",
        "k": "self_attn.k_proj.weight",
        "v": "self_attn.v_proj.weight",
        "o": "self_attn.o_proj.weight",
        "gate": "mlp.gate_proj.weight",
        "up": "mlp.up_proj.weight",
        "down": "mlp.down_proj.weight",
    }
    return reader.tensor(f"{prefix}.{names[kind]}").T


def weight_scales(value: np.ndarray) -> np.ndarray:
    scales = np.max(np.abs(value), axis=0).astype(np.float64) / 127.0
    scales[~np.isfinite(scales)] = 0.0
    scales = np.maximum(scales, np.finfo(np.float32).tiny)
    return scales


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a synthetic full-Qwen pcq .quantize seed from ACUITY int16 import metadata."
    )
    parser.add_argument("--int16-quantize", required=True, type=Path)
    parser.add_argument("--acuity-json", required=True, type=Path)
    parser.add_argument("--safetensors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lines = read_lines(args.int16_quantize)
    blocks = block_map(lines)
    graph = json.loads(args.acuity_json.read_text(encoding="ascii"))["Layers"]
    consumers = consumers_for(graph)
    reader = SafeTensorReader(args.safetensors)

    fc_info: dict[str, dict[str, Any]] = {}
    for name, info in graph.items():
        if info.get("op") != "fullconnect":
            continue
        kind, layer = classify_fullconnect(name, consumers)
        tensor = tensor_for(reader, kind, layer)
        weight_key = f"@{name}:weight"
        if weight_key not in blocks:
            raise SystemExit(f"missing int16 weight qparam for {name}")
        expected = int(info.get("parameters", {}).get("weights", tensor.shape[1]))
        if tensor.shape[1] != expected:
            raise SystemExit(f"{name} mapped to {kind}/{layer} has output {tensor.shape[1]}, expected {expected}")
        input_key = ref_name(info["inputs"][0])
        fc_info[name] = {
            "weight_scales": weight_scales(tensor),
            "input_key": input_key,
            "kind": kind,
            "layer": layer,
        }

    reader.close()

    output: list[str] = []
    activation_scales: dict[str, float] = {}
    start, end = find_section(lines, "quantize_parameters")
    output.extend(lines[: start + 1])
    index = start + 1
    replaced_weights = 0
    replaced_biases = 0
    converted_activations = 0

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
            info = fc_info[name]
            if part == "weight":
                output.extend(scale_block(key, "i8", info["weight_scales"], -1))
                replaced_weights += 1
            else:
                input_scale = activation_scales.get(info["input_key"])
                if input_scale is None:
                    input_block = blocks.get(info["input_key"])
                    if input_block is None:
                        raise SystemExit(f"missing activation qparam for {name} input {info['input_key']}")
                    input_scale, _ = activation_qparams(input_block)
                output.extend(scale_block(key, "i32", info["weight_scales"] * input_scale, 0))
                replaced_biases += 1
        else:
            scale, converted = activation_qparams(block)
            output.extend(converted)
            activation_scales[key] = scale
            converted_activations += 1
        index = next_index

    output.extend(lines[end:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as handle:
        handle.write("".join(output))

    print(f"wrote {args.output}")
    print(f"fullconnects: {len(fc_info)}")
    print(f"replaced weights: {replaced_weights}")
    print(f"replaced biases: {replaced_biases}")
    print(f"converted activation/constant qparams: {converted_activations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
