#!/usr/bin/env python3
"""Pack a text tensor into an NBG input .dat using nbg_meta.json."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


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


def product(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def select_input(meta: dict, input_name: str | None, input_index: int | None) -> tuple[str, dict]:
    inputs = list(meta.get("Inputs", {}).items())
    if not inputs:
        raise SystemExit("metadata has no Inputs")
    if input_name is not None:
        for key, info in inputs:
            if key == input_name or info.get("name") == input_name:
                return key, info
        raise SystemExit(f"input not found in metadata: {input_name}")
    index = 0 if input_index is None else input_index
    try:
        return inputs[index]
    except IndexError as exc:
        raise SystemExit(f"input index out of range: {index}") from exc


def write_packed(path: Path, values: list[float], info: dict) -> None:
    quant = info.get("quantize")
    dtype = str(info.get("dtype", "")).lower()

    if quant:
        qtype = str(quant.get("qtype", "")).lower()
        if qtype in {"i16", "int16"}:
            scale = 2.0 ** int(quant["fl"])

            def writer(handle, value: float) -> None:
                handle.write(struct.pack("<h", max(-32768, min(32767, int(round(value * scale))))))

        elif qtype in {"i8", "int8"}:
            scale = 2.0 ** int(quant["fl"])

            def writer(handle, value: float) -> None:
                handle.write(struct.pack("<b", max(-128, min(127, int(round(value * scale))))))

        elif qtype in {"u8", "uint8"}:
            scale = float(quant["scale"])
            zero_point = float(quant["zero_point"])

            def writer(handle, value: float) -> None:
                quantized = int(round(value / scale + zero_point))
                handle.write(struct.pack("<B", max(0, min(255, quantized))))

        else:
            raise SystemExit(f"unsupported quantized qtype: {qtype}")
    elif dtype == "int32":

        def writer(handle, value: float) -> None:
            handle.write(struct.pack("<i", int(round(value))))

    elif dtype == "float16":

        def writer(handle, value: float) -> None:
            handle.write(struct.pack("<e", float(value)))

    elif dtype in {"float", "float32", ""}:

        def writer(handle, value: float) -> None:
            handle.write(struct.pack("<f", float(value)))

    else:
        raise SystemExit(f"unsupported input dtype: {dtype}")

    with path.open("wb") as handle:
        for value in values:
            writer(handle, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--input-text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-name")
    parser.add_argument("--input-index", type=int)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="ascii"))
    key, info = select_input(meta, args.input_name, args.input_index)
    values = load_values(args.input_text)
    expected = product(info.get("shape", []))
    if expected and len(values) != expected:
        raise SystemExit(f"{key} expected {expected} values, found {len(values)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_packed(args.output, values, info)
    print(f"packed {len(values)} values for {key} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
