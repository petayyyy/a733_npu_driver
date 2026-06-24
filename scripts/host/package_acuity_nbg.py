#!/usr/bin/env python3
"""Package ACUITY host inference and NBG export artifacts for board runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def read_numbers(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if ":" in line:
                line = line.split(":", 1)[1]
            for part in line.replace(",", " ").split():
                try:
                    values.append(float(part))
                except ValueError:
                    pass
    return values


def product(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def candidates_for(inf_dir: Path, key: str, info: dict[str, Any], prefer_quantized: bool) -> list[Path]:
    needles = [sanitize(key)]
    if info.get("name"):
        needles.append(sanitize(str(info["name"])))
    files = sorted(inf_dir.glob("iter_0_*.tensor"))
    matches: list[Path] = []
    for needle in needles:
        if not needle:
            continue
        matches.extend(path for path in files if needle in path.name)
    unique = list(dict.fromkeys(matches))
    if prefer_quantized:
        unique.sort(key=lambda path: (not path.name.endswith(".qnt.tensor"), path.name))
    else:
        unique.sort(key=lambda path: (path.name.endswith(".qnt.tensor"), path.name))
    return unique


def write_packed(path: Path, values: list[float], info: dict[str, Any]) -> None:
    quant = info.get("quantize")
    dtype = str(info.get("dtype", "")).lower()

    if quant:
        qtype = str(quant.get("qtype", "")).lower()
        if qtype in {"i16", "int16"}:
            writer = lambda handle, value: handle.write(struct.pack("<h", max(-32768, min(32767, int(round(value))))))
        elif qtype in {"u8", "uint8"}:
            writer = lambda handle, value: handle.write(struct.pack("<B", max(0, min(255, int(round(value))))))
        elif qtype in {"i8", "int8"}:
            writer = lambda handle, value: handle.write(struct.pack("<b", max(-128, min(127, int(round(value))))))
        else:
            raise SystemExit(f"unsupported quantized input qtype for {path.name}: {qtype}")
    elif dtype == "int32":
        writer = lambda handle, value: handle.write(struct.pack("<i", int(round(value))))
    elif dtype == "float16":
        writer = lambda handle, value: handle.write(struct.pack("<e", float(value)))
    elif dtype in {"float", "float32", ""}:
        writer = lambda handle, value: handle.write(struct.pack("<f", float(value)))
    else:
        raise SystemExit(f"unsupported input dtype for {path.name}: {dtype}")

    with path.open("wb") as handle:
        for value in values:
            writer(handle, value)


def write_float_text(path: Path, values: list[float], info: dict[str, Any], source_is_quantized: bool) -> None:
    quant = info.get("quantize") if source_is_quantized else None
    if quant and str(quant.get("qtype", "")).lower() in {"i16", "int16", "i8", "int8"}:
        scale = 2.0 ** int(quant["fl"])
        values = [value / scale for value in values]
    elif quant and str(quant.get("qtype", "")).lower() in {"u8", "uint8"}:
        scale = float(quant["scale"])
        zero_point = float(quant["zero_point"])
        values = [(value - zero_point) * scale for value in values]

    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in values:
            handle.write(f"{value:.16f}\n")


def package(model_dir: Path, package_dir: Path, quant: str) -> None:
    name = model_dir.name
    export_dir = model_dir / "wksp" / f"{name}_{quant}_nbg_unify"
    inf_dir = model_dir / "inf" / f"{name}_{quant}"

    if not export_dir.is_dir():
        raise SystemExit(f"missing export directory: {export_dir}")
    if not inf_dir.is_dir():
        raise SystemExit(f"missing inference directory: {inf_dir}")

    package_dir.mkdir(parents=True, exist_ok=True)
    for old in package_dir.glob("*"):
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()

    shutil.copy2(export_dir / "network_binary.nb", package_dir / "network_binary.nb")
    shutil.copy2(export_dir / "nbg_meta.json", package_dir / "nbg_meta.json")
    meta = json.loads((export_dir / "nbg_meta.json").read_text(encoding="ascii"))

    sample_lines = ["[network]", "./network_binary.nb", "[input]"]
    for index, (key, info) in enumerate(meta.get("Inputs", {}).items()):
        tensor_candidates = candidates_for(inf_dir, key, info, prefer_quantized=bool(info.get("quantize")))
        if not tensor_candidates:
            raise SystemExit(f"no ACUITY inference tensor found for input {key}")
        tensor_path = tensor_candidates[0]
        values = read_numbers(tensor_path)
        expected = product(info.get("shape", []))
        if expected and len(values) != expected:
            raise SystemExit(f"input {key} expected {expected} values, found {len(values)} in {tensor_path.name}")
        input_path = package_dir / f"input_{index}.dat"
        write_packed(input_path, values, info)
        sample_lines.append(f"./input_{index}.dat")

    with (package_dir / "sample.txt").open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(sample_lines) + "\n")

    for index, (key, info) in enumerate(meta.get("Outputs", {}).items()):
        tensor_candidates = candidates_for(inf_dir, key, info, prefer_quantized=False)
        if not tensor_candidates:
            raise SystemExit(f"no ACUITY inference tensor found for output {key}")
        tensor_path = tensor_candidates[0]
        values = read_numbers(tensor_path)
        expected = product(info.get("shape", []))
        if expected and len(values) != expected:
            raise SystemExit(f"output {key} expected {expected} values, found {len(values)} in {tensor_path.name}")
        shutil.copy2(tensor_path, package_dir / f"host_output_{index}.raw.tensor")
        write_float_text(package_dir / f"host_output_{index}.txt", values, info, tensor_path.name.endswith(".qnt.tensor"))

    tokens = model_dir / "tokens.txt"
    if tokens.exists():
        shutil.copy2(tokens, package_dir / "tokens.txt")

    print(f"wrote package: {package_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--quant", required=True)
    args = parser.parse_args()
    package(args.model_dir, args.package_dir, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
