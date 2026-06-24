#!/usr/bin/env python3
"""Compare packaged ACUITY host outputs with FP oracle tensors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def read_values(path: Path) -> np.ndarray:
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
    return np.asarray(values, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.reshape(-1).astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0.0:
        return 1.0 if np.linalg.norm(a64 - b64) == 0.0 else 0.0
    return float(np.dot(a64, b64) / denom)


def topk(values: np.ndarray, count: int) -> list[list[float]]:
    flat = values.reshape(-1)
    idx = np.argpartition(flat, -count)[-count:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return [[int(i), float(flat[i])] for i in idx]


def compare(name: str, host: np.ndarray, oracle: np.ndarray, top_k: int) -> dict[str, Any]:
    expected = oracle.reshape(-1).astype(np.float32)
    if host.size != expected.size:
        raise SystemExit(f"{name}: host has {host.size} values, oracle has {expected.size}")
    diff = host - expected
    result: dict[str, Any] = {
        "name": name,
        "values": int(host.size),
        "cosine": cosine(host, expected),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) * diff.astype(np.float64)))) if diff.size else 0.0,
    }
    if name == "logits":
        result["host_topk"] = topk(host, top_k)
        result["oracle_topk"] = topk(expected, top_k)
        result["top1_match"] = bool(result["host_topk"][0][0] == result["oracle_topk"][0][0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--model-info", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    info = json.loads(args.model_info.read_text(encoding="ascii"))
    output_names = info.get("output_names")
    if not output_names:
        output_names = ["logits"]

    oracle = np.load(args.oracle)
    results = []
    chunk_names = info.get("lm_head_chunk_outputs") or []
    if info.get("lm_head_output_mode") == "chunks" and chunk_names:
        by_name = {}
        meta_path = args.package_dir / "nbg_meta.json"
        output_keys = []
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            output_keys = list(meta.get("Outputs", {}).keys())
        for index, fallback_name in enumerate(output_names):
            host_path = args.package_dir / f"host_output_{index}.txt"
            if not host_path.exists():
                raise SystemExit(f"missing host output: {host_path}")
            key = output_keys[index] if index < len(output_keys) else fallback_name
            key_sanitized = sanitize(key)
            name = next((candidate for candidate in output_names if sanitize(candidate) in key_sanitized), fallback_name)
            by_name[name] = read_values(host_path)
        missing = [name for name in chunk_names if name not in by_name]
        if missing:
            raise SystemExit(f"missing chunk host output(s): {missing}")
        logits = np.concatenate([by_name[name].reshape(-1) for name in chunk_names])
        if "logits" not in oracle:
            raise SystemExit(f"missing oracle tensor 'logits' in {args.oracle}")
        results.append(compare("logits", logits, np.asarray(oracle["logits"]), args.top_k))
        for name in output_names:
            if name in chunk_names:
                continue
            if name not in oracle:
                raise SystemExit(f"missing oracle tensor {name!r} in {args.oracle}")
            results.append(compare(name, by_name[name], np.asarray(oracle[name]), args.top_k))
    else:
        for index, name in enumerate(output_names):
            host_path = args.package_dir / f"host_output_{index}.txt"
            if not host_path.exists():
                raise SystemExit(f"missing host output: {host_path}")
            if name not in oracle:
                raise SystemExit(f"missing oracle tensor {name!r} in {args.oracle}")
            results.append(compare(name, read_values(host_path), np.asarray(oracle[name]), args.top_k))

    payload = {
        "package_dir": str(args.package_dir),
        "oracle": str(args.oracle),
        "model_info": str(args.model_info),
        "results": results,
        "min_cosine": min(item["cosine"] for item in results),
        "logits_cosine": next((item["cosine"] for item in results if item["name"] == "logits"), None),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")

    print(f"package: {args.package_dir}")
    print(f"min_cosine={payload['min_cosine']:.9f} logits_cosine={payload['logits_cosine']:.9f}")
    for item in results:
        extra = ""
        if item["name"] == "logits":
            extra = f" top1_match={item['top1_match']} host_top1={item['host_topk'][0][0]} oracle_top1={item['oracle_topk'][0][0]}"
        print(
            f"{item['name']}: cosine={item['cosine']:.9f} "
            f"max_abs={item['max_abs_diff']:.6f} mean_abs={item['mean_abs_diff']:.6f}{extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
