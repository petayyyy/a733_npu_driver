#!/usr/bin/env python3
"""Compare a fixed-window real-LM ONNX Runtime output with FP oracle tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def compare(name: str, actual: np.ndarray, expected: np.ndarray, top_k: int) -> dict[str, Any]:
    actual_flat = actual.reshape(-1).astype(np.float32)
    expected_flat = expected.reshape(-1).astype(np.float32)
    if actual_flat.size != expected_flat.size:
        raise SystemExit(f"{name}: ONNX Runtime has {actual_flat.size} values, oracle has {expected_flat.size}")
    diff = actual_flat - expected_flat
    result: dict[str, Any] = {
        "name": name,
        "values": int(actual_flat.size),
        "cosine": cosine(actual_flat, expected_flat),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) * diff.astype(np.float64)))) if diff.size else 0.0,
    }
    if name == "logits":
        result["onnxruntime_topk"] = topk(actual_flat, top_k)
        result["oracle_topk"] = topk(expected_flat, top_k)
        result["top1_match"] = bool(result["onnxruntime_topk"][0][0] == result["oracle_topk"][0][0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="token_ids .npy with shape 1xW")
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--model-info", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    import onnxruntime as ort

    info = json.loads(args.model_info.read_text(encoding="ascii"))
    output_names = info.get("output_names") or ["logits"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])

    token_ids = np.load(args.tokens).astype(np.int32, copy=False)
    outputs = session.run(output_names, {"token_ids": token_ids})
    oracle = np.load(args.oracle)

    results = []
    chunk_names = info.get("lm_head_chunk_outputs") or []
    if info.get("lm_head_output_mode") == "chunks" and chunk_names:
        by_name = dict(zip(output_names, outputs))
        missing = [name for name in chunk_names if name not in by_name]
        if missing:
            raise SystemExit(f"missing chunk output(s) from ONNX Runtime: {missing}")
        logits = np.concatenate([np.asarray(by_name[name]) for name in chunk_names], axis=-1)
        if "logits" not in oracle:
            raise SystemExit(f"missing oracle tensor 'logits' in {args.oracle}")
        results.append(compare("logits", logits, np.asarray(oracle["logits"]), args.top_k))
        for name, actual in zip(output_names, outputs):
            if name in chunk_names:
                continue
            if name not in oracle:
                raise SystemExit(f"missing oracle tensor {name!r} in {args.oracle}")
            results.append(compare(name, np.asarray(actual), np.asarray(oracle[name]), args.top_k))
    else:
        for name, actual in zip(output_names, outputs):
            if name not in oracle:
                raise SystemExit(f"missing oracle tensor {name!r} in {args.oracle}")
            results.append(compare(name, np.asarray(actual), np.asarray(oracle[name]), args.top_k))

    payload = {
        "onnx": str(args.onnx),
        "tokens": str(args.tokens),
        "oracle": str(args.oracle),
        "model_info": str(args.model_info),
        "threads": args.threads,
        "results": results,
        "min_cosine": min(item["cosine"] for item in results),
        "logits_cosine": next((item["cosine"] for item in results if item["name"] == "logits"), None),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")

    print(f"onnx: {args.onnx}")
    print(f"min_cosine={payload['min_cosine']:.9f} logits_cosine={payload['logits_cosine']:.9f}")
    for item in results:
        extra = ""
        if item["name"] == "logits":
            extra = (
                f" top1_match={item['top1_match']}"
                f" onnxruntime_top1={item['onnxruntime_topk'][0][0]}"
                f" oracle_top1={item['oracle_topk'][0][0]}"
            )
        print(
            f"{item['name']}: cosine={item['cosine']:.9f} "
            f"max_abs={item['max_abs_diff']:.6f} mean_abs={item['mean_abs_diff']:.6f}{extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
