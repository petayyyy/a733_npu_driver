#!/usr/bin/env python3
"""Collect SmoothQuant-style scales for the fixed-window real-LM graph.

This is a host-side build tool only. It runs the same static decoder math as
`make_real_llm_onnx.py` on calibration token windows, records per-channel
activation maxima for transformer linear inputs, and writes exact
re-parameterization scales consumed by `make_real_llm_onnx.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_real_llm_onnx import SafeTensorReader, read_config  # noqa: E402


def load_tensor(reader: SafeTensorReader, name: str) -> torch.Tensor:
    return torch.from_numpy(reader.tensor(name).astype(np.float32, copy=False))


def load_optional(reader: SafeTensorReader, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(reader.optional_tensor(name, shape).astype(np.float32, copy=False))


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * gamma


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(seq, dtype=torch.float32)
    inv_freq = 1.0 / (float(theta) ** (torch.arange(half, dtype=torch.float32) / half))
    freqs = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([freqs, freqs], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos[None, :, :]) + (rotated * sin[None, :, :])


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def update_amax(store: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    current = torch.abs(value).reshape(-1, value.shape[-1]).max(dim=0)[0].detach().cpu()
    if key in store:
        store[key] = torch.max(store[key], current)
    else:
        store[key] = current


def fixed_window(path: Path, seq_len: int) -> torch.Tensor:
    value = np.load(path)
    if value.shape != (1, seq_len):
        raise ValueError(f"{path} has shape {value.shape}, expected {(1, seq_len)}")
    return torch.from_numpy(value.astype(np.int64, copy=False))[0]


def dataset_paths(dataset: Path, max_samples: int | None) -> list[Path]:
    paths: list[Path] = []
    for line in dataset.read_text(encoding="ascii").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for item in line.split():
            paths.append(dataset.parent / item)
            if max_samples is not None and len(paths) >= max_samples:
                return paths
    return paths


def load_weights(model_dir: Path, config: dict[str, Any], max_layers: int | None) -> dict[str, Any]:
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    layers = int(config["num_hidden_layers"]) if max_layers is None else int(max_layers)

    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        weights: dict[str, Any] = {
            "embed": load_tensor(reader, "model.embed_tokens.weight"),
            "layers": [],
        }
        for layer in range(layers):
            prefix = f"model.layers.{layer}"
            weights["layers"].append(
                {
                    "attn_norm": load_tensor(reader, f"{prefix}.input_layernorm.weight"),
                    "mlp_norm": load_tensor(reader, f"{prefix}.post_attention_layernorm.weight"),
                    "q": load_tensor(reader, f"{prefix}.self_attn.q_proj.weight"),
                    "k": load_tensor(reader, f"{prefix}.self_attn.k_proj.weight"),
                    "v": load_tensor(reader, f"{prefix}.self_attn.v_proj.weight"),
                    "qb": load_optional(reader, f"{prefix}.self_attn.q_proj.bias", (dim,)),
                    "kb": load_optional(reader, f"{prefix}.self_attn.k_proj.bias", (n_kv_heads * head_dim,)),
                    "vb": load_optional(reader, f"{prefix}.self_attn.v_proj.bias", (n_kv_heads * head_dim,)),
                    "o": load_tensor(reader, f"{prefix}.self_attn.o_proj.weight"),
                    "gate": load_tensor(reader, f"{prefix}.mlp.gate_proj.weight"),
                    "up": load_tensor(reader, f"{prefix}.mlp.up_proj.weight"),
                    "down": load_tensor(reader, f"{prefix}.mlp.down_proj.weight"),
                }
            )
        return weights
    finally:
        reader.close()


def input_channel_amax(*weights: torch.Tensor) -> torch.Tensor:
    maxima = [torch.abs(weight).max(dim=0)[0].detach().cpu() for weight in weights]
    out = maxima[0]
    for value in maxima[1:]:
        out = torch.max(out, value)
    return out


def smooth_scale(act: torch.Tensor, weight: torch.Tensor, alpha: float, scale_min: float, scale_max: float) -> np.ndarray:
    act = torch.clamp(act.to(torch.float64), min=1.0e-12)
    weight = torch.clamp(weight.to(torch.float64), min=1.0e-12)
    scale = torch.pow(act, alpha) / torch.pow(weight, 1.0 - alpha)
    scale = torch.clamp(scale, min=scale_min, max=scale_max)
    return scale.numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scale-min", type=float, default=1.0e-3)
    parser.add_argument("--scale-max", type=float, default=1.0e3)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    config = read_config(args.model_dir / "config.json")
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    layers = int(config["num_hidden_layers"]) if args.max_layers is None else int(args.max_layers)
    eps = float(config.get("rms_norm_eps", 1.0e-5))
    cos, sin = rope_tables(args.seq_len, head_dim, float(config.get("rope_theta", 10000.0)))
    mask = torch.triu(torch.full((args.seq_len, args.seq_len), -10000.0, dtype=torch.float32), diagonal=1)
    attn_scale = torch.tensor(1.0 / np.sqrt(head_dim), dtype=torch.float32)

    sample_paths = dataset_paths(args.dataset, args.max_samples)
    if not sample_paths:
        raise SystemExit(f"empty dataset: {args.dataset}")

    weights = load_weights(args.model_dir, config, args.max_layers)
    amax: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for path in sample_paths:
            tokens = fixed_window(path, args.seq_len)
            hidden = weights["embed"][tokens].to(torch.float32)
            for layer_index, layer in enumerate(weights["layers"]):
                prefix = f"layer{layer_index}"
                norm = rms_norm(hidden, layer["attn_norm"], eps)
                update_amax(amax, f"{prefix}.attn_norm", norm)

                q = (norm @ layer["q"].T) + layer["qb"]
                k = (norm @ layer["k"].T) + layer["kb"]
                v = (norm @ layer["v"].T) + layer["vb"]
                q = q.reshape(args.seq_len, n_heads, head_dim).transpose(0, 1)
                k = k.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
                v = v.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
                k = torch.repeat_interleave(k, kv_repeat, dim=0)
                v = torch.repeat_interleave(v, kv_repeat, dim=0)
                probs = torch.softmax((q @ k.transpose(1, 2)) * attn_scale + mask[None, :, :], dim=-1)
                ctx = (probs @ v).transpose(0, 1).reshape(args.seq_len, dim)
                update_amax(amax, f"{prefix}.ctx", ctx)
                hidden = hidden + (ctx @ layer["o"].T)

                norm = rms_norm(hidden, layer["mlp_norm"], eps)
                update_amax(amax, f"{prefix}.mlp_norm", norm)
                gate = norm @ layer["gate"].T
                up = norm @ layer["up"].T
                gated = silu(gate) * up
                update_amax(amax, f"{prefix}.gated", gated)
                hidden = hidden + (gated @ layer["down"].T)

    output: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "dataset": str(args.dataset),
        "samples": [str(path) for path in sample_paths],
        "seq_len": args.seq_len,
        "layers": layers,
        "alpha": args.alpha,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "scales": {},
    }

    for layer_index, layer in enumerate(weights["layers"]):
        prefix = f"layer{layer_index}"
        specs = {
            f"{prefix}.attn_norm": input_channel_amax(layer["q"], layer["k"], layer["v"]),
            f"{prefix}.mlp_norm": input_channel_amax(layer["gate"], layer["up"]),
            f"{prefix}.ctx": input_channel_amax(layer["o"]),
            f"{prefix}.gated": input_channel_amax(layer["down"]),
        }
        for key, weight_max in specs.items():
            scale = smooth_scale(amax[key], weight_max, args.alpha, args.scale_min, args.scale_max)
            output[key] = scale
            summary["scales"][key] = {
                "min": float(np.min(scale)),
                "mean": float(np.mean(scale)),
                "max": float(np.max(scale)),
                "act_absmax": float(torch.max(amax[key]).item()),
                "weight_absmax": float(torch.max(weight_max).item()),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    print(f"layers={layers} samples={len(sample_paths)} scales={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
