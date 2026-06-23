#!/usr/bin/env python3
"""Dump FP32 oracle tensors for the fixed-window real-LM ONNX graph."""

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
            "final_norm": load_tensor(reader, "model.norm.weight"),
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


def topk(value: np.ndarray, count: int) -> list[list[float]]:
    flat = value.reshape(-1)
    idx = np.argpartition(flat, -count)[-count:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return [[int(i), float(flat[i])] for i in idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="token_ids .npy with shape 1xW")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

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

    token_array = np.load(args.tokens)
    if token_array.shape != (1, args.seq_len):
        raise SystemExit(f"{args.tokens} has shape {token_array.shape}, expected {(1, args.seq_len)}")
    tokens = torch.from_numpy(token_array.astype(np.int64, copy=False))[0]
    weights = load_weights(args.model_dir, config, args.max_layers)

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        hidden = weights["embed"][tokens].to(torch.float32)
        for layer_index, layer in enumerate(weights["layers"]):
            norm = rms_norm(hidden, layer["attn_norm"], eps)
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
            hidden = hidden + (ctx @ layer["o"].T)

            norm = rms_norm(hidden, layer["mlp_norm"], eps)
            gate = norm @ layer["gate"].T
            up = norm @ layer["up"].T
            hidden = hidden + ((silu(gate) * up) @ layer["down"].T)
            outputs[f"layer{layer_index}_mlp_resid"] = hidden.cpu().numpy().reshape(1, args.seq_len, dim)

        final = rms_norm(hidden, weights["final_norm"], eps)
        outputs["final_rms_out"] = final.cpu().numpy().reshape(1, args.seq_len, dim)
        logits = (final[-1] @ weights["embed"].T).cpu().numpy().reshape(1, 1, int(config["vocab_size"]))
        outputs["logits"] = logits

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **outputs)
    summary = {
        "model_dir": str(args.model_dir),
        "tokens": str(args.tokens),
        "output": str(args.output),
        "seq_len": args.seq_len,
        "layers": layers,
        "logits_topk": topk(outputs["logits"], args.top_k),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
