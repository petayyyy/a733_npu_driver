#!/usr/bin/env python3
"""Q2 Gate 2A: Simulate 24-block int16 chain, log per-layer cosine drift vs FP32.

This script models what happens when 24 int16 per-block NBGs are chained
end-to-end: each weight matrix is quantized to int16 dynamic fixed point, and
at each layer boundary the hidden state is quantized/dequantized to int16
(simulating the NBG output-input quantize cycle).

Outputs:
- Per-layer cosine drift curve (int16 hidden state vs FP32 oracle hidden state)
- End-to-end logits cosine and top-1 match
- Summary JSON for the report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_real_llm_onnx import SafeTensorReader, read_config  # noqa: E402


def quantize_int16(t: torch.Tensor, fl: Optional[int] = None) -> tuple[torch.Tensor, int]:
    """Quantize to int16 dynamic fixed point, dequantize back to float32."""
    if fl is None:
        max_abs = float(t.abs().max().item())
        if max_abs <= 0:
            return t.clone(), 0
        raw_fl = math.floor(math.log2(32767.0 / max_abs))
        fl = max(-20, min(20, raw_fl))
    scale = 2.0 ** fl
    q = torch.round(t * scale).clamp(-32768, 32767)
    return q / scale, fl


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_weights(model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    dim = int(config["hidden_size"])
    n_kv_heads = int(config.get("num_key_value_heads", int(config["num_attention_heads"])))
    head_dim = dim // int(config["num_attention_heads"])
    layers = int(config["num_hidden_layers"])
    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        w: dict[str, Any] = {
            "embed": torch.from_numpy(reader.tensor("model.embed_tokens.weight").astype(np.float32, copy=False)),
            "final_norm": torch.from_numpy(reader.tensor("model.norm.weight").astype(np.float32, copy=False)),
            "layers": [],
        }
        for layer_idx in range(layers):
            prefix = f"model.layers.{layer_idx}"
            w["layers"].append({
                "attn_norm": torch.from_numpy(reader.tensor(f"{prefix}.input_layernorm.weight").astype(np.float32, copy=False)),
                "mlp_norm": torch.from_numpy(reader.tensor(f"{prefix}.post_attention_layernorm.weight").astype(np.float32, copy=False)),
                "q": torch.from_numpy(reader.tensor(f"{prefix}.self_attn.q_proj.weight").astype(np.float32, copy=False)),
                "k": torch.from_numpy(reader.tensor(f"{prefix}.self_attn.k_proj.weight").astype(np.float32, copy=False)),
                "v": torch.from_numpy(reader.tensor(f"{prefix}.self_attn.v_proj.weight").astype(np.float32, copy=False)),
                "qb": torch.from_numpy(reader.optional_tensor(f"{prefix}.self_attn.q_proj.bias", (dim,)).astype(np.float32, copy=False)),
                "kb": torch.from_numpy(reader.optional_tensor(f"{prefix}.self_attn.k_proj.bias", (n_kv_heads * head_dim,)).astype(np.float32, copy=False)),
                "vb": torch.from_numpy(reader.optional_tensor(f"{prefix}.self_attn.v_proj.bias", (n_kv_heads * head_dim,)).astype(np.float32, copy=False)),
                "o": torch.from_numpy(reader.tensor(f"{prefix}.self_attn.o_proj.weight").astype(np.float32, copy=False)),
                "gate": torch.from_numpy(reader.tensor(f"{prefix}.mlp.gate_proj.weight").astype(np.float32, copy=False)),
                "up": torch.from_numpy(reader.tensor(f"{prefix}.mlp.up_proj.weight").astype(np.float32, copy=False)),
                "down": torch.from_numpy(reader.tensor(f"{prefix}.mlp.down_proj.weight").astype(np.float32, copy=False)),
            })
        return w
    finally:
        reader.close()


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


def qmatmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """MatMul with int16-quantized weight."""
    wq, _fl = quantize_int16(w)
    return x @ wq.T


def topk(value: np.ndarray, count: int) -> list[list[float]]:
    flat = value.reshape(-1)
    idx = np.argpartition(flat, -count)[-count:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return [[int(i), float(flat[i])] for i in idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-boundary-quant", action="store_true",
                        help="skip int16 quantization at block boundaries (only quantize weights)")
    parser.add_argument("--decode-steps", type=int, default=0,
                        help="run autoregressive decode for N steps after the initial forward pass")
    parser.add_argument("--decode-only", action="store_true",
                        help="only run the int16 decode loop, skip FP32 baseline")
    args = parser.parse_args()

    config = read_config(args.model_dir / "config.json")
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    layers = int(config["num_hidden_layers"])
    vocab = int(config["vocab_size"])
    eps = float(config.get("rms_norm_eps", 1e-5))
    theta = float(config.get("rope_theta", 10000.0))

    cos, sin = rope_tables(args.seq_len, head_dim, theta)
    mask = torch.triu(torch.full((args.seq_len, args.seq_len), -10000.0, dtype=torch.float32), diagonal=1)
    attn_scale = torch.tensor(1.0 / np.sqrt(head_dim), dtype=torch.float32)

    token_array = np.load(args.tokens)
    if token_array.shape != (1, args.seq_len):
        print(f"ERROR: {args.tokens} shape {token_array.shape}, expected (1, {args.seq_len})")
        return 1
    tokens = torch.from_numpy(token_array.astype(np.int64, copy=False))[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- FP32 Oracle Pass ---
    print("Running FP32 oracle pass...")
    w = load_weights(args.model_dir, config)
    fp32_hidden_states: list[np.ndarray] = []

    with torch.no_grad():
        hidden = w["embed"][tokens].to(torch.float32)
        fp32_hidden_states.append(hidden.cpu().numpy().reshape(1, args.seq_len, dim))

        for layer_idx, layer in enumerate(w["layers"]):
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
            up_val = norm @ layer["up"].T
            hidden = hidden + ((silu(gate) * up_val) @ layer["down"].T)
            fp32_hidden_states.append(hidden.cpu().numpy().reshape(1, args.seq_len, dim))

        final = rms_norm(hidden, w["final_norm"], eps)
        fp32_logits = (final[-1] @ w["embed"].T).cpu().numpy().reshape(1, 1, vocab)
        fp32_hidden_states.append(final.cpu().numpy().reshape(1, args.seq_len, dim))

    fp32_top1 = int(np.argmax(fp32_logits.reshape(-1)))
    print(f"FP32 oracle top-1: {fp32_top1}")

    # --- Int16 Simulation Pass ---
    boundary_quant = not args.no_boundary_quant
    print(f"Running int16 simulation pass (boundary_quant={boundary_quant})...")

    per_layer_cosine: list[dict[str, Any]] = []
    with torch.no_grad():
        # Embedding: not quantized (token embedding is int16 in ACUITY but we keep it FP32
        # to focus on decoder-block depth accumulation)
        hidden_i16 = w["embed"][tokens].to(torch.float32)
        # Quantize embedding output (simulating NBG output boundary)
        if boundary_quant:
            hidden_i16, _emb_fl = quantize_int16(hidden_i16)
        cos0 = cosine(hidden_i16.cpu().numpy(), fp32_hidden_states[0])
        per_layer_cosine.append({"layer": "embed", "cosine": cos0, "note": "post-embed"})
        print(f"  embed: cosine={cos0:.8f}")

        for layer_idx, layer in enumerate(w["layers"]):
            # Attention input norm
            norm = rms_norm(hidden_i16, layer["attn_norm"], eps)
            if boundary_quant:
                norm, _ = quantize_int16(norm)

            # QKV projections with int16 weights
            q = qmatmul(norm, layer["q"]) + layer["qb"]
            k = qmatmul(norm, layer["k"]) + layer["kb"]
            v = qmatmul(norm, layer["v"]) + layer["vb"]

            q = q.reshape(args.seq_len, n_heads, head_dim).transpose(0, 1)
            k = k.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
            v = v.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            k = torch.repeat_interleave(k, kv_repeat, dim=0)
            v = torch.repeat_interleave(v, kv_repeat, dim=0)

            if boundary_quant:
                q, _ = quantize_int16(q)
                k, _ = quantize_int16(k)

            scores = (q @ k.transpose(1, 2)) * attn_scale
            if boundary_quant:
                scores, _ = quantize_int16(scores)
            probs = torch.softmax(scores + mask[None, :, :], dim=-1)

            if boundary_quant:
                probs, _ = quantize_int16(probs)
            ctx = (probs @ v).transpose(0, 1).reshape(args.seq_len, dim)

            if boundary_quant:
                ctx, _ = quantize_int16(ctx)
            hidden_i16 = hidden_i16 + qmatmul(ctx, layer["o"])

            if boundary_quant:
                hidden_i16, _ = quantize_int16(hidden_i16)

            # MLP
            norm_mlp = rms_norm(hidden_i16, layer["mlp_norm"], eps)
            if boundary_quant:
                norm_mlp, _ = quantize_int16(norm_mlp)

            gate = qmatmul(norm_mlp, layer["gate"])
            up_val = qmatmul(norm_mlp, layer["up"])

            if boundary_quant:
                gate, _ = quantize_int16(gate)
                up_val, _ = quantize_int16(up_val)

            mlp_out = (silu(gate) * up_val)
            if boundary_quant:
                mlp_out, _ = quantize_int16(mlp_out)

            hidden_i16 = hidden_i16 + qmatmul(mlp_out, layer["down"])

            # Block boundary quantization (NBG output -> next NBG input)
            if boundary_quant:
                hidden_i16, _ = quantize_int16(hidden_i16)

            cos_val = cosine(hidden_i16.cpu().numpy(), fp32_hidden_states[layer_idx + 1])
            per_layer_cosine.append({
                "layer": layer_idx,
                "cosine": cos_val,
                "note": "post-mlp-resid" if not boundary_quant else "post-mlp-resid-boundary-quant",
            })
            print(f"  layer {layer_idx}: cosine={cos_val:.8f}")

        # Final RMSNorm + lm_head
        final_norm_i16 = rms_norm(hidden_i16, w["final_norm"], eps)
        if boundary_quant:
            final_norm_i16, _ = quantize_int16(final_norm_i16)

        cos_final = cosine(final_norm_i16.cpu().numpy(), fp32_hidden_states[-1])
        per_layer_cosine.append({"layer": "final_norm", "cosine": cos_final})
        print(f"  final_norm: cosine={cos_final:.8f}")

        # Logits projection with int16 weight (embedding is also lm_head)
        embed_i16, _ = quantize_int16(w["embed"])
        i16_logits = (final_norm_i16[-1] @ embed_i16.T).cpu().numpy().reshape(1, 1, vocab)

    i16_top1 = int(np.argmax(i16_logits.reshape(-1)))
    logits_cos = cosine(i16_logits, fp32_logits)
    top1_match = fp32_top1 == i16_top1

    print(f"\n--- Gate 2A Results ---")
    print(f"FP32 top-1: {fp32_top1}")
    print(f"Int16 top-1: {i16_top1}")
    print(f"Top-1 match: {top1_match}")
    print(f"Logits cosine: {logits_cos:.8f}")
    print(f"Boundary quant: {boundary_quant}")

    # Per-layer drift
    print(f"\n--- Per-Layer Cosine Drift ---")
    for entry in per_layer_cosine:
        print(f"  {entry['layer']}: {entry['cosine']:.8f}")

    # Save outputs
    result = {
        "model_dir": str(args.model_dir),
        "tokens": str(args.tokens),
        "seq_len": args.seq_len,
        "layers": layers,
        "boundary_quant": boundary_quant,
        "fp32_top1": fp32_top1,
        "int16_top1": i16_top1,
        "top1_match": top1_match,
        "logits_cosine": logits_cos,
        "per_layer_cosine_drift": per_layer_cosine,
        "fp32_logits_topk": topk(fp32_logits, args.top_k),
        "int16_logits_topk": topk(i16_logits, args.top_k),
    }

    out_json = args.output_dir / "q2_gate2a_simulation.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")

    np.savez(args.output_dir / "q2_gate2a_fp32_oracle.npz",
             logits=fp32_logits)
    np.savez(args.output_dir / "q2_gate2a_int16_sim.npz",
             logits=i16_logits)

    # --- Decode Loop (if requested) ---
    decode_tokens: list[int] = []
    decode_results: list[dict] = []
    if args.decode_steps > 0:
        print(f"\n--- Autoregressive Decode ({args.decode_steps} steps) ---")
        current_tokens = torch.from_numpy(token_array.astype(np.int64, copy=False))[0].clone()
        embed_i16_final, _ = quantize_int16(w["embed"])
        for step in range(args.decode_steps):
            # Forward pass with current window
            with torch.no_grad():
                hidden_i16 = w["embed"][current_tokens].to(torch.float32)
                if boundary_quant:
                    hidden_i16, _ = quantize_int16(hidden_i16)

                for layer_idx, layer in enumerate(w["layers"]):
                    norm = rms_norm(hidden_i16, layer["attn_norm"], eps)
                    if boundary_quant:
                        norm, _ = quantize_int16(norm)
                    q = qmatmul(norm, layer["q"]) + layer["qb"]
                    k = qmatmul(norm, layer["k"]) + layer["kb"]
                    v = qmatmul(norm, layer["v"]) + layer["vb"]
                    q = q.reshape(args.seq_len, n_heads, head_dim).transpose(0, 1)
                    k = k.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
                    v = v.reshape(args.seq_len, n_kv_heads, head_dim).transpose(0, 1)
                    q = apply_rope(q, cos, sin)
                    k = apply_rope(k, cos, sin)
                    k = torch.repeat_interleave(k, kv_repeat, dim=0)
                    v = torch.repeat_interleave(v, kv_repeat, dim=0)
                    if boundary_quant:
                        q, _ = quantize_int16(q)
                        k, _ = quantize_int16(k)
                    scores = (q @ k.transpose(1, 2)) * attn_scale
                    if boundary_quant:
                        scores, _ = quantize_int16(scores)
                    probs = torch.softmax(scores + mask[None, :, :], dim=-1)
                    if boundary_quant:
                        probs, _ = quantize_int16(probs)
                    ctx = (probs @ v).transpose(0, 1).reshape(args.seq_len, dim)
                    if boundary_quant:
                        ctx, _ = quantize_int16(ctx)
                    hidden_i16 = hidden_i16 + qmatmul(ctx, layer["o"])
                    if boundary_quant:
                        hidden_i16, _ = quantize_int16(hidden_i16)

                    norm_mlp = rms_norm(hidden_i16, layer["mlp_norm"], eps)
                    if boundary_quant:
                        norm_mlp, _ = quantize_int16(norm_mlp)
                    gate = qmatmul(norm_mlp, layer["gate"])
                    up_val = qmatmul(norm_mlp, layer["up"])
                    if boundary_quant:
                        gate, _ = quantize_int16(gate)
                        up_val, _ = quantize_int16(up_val)
                    mlp_out = (silu(gate) * up_val)
                    if boundary_quant:
                        mlp_out, _ = quantize_int16(mlp_out)
                    hidden_i16 = hidden_i16 + qmatmul(mlp_out, layer["down"])
                    if boundary_quant:
                        hidden_i16, _ = quantize_int16(hidden_i16)

                final_norm_i16 = rms_norm(hidden_i16, w["final_norm"], eps)
                if boundary_quant:
                    final_norm_i16, _ = quantize_int16(final_norm_i16)
                step_logits = (final_norm_i16[-1] @ embed_i16_final.T).cpu().numpy().reshape(-1)

            next_token = int(np.argmax(step_logits))
            top5_idx = np.argpartition(step_logits, -5)[-5:]
            top5_idx = top5_idx[np.argsort(step_logits[top5_idx])[::-1]]
            top5 = [[int(i), float(step_logits[i])] for i in top5_idx]

            decode_tokens.append(next_token)
            decode_results.append({
                "step": step,
                "next_token": next_token,
                "top5": top5,
            })
            print(f"  step {step}: token={next_token} top5={[t[0] for t in top5[:3]]}")

            # Slide window
            current_tokens = torch.cat([current_tokens[1:], torch.tensor([next_token], dtype=torch.int64)])

        print(f"\nDecoded tokens: {decode_tokens}")

    print(f"\nwrote {out_json}")

    result["decode_tokens"] = decode_tokens
    result["decode_results"] = decode_results
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")

    if logits_cos > 0.90 and top1_match:
        print("\n*** Gate 2A PASSES ***")
        print("Int16 block-chaining is coherent end-to-end on host.")
        return 0
    else:
        print("\n*** Gate 2A FAILS ***")
        print(f"Logits cosine {logits_cos:.6f} below 0.90 threshold")
        print("Depth accumulation kills int16 coherence.")
        print("Do NOT build the runtime. Record the drift curve.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
