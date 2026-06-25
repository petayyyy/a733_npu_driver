#!/usr/bin/env python3
"""Prepare int16 hidden state input for block0 NBG from FP32 embedding."""
import sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dump_real_llm_oracle import *
from make_real_llm_onnx import SafeTensorReader, read_config


def main():
    model_dir = Path("work/models/qwen25-0.5b-instruct")
    token_path = Path("work/generated/qwen25_05b_w32_block1/prompt_tokens.npy")
    output_dir = Path("work/generated/q2_gate2b")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_config(model_dir / "config.json")
    dim = int(config["hidden_size"])
    seq_len = 32
    fl = 16  # from block0 NBG metadata

    token_array = np.load(token_path)
    tokens = torch.from_numpy(token_array.astype(np.int64, copy=False))[0]
    w = load_weights(model_dir, config, None)

    with torch.no_grad():
        hidden = w["embed"][tokens].to(torch.float32)
        hidden_np = hidden.cpu().numpy().reshape(1, seq_len, dim).astype(np.float32)

    # Quantize to int16 with dynamic_fixed_point fl=16
    scale = 2.0 ** fl
    hidden_int16 = np.round(hidden_np * scale).clip(-32768, 32767).astype(np.int16)
    hidden_int16.tofile(output_dir / "hidden_in_int16.bin")

    print(f"Wrote {output_dir / 'hidden_in_int16.bin'}: {hidden_int16.nbytes} bytes")
    print(f"Shape: {hidden_int16.shape}, fl={fl}")
    print(f"FP32 range: [{hidden_np.min():.6f}, {hidden_np.max():.6f}]")
    print(f"Int16 range: [{hidden_int16.min()}, {hidden_int16.max()}]")

    # Also save the expected block0 output for validation
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    eps = float(config.get("rms_norm_eps", 1e-5))
    cos, sin = rope_tables(seq_len, head_dim, float(config.get("rope_theta", 10000.0)))
    mask_t = torch.triu(torch.full((seq_len, seq_len), -10000.0, dtype=torch.float32), diagonal=1)
    attn_scale = torch.tensor(1.0 / np.sqrt(head_dim), dtype=torch.float32)

    with torch.no_grad():
        layer = w["layers"][0]
        norm = rms_norm(hidden, layer["attn_norm"], eps)
        q = (norm @ layer["q"].T) + layer["qb"]
        k = (norm @ layer["k"].T) + layer["kb"]
        v = (norm @ layer["v"].T) + layer["vb"]
        q = q.reshape(seq_len, n_heads, head_dim).transpose(0, 1)
        k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(0, 1)
        v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(0, 1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = torch.repeat_interleave(k, kv_repeat, dim=0)
        v = torch.repeat_interleave(v, kv_repeat, dim=0)
        probs = torch.softmax((q @ k.transpose(1, 2)) * attn_scale + mask_t[None, :, :], dim=-1)
        ctx = (probs @ v).transpose(0, 1).reshape(seq_len, dim)
        hidden = hidden + (ctx @ layer["o"].T)
        norm = rms_norm(hidden, layer["mlp_norm"], eps)
        gate = norm @ layer["gate"].T
        up_val = norm @ layer["up"].T
        hidden = hidden + ((silu(gate) * up_val) @ layer["down"].T)
        block0_out = hidden.cpu().numpy().reshape(1, seq_len, dim).astype(np.float32)

    np.savez(output_dir / "block0_expected_output.npz", output=block0_out)
    print(f"Wrote expected block0 output: shape {block0_out.shape}")


if __name__ == "__main__":
    main()
