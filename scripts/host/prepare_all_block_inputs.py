#!/usr/bin/env python3
"""Generate hidden_in.npy + dataset.txt for all decoder blocks 0-23
from the FP32 oracle's intermediate hidden states."""
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
    seq_len = 32

    config = read_config(model_dir / "config.json")
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    eps = float(config.get("rms_norm_eps", 1e-5))

    cos, sin = rope_tables(seq_len, head_dim, float(config.get("rope_theta", 10000.0)))
    mask_t = torch.triu(torch.full((seq_len, seq_len), -10000.0, dtype=torch.float32), diagonal=1)
    attn_scale = torch.tensor(1.0 / np.sqrt(head_dim), dtype=torch.float32)

    token_array = np.load(token_path)
    tokens = torch.from_numpy(token_array.astype(np.int64, copy=False))[0]
    w = load_weights(model_dir, config, None)

    with torch.no_grad():
        hidden = w["embed"][tokens].to(torch.float32)

        for layer_idx, layer in enumerate(w["layers"]):
            # Save current hidden state as input for this block
            block_dir = Path(f"work/generated/qwen25_05b_w32_block{layer_idx}")
            block_dir.mkdir(parents=True, exist_ok=True)
            hidden_np = hidden.cpu().numpy().reshape(1, seq_len, dim).astype(np.float32)
            np.save(block_dir / "hidden_in.npy", hidden_np)
            (block_dir / "dataset.txt").write_text("hidden_in.npy\n", encoding="ascii")
            print(f"block{layer_idx}: saved input [{hidden_np.min():.4f}, {hidden_np.max():.4f}]")

            # Run this layer to get next block's input
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

    # Save final hidden state for final stage input
    final_dir = Path("work/generated/qwen25_05b_w32_final")
    final_dir.mkdir(parents=True, exist_ok=True)
    hidden_np = hidden.cpu().numpy().reshape(1, seq_len, dim).astype(np.float32)
    np.save(final_dir / "hidden_in.npy", hidden_np)
    (final_dir / "dataset.txt").write_text("hidden_in.npy\n", encoding="ascii")
    print(f"final: saved input [{hidden_np.min():.4f}, {hidden_np.max():.4f}]")

    # For embedding stage, use token_ids.npy
    embed_dir = Path("work/generated/qwen25_05b_w32_embed")
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_dir / "token_ids.npy", token_array)
    (embed_dir / "dataset.txt").write_text("token_ids.npy\n", encoding="ascii")
    print(f"embed: saved token_ids.npy")


if __name__ == "__main__":
    main()
