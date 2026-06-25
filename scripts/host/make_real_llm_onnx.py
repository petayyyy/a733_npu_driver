#!/usr/bin/env python3
"""Build a fixed-window real decoder LM ONNX graph from HF safetensors.

The generated graph follows the T2/T3 NPU-only contract: token ids enter the
NBG, token embedding Gather, every decoder layer, final RMSNorm, and sliced
last-token logits are inside the graph. CPU-side code may tokenize and select
the next token, but model-layer compute is represented in ONNX for ACUITY.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


BATCH = 1


class SafeTensorReader:
    """Minimal safetensors reader for unsharded BF16/F32 checkpoints."""

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
        if dtype == "F32":
            return np.frombuffer(data, dtype="<f4").reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(shape)
        raise ValueError(f"unsupported tensor dtype for {name}: {dtype}")

    def optional_tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name in self.header:
            value = self.tensor(name)
            if value.shape != shape:
                raise ValueError(f"unexpected tensor shape for {name}: {value.shape}, expected {shape}")
            return value
        return np.zeros(shape, dtype=np.float32)


def f32_tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name=name)


def i64_tensor(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def i32_tensor(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int32), name=name)


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    positions = np.arange(seq, dtype=np.float32)
    inv_freq = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float32) / half))
    freqs = np.outer(positions, inv_freq)
    angles = np.concatenate([freqs, freqs], axis=1)
    cos = np.cos(angles).reshape(1, 1, seq, head_dim)
    sin = np.sin(angles).reshape(1, 1, seq, head_dim)
    return cos.astype(np.float32), sin.astype(np.float32)


def add_rms_norm(nodes: list[onnx.NodeProto], prefix: str, x: str, gamma: str) -> str:
    squared = f"{prefix}_squared"
    mean = f"{prefix}_mean"
    mean_eps = f"{prefix}_mean_eps"
    denom = f"{prefix}_denom"
    inv_denom = f"{prefix}_inv_denom"
    norm = f"{prefix}_norm"
    out = f"{prefix}_out"

    nodes.extend(
        [
            helper.make_node("Mul", [x, x], [squared], name=f"{prefix}_square"),
            helper.make_node("ReduceMean", [squared], [mean], name=f"{prefix}_mean", axes=[2], keepdims=1),
            helper.make_node("Add", [mean, "eps"], [mean_eps], name=f"{prefix}_eps"),
            helper.make_node("Sqrt", [mean_eps], [denom], name=f"{prefix}_sqrt"),
            helper.make_node("Reciprocal", [denom], [inv_denom], name=f"{prefix}_reciprocal"),
            helper.make_node("Mul", [x, inv_denom], [norm], name=f"{prefix}_scale_to_unit"),
            helper.make_node("Mul", [norm, gamma], [out], name=f"{prefix}_gamma"),
        ]
    )
    return out


def add_rope(nodes: list[onnx.NodeProto], prefix: str, x: str) -> str:
    first = f"{prefix}_first_half"
    second = f"{prefix}_second_half"
    neg_second = f"{prefix}_neg_second_half"
    rotated = f"{prefix}_rotated"
    x_cos = f"{prefix}_x_cos"
    rotated_sin = f"{prefix}_rotated_sin"
    out = f"{prefix}_out"

    nodes.extend(
        [
            helper.make_node(
                "Slice",
                [x, "slice_start_0", "slice_end_half", "slice_axis_last", "slice_step_1"],
                [first],
                name=f"{prefix}_slice_first",
            ),
            helper.make_node(
                "Slice",
                [x, "slice_start_half", "slice_end_head", "slice_axis_last", "slice_step_1"],
                [second],
                name=f"{prefix}_slice_second",
            ),
            helper.make_node("Neg", [second], [neg_second], name=f"{prefix}_neg_second"),
            helper.make_node("Concat", [neg_second, first], [rotated], name=f"{prefix}_rotate_half", axis=3),
            helper.make_node("Mul", [x, "rope_cos"], [x_cos], name=f"{prefix}_mul_cos"),
            helper.make_node("Mul", [rotated, "rope_sin"], [rotated_sin], name=f"{prefix}_mul_sin"),
            helper.make_node("Add", [x_cos, rotated_sin], [out], name=f"{prefix}_rope_add"),
        ]
    )
    return out


def add_gqa_repeat(nodes: list[onnx.NodeProto], prefix: str, x: str) -> str:
    expanded = f"{prefix}_expanded"
    tiled = f"{prefix}_tiled"
    out = f"{prefix}_out"

    nodes.extend(
        [
            helper.make_node("Reshape", [x, "shape_kv_expand"], [expanded], name=f"{prefix}_expand"),
            helper.make_node("Tile", [expanded, "kv_tile_repeats"], [tiled], name=f"{prefix}_tile"),
            helper.make_node("Reshape", [tiled, "shape_heads"], [out], name=f"{prefix}_merge"),
        ]
    )
    return out


def add_attention(nodes: list[onnx.NodeProto], prefix: str, hidden: str, use_smoothquant: bool) -> str:
    norm = add_rms_norm(nodes, f"{prefix}_attn_rms", hidden, f"{prefix}_attn_gamma")
    q_linear = f"{prefix}_q_linear"
    k_linear = f"{prefix}_k_linear"
    v_linear = f"{prefix}_v_linear"
    q = f"{prefix}_q"
    k = f"{prefix}_k"
    v = f"{prefix}_v"
    q_reshape = f"{prefix}_q_reshape"
    k_reshape = f"{prefix}_k_reshape"
    v_reshape = f"{prefix}_v_reshape"
    q_heads = f"{prefix}_q_heads"
    k_heads = f"{prefix}_k_heads"
    v_heads = f"{prefix}_v_heads"

    nodes.extend(
        [
            helper.make_node("MatMul", [norm, f"{prefix}_wq"], [q_linear], name=f"{prefix}_q_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_wk"], [k_linear], name=f"{prefix}_k_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_wv"], [v_linear], name=f"{prefix}_v_proj"),
            helper.make_node("Add", [q_linear, f"{prefix}_bq"], [q], name=f"{prefix}_q_bias"),
            helper.make_node("Add", [k_linear, f"{prefix}_bk"], [k], name=f"{prefix}_k_bias"),
            helper.make_node("Add", [v_linear, f"{prefix}_bv"], [v], name=f"{prefix}_v_bias"),
            helper.make_node("Reshape", [q, "shape_q"], [q_reshape], name=f"{prefix}_q_reshape"),
            helper.make_node("Reshape", [k, "shape_kv"], [k_reshape], name=f"{prefix}_k_reshape"),
            helper.make_node("Reshape", [v, "shape_kv"], [v_reshape], name=f"{prefix}_v_reshape"),
            helper.make_node("Transpose", [q_reshape], [q_heads], name=f"{prefix}_q_heads", perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", [k_reshape], [k_heads], name=f"{prefix}_k_heads", perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", [v_reshape], [v_heads], name=f"{prefix}_v_heads", perm=[0, 2, 1, 3]),
        ]
    )

    q_rope = add_rope(nodes, f"{prefix}_q_rope", q_heads)
    k_rope = add_rope(nodes, f"{prefix}_k_rope", k_heads)
    k_rep = add_gqa_repeat(nodes, f"{prefix}_k_repeat", k_rope)
    v_rep = add_gqa_repeat(nodes, f"{prefix}_v_repeat", v_heads)

    k_t = f"{prefix}_k_t"
    scores = f"{prefix}_scores"
    scaled = f"{prefix}_scores_scaled"
    masked = f"{prefix}_scores_masked"
    probs = f"{prefix}_probs"
    ctx = f"{prefix}_ctx"
    ctx_t = f"{prefix}_ctx_t"
    ctx_flat = f"{prefix}_ctx_flat"
    ctx_smooth = f"{prefix}_ctx_smooth"
    attn_out = f"{prefix}_attn_out"
    out = f"{prefix}_attn_resid"

    nodes.extend(
        [
            helper.make_node("Transpose", [k_rep], [k_t], name=f"{prefix}_k_transpose", perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", [q_rope, k_t], [scores], name=f"{prefix}_attn_scores"),
            helper.make_node("Mul", [scores, "scale_attn"], [scaled], name=f"{prefix}_attn_scale"),
            helper.make_node("Add", [scaled, "causal_mask"], [masked], name=f"{prefix}_causal_mask"),
            helper.make_node("Softmax", [masked], [probs], name=f"{prefix}_softmax", axis=3),
            helper.make_node("MatMul", [probs, v_rep], [ctx], name=f"{prefix}_attn_context"),
            helper.make_node("Transpose", [ctx], [ctx_t], name=f"{prefix}_ctx_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", [ctx_t, "shape_hidden"], [ctx_flat], name=f"{prefix}_ctx_merge_heads"),
        ]
    )
    if use_smoothquant:
        nodes.append(helper.make_node("Mul", [ctx_flat, f"{prefix}_ctx_smooth_inv"], [ctx_smooth], name=f"{prefix}_ctx_smooth"))
        out_proj_input = ctx_smooth
    else:
        out_proj_input = ctx_flat
    nodes.extend(
        [
            helper.make_node("MatMul", [out_proj_input, f"{prefix}_wo"], [attn_out], name=f"{prefix}_out_proj"),
            helper.make_node("Add", [hidden, attn_out], [out], name=f"{prefix}_attn_residual"),
        ]
    )
    return out


def add_swiglu(nodes: list[onnx.NodeProto], prefix: str, hidden: str, use_smoothquant: bool) -> str:
    norm = add_rms_norm(nodes, f"{prefix}_mlp_rms", hidden, f"{prefix}_mlp_gamma")
    gate = f"{prefix}_gate"
    up = f"{prefix}_up"
    gate_sigmoid = f"{prefix}_gate_sigmoid"
    silu = f"{prefix}_silu"
    gated = f"{prefix}_gated"
    gated_smooth = f"{prefix}_gated_smooth"
    down = f"{prefix}_down"
    out = f"{prefix}_mlp_resid"

    nodes.extend(
        [
            helper.make_node("MatMul", [norm, f"{prefix}_w_gate"], [gate], name=f"{prefix}_gate_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_w_up"], [up], name=f"{prefix}_up_proj"),
            helper.make_node("Sigmoid", [gate], [gate_sigmoid], name=f"{prefix}_sigmoid"),
            helper.make_node("Mul", [gate, gate_sigmoid], [silu], name=f"{prefix}_silu"),
            helper.make_node("Mul", [silu, up], [gated], name=f"{prefix}_swiglu_mul"),
        ]
    )
    if use_smoothquant:
        nodes.append(
            helper.make_node(
                "Mul",
                [gated, f"{prefix}_gated_smooth_inv"],
                [gated_smooth],
                name=f"{prefix}_gated_smooth",
            )
        )
        down_input = gated_smooth
    else:
        down_input = gated
    nodes.extend(
        [
            helper.make_node("MatMul", [down_input, f"{prefix}_w_down"], [down], name=f"{prefix}_down_proj"),
            helper.make_node("Add", [hidden, down], [out], name=f"{prefix}_mlp_residual"),
        ]
    )
    return out


def add_decoder_layer(nodes: list[onnx.NodeProto], prefix: str, hidden: str, use_smoothquant: bool) -> str:
    after_attn = add_attention(nodes, prefix, hidden, use_smoothquant)
    return add_swiglu(nodes, prefix, after_attn, use_smoothquant)


def add_token_embedding(nodes: list[onnx.NodeProto], vocab: int, chunk_size: int | None, chunk_token_embedding: bool) -> str:
    if not chunk_token_embedding:
        nodes.append(helper.make_node("Gather", ["token_embed", "token_ids"], ["hidden0"], name="token_gather", axis=0))
        return "hidden0"

    chunks = vocab_chunks(vocab, chunk_size)
    masked_chunks: list[str] = []
    for index, _ in enumerate(chunks):
        relative = f"token_chunk{index}_relative"
        relative_safe = f"token_chunk{index}_relative_safe"
        gathered = f"token_chunk{index}_gathered"
        ge_start = f"token_chunk{index}_ge_start"
        lt_end = f"token_chunk{index}_lt_end"
        mask_bool = f"token_chunk{index}_mask_bool"
        mask_float = f"token_chunk{index}_mask_float"
        mask = f"token_chunk{index}_mask"
        masked = f"token_chunk{index}_masked"
        nodes.extend(
            [
                helper.make_node("Sub", ["token_ids", f"token_chunk{index}_start"], [relative], name=f"token_chunk{index}_sub_start"),
                helper.make_node(
                    "Greater",
                    ["token_ids", f"token_chunk{index}_start_minus_one"],
                    [ge_start],
                    name=f"token_chunk{index}_ge_start",
                ),
                helper.make_node("Less", ["token_ids", f"token_chunk{index}_end"], [lt_end], name=f"token_chunk{index}_lt_end"),
                helper.make_node("And", [ge_start, lt_end], [mask_bool], name=f"token_chunk{index}_mask_bool"),
                helper.make_node("Where", [mask_bool, relative, "token_zero"], [relative_safe], name=f"token_chunk{index}_safe_index"),
                helper.make_node(
                    "Gather",
                    [f"token_embed_chunk{index}", relative_safe],
                    [gathered],
                    name=f"token_gather_chunk{index}",
                    axis=0,
                ),
                helper.make_node("Cast", [mask_bool], [mask_float], name=f"token_chunk{index}_mask_float", to=TensorProto.FLOAT),
                helper.make_node("Reshape", [mask_float, "shape_token_mask"], [mask], name=f"token_chunk{index}_mask_reshape"),
                helper.make_node("Mul", [gathered, mask], [masked], name=f"token_chunk{index}_apply_mask"),
            ]
        )
        masked_chunks.append(masked)

    hidden = masked_chunks[0]
    for index, masked in enumerate(masked_chunks[1:], start=1):
        out = f"hidden0_chunked_sum{index}"
        nodes.append(helper.make_node("Add", [hidden, masked], [out], name=f"token_chunk{index}_accumulate"))
        hidden = out
    return hidden


def reshape_gamma(value: np.ndarray) -> np.ndarray:
    return value.reshape(1, 1, value.shape[0])


def linear_weight(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.T)


def load_smoothquant(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    arrays = np.load(path)
    return {key: np.asarray(arrays[key], dtype=np.float32) for key in arrays.files if not key.startswith("__")}


def smooth_scale(smoothquant: dict[str, np.ndarray], key: str, size: int) -> np.ndarray:
    if not smoothquant:
        return np.ones(size, dtype=np.float32)
    if key not in smoothquant:
        raise KeyError(f"missing SmoothQuant scale {key!r}")
    scale = np.asarray(smoothquant[key], dtype=np.float32).reshape(-1)
    if scale.shape != (size,):
        raise ValueError(f"SmoothQuant scale {key!r} has shape {scale.shape}, expected {(size,)}")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"SmoothQuant scale {key!r} contains non-positive or non-finite values")
    return scale


def scale_linear_input(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(linear_weight(weight) * scale.reshape(-1, 1), dtype=np.float32)


def vocab_chunks(vocab: int, chunk_size: int | None) -> list[tuple[int, int]]:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= vocab:
        return [(0, vocab)]
    if chunk_size > 65535:
        raise ValueError("--lm-head-chunk-size must be <= 65535 for the T11 verifier-limit test")
    return [(start, min(start + chunk_size, vocab)) for start in range(0, vocab, chunk_size)]


def build_initializers(
    reader: SafeTensorReader,
    config: dict[str, Any],
    seq: int,
    layers: int,
    smoothquant: dict[str, np.ndarray] | None = None,
    lm_head_chunk_size: int | None = None,
    chunk_token_embedding: bool = False,
) -> list[onnx.TensorProto]:
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    vocab = int(config["vocab_size"])
    theta = float(config.get("rope_theta", 10000.0))
    eps = float(config.get("rms_norm_eps", 1e-5))

    rope_cos, rope_sin = rope_tables(seq, head_dim, theta)
    token_embed = reader.tensor("model.embed_tokens.weight")
    if token_embed.shape != (vocab, dim):
        raise ValueError(f"unexpected embedding shape {token_embed.shape}, expected {(vocab, dim)}")

    chunks = vocab_chunks(vocab, lm_head_chunk_size)
    values: list[onnx.TensorProto] = [
        f32_tensor("eps", np.array([eps], dtype=np.float32)),
        f32_tensor("scale_attn", np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)),
        f32_tensor("causal_mask", np.triu(np.full((1, 1, seq, seq), -10000.0, dtype=np.float32), k=1)),
        f32_tensor("rope_cos", rope_cos),
        f32_tensor("rope_sin", rope_sin),
        f32_tensor("final_rms_gamma", reshape_gamma(reader.tensor("model.norm.weight"))),
        i64_tensor("shape_q", [BATCH, seq, n_heads, head_dim]),
        i64_tensor("shape_kv", [BATCH, seq, n_kv_heads, head_dim]),
        i64_tensor("shape_kv_expand", [BATCH, n_kv_heads, 1, seq, head_dim]),
        i64_tensor("kv_tile_repeats", [1, 1, kv_repeat, 1, 1]),
        i64_tensor("shape_heads", [BATCH, n_heads, seq, head_dim]),
        i64_tensor("shape_hidden", [BATCH, seq, dim]),
        i64_tensor("slice_start_0", [0]),
        i64_tensor("slice_start_half", [head_dim // 2]),
        i64_tensor("slice_end_half", [head_dim // 2]),
        i64_tensor("slice_end_head", [head_dim]),
        i64_tensor("slice_axis_last", [3]),
        i64_tensor("slice_step_1", [1]),
        i64_tensor("last_token_start", [seq - 1]),
        i64_tensor("last_token_end", [seq]),
        i64_tensor("sequence_axis", [1]),
    ]
    if chunk_token_embedding:
        values.append(i64_tensor("shape_token_mask", [BATCH, seq, 1]))
        values.append(i32_tensor("token_zero", [0]))
        for index, (start, end) in enumerate(chunks):
            values.extend(
                [
                    f32_tensor(f"token_embed_chunk{index}", token_embed[start:end]),
                    i32_tensor(f"token_chunk{index}_start", [start]),
                    i32_tensor(f"token_chunk{index}_start_minus_one", [start - 1]),
                    i32_tensor(f"token_chunk{index}_end", [end]),
                ]
            )
    else:
        values.append(f32_tensor("token_embed", token_embed))
    if len(chunks) > 1 and not chunk_token_embedding:
        values.append(i64_tensor("embedding_vocab_axis", [0]))
        for index, (start, end) in enumerate(chunks):
            values.extend(
                [
                    i64_tensor(f"lm_head_chunk{index}_start", [start]),
                    i64_tensor(f"lm_head_chunk{index}_end", [end]),
                ]
            )

    smoothquant = smoothquant or {}
    for layer in range(layers):
        hf = f"model.layers.{layer}"
        prefix = f"layer{layer}"
        attn_scale = smooth_scale(smoothquant, f"{prefix}.attn_norm", dim)
        mlp_scale = smooth_scale(smoothquant, f"{prefix}.mlp_norm", dim)
        ctx_scale = smooth_scale(smoothquant, f"{prefix}.ctx", dim)
        gated_scale = smooth_scale(smoothquant, f"{prefix}.gated", int(config["intermediate_size"]))
        values.extend(
            [
                f32_tensor(
                    f"{prefix}_attn_gamma",
                    reshape_gamma(reader.tensor(f"{hf}.input_layernorm.weight") / attn_scale),
                ),
                f32_tensor(
                    f"{prefix}_mlp_gamma",
                    reshape_gamma(reader.tensor(f"{hf}.post_attention_layernorm.weight") / mlp_scale),
                ),
                f32_tensor(f"{prefix}_wq", scale_linear_input(reader.tensor(f"{hf}.self_attn.q_proj.weight"), attn_scale)),
                f32_tensor(f"{prefix}_wk", scale_linear_input(reader.tensor(f"{hf}.self_attn.k_proj.weight"), attn_scale)),
                f32_tensor(f"{prefix}_wv", scale_linear_input(reader.tensor(f"{hf}.self_attn.v_proj.weight"), attn_scale)),
                f32_tensor(f"{prefix}_bq", reader.optional_tensor(f"{hf}.self_attn.q_proj.bias", (dim,))),
                f32_tensor(
                    f"{prefix}_bk",
                    reader.optional_tensor(f"{hf}.self_attn.k_proj.bias", (n_kv_heads * head_dim,)),
                ),
                f32_tensor(
                    f"{prefix}_bv",
                    reader.optional_tensor(f"{hf}.self_attn.v_proj.bias", (n_kv_heads * head_dim,)),
                ),
                f32_tensor(f"{prefix}_wo", scale_linear_input(reader.tensor(f"{hf}.self_attn.o_proj.weight"), ctx_scale)),
                f32_tensor(f"{prefix}_w_gate", scale_linear_input(reader.tensor(f"{hf}.mlp.gate_proj.weight"), mlp_scale)),
                f32_tensor(f"{prefix}_w_up", scale_linear_input(reader.tensor(f"{hf}.mlp.up_proj.weight"), mlp_scale)),
                f32_tensor(f"{prefix}_w_down", scale_linear_input(reader.tensor(f"{hf}.mlp.down_proj.weight"), gated_scale)),
            ]
        )
        if smoothquant:
            values.extend(
                [
                    f32_tensor(f"{prefix}_ctx_smooth_inv", (1.0 / ctx_scale).reshape(1, 1, dim)),
                    f32_tensor(
                        f"{prefix}_gated_smooth_inv",
                        (1.0 / gated_scale).reshape(1, 1, int(config["intermediate_size"])),
                    ),
                ]
            )
    return values


def add_lm_head(
    nodes: list[onnx.NodeProto],
    final_last_token: str,
    vocab: int,
    lm_head_chunk_size: int | None,
    output_mode: str,
    chunk_token_embedding: bool,
) -> list[str]:
    chunks = vocab_chunks(vocab, lm_head_chunk_size)
    if len(chunks) == 1:
        nodes.append(helper.make_node("Transpose", ["token_embed"], ["lm_head"], name="tie_lm_head_transpose", perm=[1, 0]))
        nodes.append(helper.make_node("MatMul", [final_last_token, "lm_head"], ["logits"], name="logits"))
        return ["logits"]

    logits_chunks: list[str] = []
    for index, _ in enumerate(chunks):
        embed_chunk = f"lm_head_embed_chunk{index}"
        head_chunk = f"lm_head_chunk{index}"
        logits_chunk = f"logits_chunk{index}"
        if chunk_token_embedding:
            embed_source = f"token_embed_chunk{index}"
        else:
            nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        "token_embed",
                        f"lm_head_chunk{index}_start",
                        f"lm_head_chunk{index}_end",
                        "embedding_vocab_axis",
                        "slice_step_1",
                    ],
                    [embed_chunk],
                    name=f"slice_lm_head_chunk{index}",
                )
            )
            embed_source = embed_chunk
        nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    [embed_source],
                    [head_chunk],
                    name=f"tie_lm_head_transpose_chunk{index}",
                    perm=[1, 0],
                ),
                helper.make_node("MatMul", [final_last_token, head_chunk], [logits_chunk], name=f"logits_chunk{index}"),
            ]
        )
        logits_chunks.append(logits_chunk)

    if output_mode == "chunks":
        return logits_chunks
    nodes.append(helper.make_node("Concat", logits_chunks, ["logits"], name="concat_logits_chunks", axis=2))
    return ["logits"]


def parse_tokens(value: str, seq: int, vocab: int) -> list[int]:
    tokens = [int(part) for part in value.replace(",", " ").split()]
    if len(tokens) > seq:
        raise argparse.ArgumentTypeError(f"expected at most {seq} token ids")
    invalid = [token for token in tokens if token < 0 or token >= vocab]
    if invalid:
        raise argparse.ArgumentTypeError(f"token ids must be in [0, {vocab - 1}], got {invalid}")
    pad = [0] * (seq - len(tokens))
    return pad + tokens


def default_tokens(config: dict[str, Any], seq: int) -> list[int]:
    bos = int(config.get("bos_token_id", 1))
    eos = int(config.get("eos_token_id", 2))
    seed = [bos, 9690, 198, 2683, 359, 260, 1730, 30, eos]
    seed = [token for token in seed if token < int(config["vocab_size"])]
    return ([0] * max(0, seq - len(seed))) + seed[-seq:]


def initializer_bytes(model: onnx.ModelProto) -> int:
    total = 0
    for tensor in model.graph.initializer:
        if tensor.raw_data:
            total += len(tensor.raw_data)
        else:
            total += len(tensor.SerializeToString())
    return total


def build_block_initializers(
    reader: SafeTensorReader,
    config: dict[str, Any],
    seq: int,
    layer: int,
) -> list[onnx.TensorProto]:
    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    head_dim = dim // n_heads
    kv_repeat = n_heads // n_kv_heads
    theta = float(config.get("rope_theta", 10000.0))
    eps = float(config.get("rms_norm_eps", 1e-5))
    inter = int(config["intermediate_size"])

    rope_cos, rope_sin = rope_tables(seq, head_dim, theta)
    hf = f"model.layers.{layer}"
    prefix = f"layer{layer}"

    return [
        f32_tensor("eps", np.array([eps], dtype=np.float32)),
        f32_tensor("scale_attn", np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)),
        f32_tensor("causal_mask", np.triu(np.full((1, 1, seq, seq), -10000.0, dtype=np.float32), k=1)),
        f32_tensor("rope_cos", rope_cos),
        f32_tensor("rope_sin", rope_sin),
        f32_tensor(f"{prefix}_attn_gamma", reshape_gamma(reader.tensor(f"{hf}.input_layernorm.weight"))),
        f32_tensor(f"{prefix}_mlp_gamma", reshape_gamma(reader.tensor(f"{hf}.post_attention_layernorm.weight"))),
        f32_tensor(f"{prefix}_wq", linear_weight(reader.tensor(f"{hf}.self_attn.q_proj.weight"))),
        f32_tensor(f"{prefix}_wk", linear_weight(reader.tensor(f"{hf}.self_attn.k_proj.weight"))),
        f32_tensor(f"{prefix}_wv", linear_weight(reader.tensor(f"{hf}.self_attn.v_proj.weight"))),
        f32_tensor(f"{prefix}_bq", reader.optional_tensor(f"{hf}.self_attn.q_proj.bias", (dim,))),
        f32_tensor(f"{prefix}_bk", reader.optional_tensor(f"{hf}.self_attn.k_proj.bias", (n_kv_heads * head_dim,))),
        f32_tensor(f"{prefix}_bv", reader.optional_tensor(f"{hf}.self_attn.v_proj.bias", (n_kv_heads * head_dim,))),
        f32_tensor(f"{prefix}_wo", linear_weight(reader.tensor(f"{hf}.self_attn.o_proj.weight"))),
        f32_tensor(f"{prefix}_w_gate", linear_weight(reader.tensor(f"{hf}.mlp.gate_proj.weight"))),
        f32_tensor(f"{prefix}_w_up", linear_weight(reader.tensor(f"{hf}.mlp.up_proj.weight"))),
        f32_tensor(f"{prefix}_w_down", linear_weight(reader.tensor(f"{hf}.mlp.down_proj.weight"))),
        i64_tensor("shape_q", [BATCH, seq, n_heads, head_dim]),
        i64_tensor("shape_kv", [BATCH, seq, n_kv_heads, head_dim]),
        i64_tensor("shape_kv_expand", [BATCH, n_kv_heads, 1, seq, head_dim]),
        i64_tensor("kv_tile_repeats", [1, 1, kv_repeat, 1, 1]),
        i64_tensor("shape_heads", [BATCH, n_heads, seq, head_dim]),
        i64_tensor("shape_hidden", [BATCH, seq, dim]),
        i64_tensor("slice_start_0", [0]),
        i64_tensor("slice_start_half", [head_dim // 2]),
        i64_tensor("slice_end_half", [head_dim // 2]),
        i64_tensor("slice_end_head", [head_dim]),
        i64_tensor("slice_axis_last", [3]),
        i64_tensor("slice_step_1", [1]),
    ]


def build_single_block_onnx(
    model_dir: Path,
    output_dir: Path,
    seq: int,
    layer: int,
) -> None:
    config = read_config(model_dir / "config.json")
    dim = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    if layer < 0 or layer >= layers:
        raise ValueError(f"layer {layer} out of range [0, {layers - 1}]")

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes: list[onnx.NodeProto] = []
    hidden_input = "hidden_in"
    hidden_output = add_decoder_layer(nodes, f"layer{layer}", hidden_input, False)

    inputs = [helper.make_tensor_value_info("hidden_in", TensorProto.FLOAT, [BATCH, seq, dim])]
    outputs = [helper.make_tensor_value_info(hidden_output, TensorProto.FLOAT, [BATCH, seq, dim])]

    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        initializers = build_block_initializers(reader, config, seq, layer)
        graph = helper.make_graph(nodes, f"qwen_block_l{layer}_w{seq}", inputs, outputs, initializers)
    finally:
        reader.close()

    model = helper.make_model(graph, producer_name="a733_npu_driver", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx_path = output_dir / "real_llm.onnx"
    onnx.save(model, onnx_path)
    init_bytes = initializer_bytes(model)

    (output_dir / "inputs_outputs.txt").write_text(
        f"--inputs hidden_in --input-size-list '1 32 896' --outputs {hidden_output}\n",
        encoding="ascii",
    )
    (output_dir / "model_info.json").write_text(
        json.dumps({
            "mode": "single_block",
            "layer": layer,
            "seq_len": seq,
            "hidden_size": dim,
            "onnx_path": str(onnx_path),
            "initializer_bytes": init_bytes,
        }, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(f"wrote single block L={layer} to {onnx_path} ({init_bytes} init bytes)")


def build_embedding_onnx(
    model_dir: Path,
    output_dir: Path,
    seq: int,
    lm_head_chunk_size: int | None,
) -> None:
    config = read_config(model_dir / "config.json")
    vocab = int(config["vocab_size"])
    dim = int(config["hidden_size"])
    chunks = vocab_chunks(vocab, lm_head_chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes: list[onnx.NodeProto] = []
    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        token_embed = reader.tensor("model.embed_tokens.weight")
    finally:
        reader.close()

    initializers: list[onnx.TensorProto] = []
    masked_chunks: list[str] = []
    for index, (start, end) in enumerate(chunks):
        initializers.extend([
            f32_tensor(f"token_embed_chunk{index}", token_embed[start:end]),
            i32_tensor(f"token_chunk{index}_start", [start]),
            i32_tensor(f"token_chunk{index}_start_minus_one", [start - 1]),
            i32_tensor(f"token_chunk{index}_end", [end]),
        ])
        relative = f"token_chunk{index}_relative"
        relative_safe = f"token_chunk{index}_relative_safe"
        gathered = f"token_chunk{index}_gathered"
        ge_start = f"token_chunk{index}_ge_start"
        lt_end = f"token_chunk{index}_lt_end"
        mask_bool = f"token_chunk{index}_mask_bool"
        mask_float = f"token_chunk{index}_mask_float"
        mask = f"token_chunk{index}_mask"
        masked = f"token_chunk{index}_masked"
        nodes.extend([
            helper.make_node("Sub", ["token_ids", f"token_chunk{index}_start"], [relative], name=f"token_chunk{index}_sub_start"),
            helper.make_node("Greater", ["token_ids", f"token_chunk{index}_start_minus_one"], [ge_start], name=f"token_chunk{index}_ge_start"),
            helper.make_node("Less", ["token_ids", f"token_chunk{index}_end"], [lt_end], name=f"token_chunk{index}_lt_end"),
            helper.make_node("And", [ge_start, lt_end], [mask_bool], name=f"token_chunk{index}_mask_bool"),
            helper.make_node("Where", [mask_bool, relative, "token_zero"], [relative_safe], name=f"token_chunk{index}_safe_index"),
            helper.make_node("Gather", [f"token_embed_chunk{index}", relative_safe], [gathered], name=f"token_gather_chunk{index}", axis=0),
            helper.make_node("Cast", [mask_bool], [mask_float], name=f"token_chunk{index}_mask_float", to=TensorProto.FLOAT),
            helper.make_node("Reshape", [mask_float, "shape_token_mask"], [mask], name=f"token_chunk{index}_mask_reshape"),
            helper.make_node("Mul", [gathered, mask], [masked], name=f"token_chunk{index}_apply_mask"),
        ])
        masked_chunks.append(masked)

    initializers.extend([
        i64_tensor("shape_token_mask", [BATCH, seq, 1]),
        i32_tensor("token_zero", [0]),
    ])
    hidden = masked_chunks[0]
    for index, masked in enumerate(masked_chunks[1:], start=1):
        out = f"hidden_emb_chunked_sum{index}"
        nodes.append(helper.make_node("Add", [hidden, masked], [out], name=f"token_chunk{index}_accumulate"))
        hidden = out

    inputs = [helper.make_tensor_value_info("token_ids", TensorProto.INT32, [BATCH, seq])]
    outputs = [helper.make_tensor_value_info(hidden, TensorProto.FLOAT, [BATCH, seq, dim])]
    graph = helper.make_graph(nodes, f"qwen_embedding_w{seq}", inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="a733_npu_driver", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx_path = output_dir / "real_llm.onnx"
    onnx.save(model, onnx_path)
    init_bytes = initializer_bytes(model)
    (output_dir / "inputs_outputs.txt").write_text(
        f"--inputs token_ids --input-size-list {seq} --outputs {hidden}\n", encoding="ascii")
    print(f"wrote embedding stage to {onnx_path} ({init_bytes} init bytes)")


def build_final_onnx(
    model_dir: Path,
    output_dir: Path,
    seq: int,
    lm_head_chunk_size: int | None,
) -> None:
    config = read_config(model_dir / "config.json")
    dim = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    eps = float(config.get("rms_norm_eps", 1e-5))
    chunks = vocab_chunks(vocab, lm_head_chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes: list[onnx.NodeProto] = []
    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        token_embed = reader.tensor("model.embed_tokens.weight")
        final_gamma = reader.tensor("model.norm.weight")
    finally:
        reader.close()

    initializers: list[onnx.TensorProto] = [
        f32_tensor("final_rms_gamma", reshape_gamma(final_gamma)),
        f32_tensor("eps", np.array([eps], dtype=np.float32)),
        i64_tensor("slice_start_0", [0]),
        i64_tensor("slice_step_1", [1]),
        i64_tensor("last_token_start", [seq - 1]),
        i64_tensor("last_token_end", [seq]),
        i64_tensor("sequence_axis", [1]),
        i64_tensor("slice_start_half", [dim // 2]),
    ]

    final = add_rms_norm(nodes, "final_rms", "hidden_in", "final_rms_gamma")
    nodes.append(helper.make_node(
        "Slice", [final, "last_token_start", "last_token_end", "sequence_axis", "slice_step_1"],
        ["final_last_token"], name="slice_last_hidden"))

    logit_output_names: list[str] = []
    for index, (start, end) in enumerate(chunks):
        embed_chunk = f"lm_head_embed_chunk{index}"
        head_chunk = f"lm_head_chunk{index}"
        logits_chunk = f"logits_chunk{index}"
        initializers.extend([
            i64_tensor(f"lm_head_chunk{index}_start", [start]),
            i64_tensor(f"lm_head_chunk{index}_end", [end]),
        ])
        nodes.extend([
            helper.make_node("Slice", ["token_embed", f"lm_head_chunk{index}_start", f"lm_head_chunk{index}_end", "embedding_axis", "slice_step_1"], [embed_chunk], name=f"slice_lm_head_chunk{index}"),
            helper.make_node("Transpose", [embed_chunk], [head_chunk], name=f"tie_lm_head_transpose_chunk{index}", perm=[1, 0]),
            helper.make_node("MatMul", ["final_last_token", head_chunk], [logits_chunk], name=f"logits_chunk{index}"),
        ])
        logit_output_names.append(logits_chunk)

    if len(chunks) > 1:
        initializers.append(i64_tensor("embedding_axis", [0]))
        nodes.append(helper.make_node("Concat", logit_output_names, ["logits"], name="concat_logits_chunks", axis=2))
        logit_name = "logits"
        logit_shape = [BATCH, 1, vocab]
    else:
        logit_name = logit_output_names[0]
        logit_shape = [BATCH, 1, vocab]

    inputs = [helper.make_tensor_value_info("hidden_in", TensorProto.FLOAT, [BATCH, seq, dim])]
    # Add token_embed as initializer
    initializers.append(f32_tensor("token_embed", token_embed))
    outputs = [helper.make_tensor_value_info(logit_name, TensorProto.FLOAT, logit_shape)]
    graph = helper.make_graph(nodes, f"qwen_final_w{seq}", inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="a733_npu_driver", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx_path = output_dir / "real_llm.onnx"
    onnx.save(model, onnx_path)
    init_bytes = initializer_bytes(model)
    (output_dir / "inputs_outputs.txt").write_text(
        f"--inputs hidden_in --input-size-list 32,896 --outputs {logit_name}\n", encoding="ascii")
    print(f"wrote final stage to {onnx_path} ({init_bytes} init bytes)")


def build_model(
    model_dir: Path,
    output_dir: Path,
    seq: int,
    token_values: list[int] | None,
    max_layers: int | None,
    check_model: bool,
    smoothquant_path: Path | None,
    debug_layer_outputs: bool,
    lm_head_chunk_size: int | None,
    lm_head_output_mode: str,
    chunk_token_embedding: bool,
) -> None:
    config = read_config(model_dir / "config.json")
    if config.get("rope_interleaved") not in (None, False):
        raise ValueError("rope_interleaved=true is not supported by this fixed graph generator")
    if not bool(config.get("tie_word_embeddings", False)):
        raise ValueError("only tied embedding/lm_head checkpoints are supported right now")

    dim = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    layers = int(config["num_hidden_layers"]) if max_layers is None else int(max_layers)
    vocab = int(config["vocab_size"])
    if dim % n_heads != 0:
        raise ValueError(f"hidden_size {dim} is not divisible by num_attention_heads {n_heads}")
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"num_attention_heads {n_heads} is not divisible by num_key_value_heads {n_kv_heads}")
    if layers <= 0 or layers > int(config["num_hidden_layers"]):
        raise ValueError(f"--max-layers must be in [1, {config['num_hidden_layers']}]")
    if lm_head_output_mode not in {"concat", "chunks"}:
        raise ValueError("--lm-head-output-mode must be concat or chunks")
    if lm_head_output_mode == "chunks" and not lm_head_chunk_size:
        raise ValueError("--lm-head-output-mode chunks requires --lm-head-chunk-size")
    if chunk_token_embedding and not lm_head_chunk_size:
        raise ValueError("--chunk-token-embedding requires --lm-head-chunk-size")

    smoothquant = load_smoothquant(smoothquant_path)
    use_smoothquant = bool(smoothquant)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray([token_values if token_values is not None else default_tokens(config, seq)], dtype=np.int32)
    if tokens.shape != (BATCH, seq):
        raise ValueError(f"token tensor shape must be {(BATCH, seq)}, got {tokens.shape}")

    nodes: list[onnx.NodeProto] = []
    hidden = add_token_embedding(nodes, vocab, lm_head_chunk_size, chunk_token_embedding)
    debug_outputs: list[onnx.ValueInfoProto] = []
    debug_output_names: list[str] = []
    for layer in range(layers):
        hidden = add_decoder_layer(nodes, f"layer{layer}", hidden, use_smoothquant)
        if debug_layer_outputs:
            debug_outputs.append(helper.make_tensor_value_info(hidden, TensorProto.FLOAT, [BATCH, seq, dim]))
            debug_output_names.append(hidden)
    final = add_rms_norm(nodes, "final_rms", hidden, "final_rms_gamma")
    if debug_layer_outputs:
        debug_outputs.append(helper.make_tensor_value_info(final, TensorProto.FLOAT, [BATCH, seq, dim]))
        debug_output_names.append("final_rms_out")
    nodes.append(
        helper.make_node(
            "Slice",
            [final, "last_token_start", "last_token_end", "sequence_axis", "slice_step_1"],
            ["final_last_token"],
            name="slice_last_hidden",
        )
    )
    logit_output_names = add_lm_head(
        nodes,
        "final_last_token",
        vocab,
        lm_head_chunk_size,
        lm_head_output_mode,
        chunk_token_embedding,
    )
    chunk_ranges = vocab_chunks(vocab, lm_head_chunk_size)
    if logit_output_names == ["logits"]:
        logit_outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, 1, vocab])]
        chunk_output_names = [f"logits_chunk{index}" for index, _ in enumerate(chunk_ranges)] if len(chunk_ranges) > 1 else []
    else:
        logit_outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [BATCH, 1, end - start])
            for name, (start, end) in zip(logit_output_names, chunk_ranges)
        ]
        chunk_output_names = logit_output_names
    model_outputs = logit_outputs + debug_outputs
    output_names = logit_output_names + debug_output_names

    reader = SafeTensorReader(model_dir / "model.safetensors")
    try:
        graph = helper.make_graph(
            nodes,
            f"real_llm_fixed_w{seq}",
            [helper.make_tensor_value_info("token_ids", TensorProto.INT32, [BATCH, seq])],
            model_outputs,
            build_initializers(reader, config, seq, layers, smoothquant, lm_head_chunk_size, chunk_token_embedding),
        )
    finally:
        reader.close()

    model = helper.make_model(
        graph,
        producer_name="a733_npu_driver",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    if check_model:
        onnx.checker.check_model(model)

    onnx_path = output_dir / "real_llm.onnx"
    init_bytes = initializer_bytes(model)
    external_data = None
    if init_bytes >= 1900 * 1024 * 1024:
        external_data = "real_llm.onnx.data"
        old_cwd = Path.cwd()
        try:
            os.chdir(output_dir)
            onnx.save_model(
                model,
                onnx_path.name,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data,
                size_threshold=1024,
            )
        finally:
            os.chdir(old_cwd)
    else:
        onnx.save(model, onnx_path)
    np.save(output_dir / "token_ids.npy", tokens)
    (output_dir / "tokens.txt").write_text(
        " ".join(str(int(value)) for value in tokens.reshape(-1)) + "\n",
        encoding="ascii",
    )
    (output_dir / "dataset.txt").write_text("token_ids.npy\n", encoding="ascii")
    (output_dir / "inputs_outputs.txt").write_text(
        f"--inputs token_ids --input-size-list '{seq}' --outputs {' '.join(output_names)}\n",
        encoding="ascii",
    )
    (output_dir / "model_info.json").write_text(
        json.dumps(
            {
                "source_model_dir": str(model_dir),
                "seq_len": seq,
                "layers": layers,
                "hidden_size": dim,
                "intermediate_size": int(config["intermediate_size"]),
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_kv_heads,
                "head_dim": dim // n_heads,
                "vocab_size": vocab,
                "rope_theta": float(config.get("rope_theta", 10000.0)),
                "rms_norm_eps": float(config.get("rms_norm_eps", 1e-5)),
                "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
                "lm_head": "chunked transpose(model.embed_tokens.weight)" if len(chunk_ranges) > 1 else "transpose(model.embed_tokens.weight)",
                "lm_head_chunk_size": lm_head_chunk_size,
                "lm_head_chunks": len(chunk_ranges),
                "lm_head_chunk_ranges": chunk_ranges,
                "lm_head_chunk_outputs": chunk_output_names,
                "lm_head_output_mode": lm_head_output_mode,
                "token_embedding": "chunked Gather/mask sum" if chunk_token_embedding else "Gather(token_embed)",
                "token_embedding_chunked": chunk_token_embedding,
                "output": "last-token logits",
                "output_names": output_names,
                "debug_layer_outputs": debug_layer_outputs,
                "smoothquant_scales": str(smoothquant_path) if smoothquant_path else None,
                "onnx_path": str(onnx_path),
                "initializer_bytes": init_bytes,
                "external_data": external_data,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path, help="HF directory with config.json and model.safetensors")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=32, help="fixed input window length")
    parser.add_argument("--tokens", help="space- or comma-separated token ids; left-padded to --seq-len")
    parser.add_argument(
        "--max-layers",
        type=int,
        help="diagnostic limit; omit for the real full-depth T4 graph",
    )
    parser.add_argument(
        "--smoothquant-scales",
        type=Path,
        help="NPZ file from make_real_llm_smoothquant_scales.py with per-layer smoothing scales",
    )
    parser.add_argument(
        "--debug-layer-outputs",
        action="store_true",
        help="also expose each decoder layer output and final RMSNorm output for host cosine checks",
    )
    parser.add_argument(
        "--lm-head-chunk-size",
        type=int,
        help="tile tied lm_head into vocab chunks of this size before concatenating logits",
    )
    parser.add_argument(
        "--lm-head-output-mode",
        choices=["concat", "chunks"],
        default="concat",
        help="return chunked lm_head outputs directly instead of concatenating them in the graph",
    )
    parser.add_argument(
        "--chunk-token-embedding",
        action="store_true",
        help="split the token embedding table into vocab chunks and gather from masked chunks",
    )
    parser.add_argument("--no-check", action="store_true", help="skip onnx.checker for very large debug builds")
    parser.add_argument(
        "--export-block",
        type=int,
        help="export a single decoder block layer N (0-indexed) as a standalone ONNX",
    )
    parser.add_argument(
        "--export-embedding",
        action="store_true",
        help="export token embedding stage as standalone ONNX (tokens -> hidden)",
    )
    parser.add_argument(
        "--export-final",
        action="store_true",
        help="export final RMSNorm + lm_head stage as standalone ONNX (hidden -> logits)",
    )
    args = parser.parse_args()

    config = read_config(args.model_dir / "config.json")

    if args.export_block is not None:
        build_single_block_onnx(
            args.model_dir,
            args.output_dir,
            args.seq_len,
            args.export_block,
        )
        return

    if args.export_embedding:
        build_embedding_onnx(args.model_dir, args.output_dir, args.seq_len, args.lm_head_chunk_size)
        return

    if args.export_final:
        build_final_onnx(args.model_dir, args.output_dir, args.seq_len, args.lm_head_chunk_size)
        return

    tokens = parse_tokens(args.tokens, args.seq_len, int(config["vocab_size"])) if args.tokens else None
    build_model(
        args.model_dir,
        args.output_dir,
        args.seq_len,
        tokens,
        args.max_layers,
        not args.no_check,
        args.smoothquant_scales,
        args.debug_layer_outputs,
        args.lm_head_chunk_size,
        args.lm_head_output_mode,
        args.chunk_token_embedding,
    )


if __name__ == "__main__":
    main()
