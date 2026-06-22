#!/usr/bin/env python3
"""Generate an architecturally faithful tiny fixed-shape decoder LM ONNX model.

The graph mirrors the small-model operator set used by Qwen/Llama/SmolLM style
decoders: RMSNorm, RoPE, multi-head causal attention with GQA, SwiGLU, token
embedding Gather, and logits MatMul.  It is intentionally tiny and deterministic
so ACUITY/VIPLite compatibility can be tested separately from model scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


BATCH = 1
SEQ = 16
DIM = 64
LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = DIM // N_HEADS
KV_REPEAT = N_HEADS // N_KV_HEADS
INTERMEDIATE = 192
VOCAB = 256
DEFAULT_TOKENS = (1, 5, 9, 2, 13, 21, 34, 55, 89, 144, 233, 3, 8, 15, 42, 7)


def f32_tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name=name)


def i64_tensor(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def rope_tables() -> tuple[np.ndarray, np.ndarray]:
    half = HEAD_DIM // 2
    positions = np.arange(SEQ, dtype=np.float32)
    inv_freq = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    freqs = np.outer(positions, inv_freq)
    angles = np.concatenate([freqs, freqs], axis=1)
    cos = np.cos(angles).reshape(1, 1, SEQ, HEAD_DIM)
    sin = np.sin(angles).reshape(1, 1, SEQ, HEAD_DIM)
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


def add_attention(nodes: list[onnx.NodeProto], prefix: str, hidden: str) -> str:
    norm = add_rms_norm(nodes, f"{prefix}_attn_rms", hidden, f"{prefix}_attn_gamma")
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
            helper.make_node("MatMul", [norm, f"{prefix}_wq"], [q], name=f"{prefix}_q_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_wk"], [k], name=f"{prefix}_k_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_wv"], [v], name=f"{prefix}_v_proj"),
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
            helper.make_node("MatMul", [ctx_flat, f"{prefix}_wo"], [attn_out], name=f"{prefix}_out_proj"),
            helper.make_node("Add", [hidden, attn_out], [out], name=f"{prefix}_attn_residual"),
        ]
    )
    return out


def add_swiglu(nodes: list[onnx.NodeProto], prefix: str, hidden: str) -> str:
    norm = add_rms_norm(nodes, f"{prefix}_mlp_rms", hidden, f"{prefix}_mlp_gamma")
    gate = f"{prefix}_gate"
    up = f"{prefix}_up"
    gate_sigmoid = f"{prefix}_gate_sigmoid"
    silu = f"{prefix}_silu"
    gated = f"{prefix}_gated"
    down = f"{prefix}_down"
    out = f"{prefix}_mlp_resid"

    nodes.extend(
        [
            helper.make_node("MatMul", [norm, f"{prefix}_w_gate"], [gate], name=f"{prefix}_gate_proj"),
            helper.make_node("MatMul", [norm, f"{prefix}_w_up"], [up], name=f"{prefix}_up_proj"),
            helper.make_node("Sigmoid", [gate], [gate_sigmoid], name=f"{prefix}_sigmoid"),
            helper.make_node("Mul", [gate, gate_sigmoid], [silu], name=f"{prefix}_silu"),
            helper.make_node("Mul", [silu, up], [gated], name=f"{prefix}_swiglu_mul"),
            helper.make_node("MatMul", [gated, f"{prefix}_w_down"], [down], name=f"{prefix}_down_proj"),
            helper.make_node("Add", [hidden, down], [out], name=f"{prefix}_mlp_residual"),
        ]
    )
    return out


def add_decoder_layer(nodes: list[onnx.NodeProto], prefix: str, hidden: str) -> str:
    after_attn = add_attention(nodes, prefix, hidden)
    return add_swiglu(nodes, prefix, after_attn)


def initializers(rng: np.random.Generator) -> list[onnx.TensorProto]:
    rope_cos, rope_sin = rope_tables()
    values: list[onnx.TensorProto] = [
        f32_tensor("eps", np.array([1e-5], dtype=np.float32)),
        f32_tensor("scale_attn", np.array([1.0 / np.sqrt(HEAD_DIM)], dtype=np.float32)),
        f32_tensor("causal_mask", np.triu(np.full((1, 1, SEQ, SEQ), -10000.0, dtype=np.float32), k=1)),
        f32_tensor("rope_cos", rope_cos),
        f32_tensor("rope_sin", rope_sin),
        f32_tensor("token_embed", rng.normal(0.0, 0.16, size=(VOCAB, DIM)).astype(np.float32)),
        f32_tensor("final_rms_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
        f32_tensor("lm_head", rng.normal(0.0, 0.10, size=(DIM, VOCAB)).astype(np.float32)),
        i64_tensor("shape_q", [BATCH, SEQ, N_HEADS, HEAD_DIM]),
        i64_tensor("shape_kv", [BATCH, SEQ, N_KV_HEADS, HEAD_DIM]),
        i64_tensor("shape_kv_expand", [BATCH, N_KV_HEADS, 1, SEQ, HEAD_DIM]),
        i64_tensor("kv_tile_repeats", [1, 1, KV_REPEAT, 1, 1]),
        i64_tensor("shape_heads", [BATCH, N_HEADS, SEQ, HEAD_DIM]),
        i64_tensor("shape_hidden", [BATCH, SEQ, DIM]),
        i64_tensor("slice_start_0", [0]),
        i64_tensor("slice_start_half", [HEAD_DIM // 2]),
        i64_tensor("slice_end_half", [HEAD_DIM // 2]),
        i64_tensor("slice_end_head", [HEAD_DIM]),
        i64_tensor("slice_axis_last", [3]),
        i64_tensor("slice_step_1", [1]),
        i64_tensor("last_token_start", [SEQ - 1]),
        i64_tensor("last_token_end", [SEQ]),
        i64_tensor("sequence_axis", [1]),
    ]

    for layer in range(LAYERS):
        prefix = f"layer{layer}"
        values.extend(
            [
                f32_tensor(f"{prefix}_attn_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
                f32_tensor(f"{prefix}_mlp_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
                f32_tensor(f"{prefix}_wq", rng.normal(0.0, 0.08, size=(DIM, DIM)).astype(np.float32)),
                f32_tensor(
                    f"{prefix}_wk",
                    rng.normal(0.0, 0.08, size=(DIM, N_KV_HEADS * HEAD_DIM)).astype(np.float32),
                ),
                f32_tensor(
                    f"{prefix}_wv",
                    rng.normal(0.0, 0.08, size=(DIM, N_KV_HEADS * HEAD_DIM)).astype(np.float32),
                ),
                f32_tensor(f"{prefix}_wo", rng.normal(0.0, 0.08, size=(DIM, DIM)).astype(np.float32)),
                f32_tensor(f"{prefix}_w_gate", rng.normal(0.0, 0.07, size=(DIM, INTERMEDIATE)).astype(np.float32)),
                f32_tensor(f"{prefix}_w_up", rng.normal(0.0, 0.07, size=(DIM, INTERMEDIATE)).astype(np.float32)),
                f32_tensor(f"{prefix}_w_down", rng.normal(0.0, 0.07, size=(INTERMEDIATE, DIM)).astype(np.float32)),
            ]
        )
    return values


def parse_tokens(value: str) -> list[int]:
    tokens = [int(part) for part in value.replace(",", " ").split()]
    if len(tokens) != SEQ:
        raise argparse.ArgumentTypeError(f"expected exactly {SEQ} token ids")
    invalid = [token for token in tokens if token < 0 or token >= VOCAB]
    if invalid:
        raise argparse.ArgumentTypeError(f"token ids must be in [0, {VOCAB - 1}], got {invalid}")
    return tokens


def build_model(output_dir: Path, logits: str, seed: int, token_values: list[int]) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray([token_values], dtype=np.int32)

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Gather", ["token_embed", "token_ids"], ["hidden0"], name="token_gather", axis=0)
    ]
    hidden = "hidden0"
    for layer in range(LAYERS):
        hidden = add_decoder_layer(nodes, f"layer{layer}", hidden)
    final = add_rms_norm(nodes, "final_rms", hidden, "final_rms_gamma")
    logits_shape = [BATCH, SEQ, VOCAB]
    logits_input = final
    if logits == "last":
        logits_input = "final_last_token"
        logits_shape = [BATCH, 1, VOCAB]
        nodes.append(
            helper.make_node(
                "Slice",
                [final, "last_token_start", "last_token_end", "sequence_axis", "slice_step_1"],
                [logits_input],
                name="slice_last_hidden",
            )
        )
    nodes.append(helper.make_node("MatMul", [logits_input, "lm_head"], ["logits"], name="logits"))

    graph = helper.make_graph(
        nodes,
        "tiny_faithful_block",
        [helper.make_tensor_value_info("token_ids", TensorProto.INT32, [BATCH, SEQ])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, logits_shape)],
        initializers(rng),
    )
    model = helper.make_model(
        graph,
        producer_name="a733_npu_driver",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx.save(model, output_dir / "tiny_faithful_block.onnx")
    np.save(output_dir / "token_ids.npy", tokens)
    (output_dir / "tokens.txt").write_text(
        " ".join(str(int(value)) for value in tokens.reshape(-1)) + "\n",
        encoding="ascii",
    )
    (output_dir / "dataset.txt").write_text("token_ids.npy\n", encoding="ascii")
    (output_dir / "inputs_outputs.txt").write_text(
        "--inputs token_ids --input-size-list '16' --outputs logits\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--logits",
        choices=["full", "last"],
        default="full",
        help="emit logits for the full window or only the last autoregressive position",
    )
    parser.add_argument("--seed", type=int, default=733, help="deterministic RNG seed for generated weights")
    parser.add_argument(
        "--tokens",
        type=parse_tokens,
        default=list(DEFAULT_TOKENS),
        help="16 token ids for the validation/calibration input; accepts space- or comma-separated values",
    )
    args = parser.parse_args()
    build_model(args.output_dir, args.logits, args.seed, args.tokens)


if __name__ == "__main__":
    main()
