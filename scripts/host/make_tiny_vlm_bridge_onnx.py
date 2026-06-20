#!/usr/bin/env python3
"""Generate a tiny fixed-shape VLM bridge ONNX probe.

The graph accepts an image embedding and token IDs, then runs the VLM-side
projector/adapter, token embedding lookup, decoder block, and logits projection
inside one static ONNX graph. It is intended to validate the NPU-only bridge
between a separately validated MobileCLIP-S0 NPU encoder and an NPU language
decoder path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TEXT_SEQ = 4
TOTAL_SEQ = 5
DIM = 8
HIDDEN = 16
VOCAB = 16
IMAGE_DIM = 512


def f32_tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name=name)


def add_layer_norm(nodes, prefix: str, x: str, gamma: str, beta: str, eps: str) -> str:
    mean = f"{prefix}_mean"
    centered = f"{prefix}_centered"
    squared = f"{prefix}_squared"
    var = f"{prefix}_var"
    var_eps = f"{prefix}_var_eps"
    denom = f"{prefix}_denom"
    norm = f"{prefix}_norm"
    scaled = f"{prefix}_scaled"
    out = f"{prefix}_out"

    nodes.extend(
        [
            helper.make_node("ReduceMean", [x], [mean], name=f"{prefix}_mean", axes=[2], keepdims=1),
            helper.make_node("Sub", [x, mean], [centered], name=f"{prefix}_sub"),
            helper.make_node("Mul", [centered, centered], [squared], name=f"{prefix}_square"),
            helper.make_node("ReduceMean", [squared], [var], name=f"{prefix}_var", axes=[2], keepdims=1),
            helper.make_node("Add", [var, eps], [var_eps], name=f"{prefix}_eps"),
            helper.make_node("Sqrt", [var_eps], [denom], name=f"{prefix}_sqrt"),
            helper.make_node("Div", [centered, denom], [norm], name=f"{prefix}_div"),
            helper.make_node("Mul", [norm, gamma], [scaled], name=f"{prefix}_scale"),
            helper.make_node("Add", [scaled, beta], [out], name=f"{prefix}_bias"),
        ]
    )
    return out


def add_gelu(nodes, prefix: str, x: str, inv_sqrt2: str, half: str, one: str) -> str:
    scaled = f"{prefix}_scaled"
    erf = f"{prefix}_erf"
    one_plus = f"{prefix}_one_plus"
    gated = f"{prefix}_gated"
    out = f"{prefix}_out"

    nodes.extend(
        [
            helper.make_node("Mul", [x, inv_sqrt2], [scaled], name=f"{prefix}_mul_inv_sqrt2"),
            helper.make_node("Erf", [scaled], [erf], name=f"{prefix}_erf"),
            helper.make_node("Add", [erf, one], [one_plus], name=f"{prefix}_one_plus"),
            helper.make_node("Mul", [x, one_plus], [gated], name=f"{prefix}_gate"),
            helper.make_node("Mul", [gated, half], [out], name=f"{prefix}_half"),
        ]
    )
    return out


def add_decoder_block(nodes, hidden_in: str) -> str:
    ln1 = add_layer_norm(nodes, "ln1", hidden_in, "ln1_gamma", "ln1_beta", "eps")
    nodes.extend(
        [
            helper.make_node("MatMul", [ln1, "wq"], ["q"], name="q_proj"),
            helper.make_node("MatMul", [ln1, "wk"], ["k"], name="k_proj"),
            helper.make_node("MatMul", [ln1, "wv"], ["v"], name="v_proj"),
            helper.make_node("Transpose", ["k"], ["k_t"], name="k_transpose", perm=[0, 2, 1]),
            helper.make_node("MatMul", ["q", "k_t"], ["attn_scores_raw"], name="attn_scores"),
            helper.make_node("Mul", ["attn_scores_raw", "scale_attn"], ["attn_scores_scaled"], name="attn_scale"),
            helper.make_node("Add", ["attn_scores_scaled", "causal_mask"], ["attn_scores_masked"], name="causal_mask"),
            helper.make_node("Softmax", ["attn_scores_masked"], ["attn_probs"], name="attn_softmax", axis=2),
            helper.make_node("MatMul", ["attn_probs", "v"], ["attn_ctx"], name="attn_context"),
            helper.make_node("MatMul", ["attn_ctx", "wo"], ["attn_out"], name="out_proj"),
            helper.make_node("Add", [hidden_in, "attn_out"], ["resid1"], name="residual_attn"),
        ]
    )

    ln2 = add_layer_norm(nodes, "ln2", "resid1", "ln2_gamma", "ln2_beta", "eps")
    nodes.append(helper.make_node("MatMul", [ln2, "w1"], ["mlp_fc1"], name="mlp_fc1"))
    gelu = add_gelu(nodes, "gelu", "mlp_fc1", "inv_sqrt2", "half", "one")
    nodes.extend(
        [
            helper.make_node("MatMul", [gelu, "w2"], ["mlp_out"], name="mlp_fc2"),
            helper.make_node("Add", ["resid1", "mlp_out"], ["resid2"], name="residual_mlp"),
        ]
    )

    lnf = add_layer_norm(nodes, "lnf", "resid2", "lnf_gamma", "lnf_beta", "eps")
    nodes.append(helper.make_node("MatMul", [lnf, "w_logits"], ["logits"], name="logits"))
    return "logits"


def initializers(rng: np.random.Generator) -> list[onnx.TensorProto]:
    values = [
        f32_tensor("eps", np.array([1e-5], dtype=np.float32)),
        f32_tensor("half", np.array([0.5], dtype=np.float32)),
        f32_tensor("one", np.array([1.0], dtype=np.float32)),
        f32_tensor("scale_attn", np.array([1.0 / np.sqrt(DIM)], dtype=np.float32)),
        f32_tensor("inv_sqrt2", np.array([1.0 / np.sqrt(2.0)], dtype=np.float32)),
        f32_tensor("ln1_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
        f32_tensor("ln1_beta", np.zeros((1, 1, DIM), dtype=np.float32)),
        f32_tensor("ln2_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
        f32_tensor("ln2_beta", np.zeros((1, 1, DIM), dtype=np.float32)),
        f32_tensor("lnf_gamma", np.ones((1, 1, DIM), dtype=np.float32)),
        f32_tensor("lnf_beta", np.zeros((1, 1, DIM), dtype=np.float32)),
        f32_tensor("causal_mask", np.triu(np.full((1, TOTAL_SEQ, TOTAL_SEQ), -10000.0, dtype=np.float32), k=1)),
        f32_tensor("token_embed", rng.normal(0.0, 0.35, size=(VOCAB, DIM)).astype(np.float32)),
        f32_tensor("text_pos_embed", rng.normal(0.0, 0.08, size=(1, TEXT_SEQ, DIM)).astype(np.float32)),
        f32_tensor("image_proj", rng.normal(0.0, 0.05, size=(IMAGE_DIM, DIM)).astype(np.float32)),
        f32_tensor("image_proj_bias", rng.normal(0.0, 0.03, size=(1, DIM)).astype(np.float32)),
    ]

    for name, shape in [
        ("wq", (DIM, DIM)),
        ("wk", (DIM, DIM)),
        ("wv", (DIM, DIM)),
        ("wo", (DIM, DIM)),
        ("w1", (DIM, HIDDEN)),
        ("w2", (HIDDEN, DIM)),
        ("w_logits", (DIM, VOCAB)),
    ]:
        values.append(f32_tensor(name, rng.normal(0.0, 0.15, size=shape).astype(np.float32)))

    return values


def build_model(output_dir: Path) -> None:
    rng = np.random.default_rng(733)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.array([[1, 5, 9, 2]], dtype=np.int32)
    image_embed = rng.normal(0.0, 0.08, size=(1, IMAGE_DIM)).astype(np.float32)

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Gather", ["token_embed", "token_ids"], ["token_embeds"], name="token_gather", axis=0),
        helper.make_node("Add", ["token_embeds", "text_pos_embed"], ["text_hidden"], name="add_text_pos_embed"),
        helper.make_node("MatMul", ["image_embed", "image_proj"], ["image_proj_raw"], name="image_projector"),
        helper.make_node("Add", ["image_proj_raw", "image_proj_bias"], ["image_projected"], name="image_projector_bias"),
        helper.make_node("Reshape", ["image_projected", "image_prefix_shape"], ["image_prefix"], name="image_prefix_reshape"),
        helper.make_node("Concat", ["image_prefix", "text_hidden"], ["hidden_in"], name="vlm_prefix_concat", axis=1),
    ]
    logits = add_decoder_block(nodes, "hidden_in")

    init = initializers(rng)
    init.append(numpy_helper.from_array(np.array([1, 1, DIM], dtype=np.int64), name="image_prefix_shape"))

    graph = helper.make_graph(
        nodes,
        "tiny_vlm_bridge",
        [
            helper.make_tensor_value_info("image_embed", TensorProto.FLOAT, [1, IMAGE_DIM]),
            helper.make_tensor_value_info("token_ids", TensorProto.INT32, [1, TEXT_SEQ]),
        ],
        [helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, TOTAL_SEQ, VOCAB])],
        init,
    )
    model = helper.make_model(
        graph,
        producer_name="a733_npu_driver",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx.save(model, output_dir / "tiny_vlm_bridge.onnx")
    np.save(output_dir / "image_embed.npy", image_embed)
    np.save(output_dir / "token_ids.npy", tokens)
    (output_dir / "tokens.txt").write_text(
        " ".join(str(int(value)) for value in tokens.reshape(-1)) + "\n",
        encoding="ascii",
    )
    (output_dir / "dataset.txt").write_text("image_embed.npy token_ids.npy\n", encoding="ascii")
    (output_dir / "dataset0.txt").write_text("image_embed.npy\n", encoding="ascii")
    (output_dir / "dataset1.txt").write_text("token_ids.npy\n", encoding="ascii")
    (output_dir / "image_embed_dataset.txt").write_text("image_embed.npy\n", encoding="ascii")
    (output_dir / "token_ids_dataset.txt").write_text("token_ids.npy\n", encoding="ascii")
    (output_dir / "inputs_outputs.txt").write_text(
        "--inputs 'image_embed token_ids' --input-size-list '512#4' --outputs logits\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_model(args.output_dir)


if __name__ == "__main__":
    main()
