#!/usr/bin/env python3
"""Generate tiny fixed-shape language-model ONNX probes.

Two input variants are supported:

* ``gather``: integer token IDs feed an ONNX Gather over a token embedding
  table, then a decoder block produces logits.
* ``onehot``: a one-hot token tensor feeds an embedding MatMul, then the same
  decoder block produces logits.

The one-hot form is useful when a compiler/runtime cannot accept integer token
inputs but can still run all model-layer math, including the embedding
projection, on the NPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SEQ = 4
DIM = 8
HIDDEN = 16
VOCAB = 16


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
        f32_tensor("causal_mask", np.triu(np.full((1, SEQ, SEQ), -10000.0, dtype=np.float32), k=1)),
        f32_tensor("token_embed", rng.normal(0.0, 0.35, size=(VOCAB, DIM)).astype(np.float32)),
        f32_tensor("pos_embed", rng.normal(0.0, 0.08, size=(1, SEQ, DIM)).astype(np.float32)),
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


def write_common_metadata(output_dir: Path, variant: str, tokens: np.ndarray) -> None:
    (output_dir / "tokens.txt").write_text(
        " ".join(str(int(value)) for value in tokens.reshape(-1)) + "\n",
        encoding="ascii",
    )
    if variant == "gather":
        np.save(output_dir / "token_ids.npy", tokens.astype(np.int32))
        (output_dir / "dataset.txt").write_text("token_ids.npy\n", encoding="ascii")
        (output_dir / "inputs_outputs.txt").write_text(
            "--inputs token_ids --input-size-list '4' --outputs logits\n",
            encoding="ascii",
        )
    else:
        onehot = np.eye(VOCAB, dtype=np.float32)[tokens.reshape(-1)].reshape(1, SEQ, VOCAB)
        np.save(output_dir / "token_onehot.npy", onehot)
        (output_dir / "dataset.txt").write_text("token_onehot.npy\n", encoding="ascii")
        (output_dir / "inputs_outputs.txt").write_text(
            "--inputs token_onehot --input-size-list '4,16' --outputs logits\n",
            encoding="ascii",
        )


def build_model(output_dir: Path, variant: str) -> None:
    rng = np.random.default_rng(733)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.array([[1, 5, 9, 2]], dtype=np.int32)

    nodes: list[onnx.NodeProto] = []
    inputs: list[onnx.ValueInfoProto]
    if variant == "gather":
        inputs = [helper.make_tensor_value_info("token_ids", TensorProto.INT32, [1, SEQ])]
        nodes.append(helper.make_node("Gather", ["token_embed", "token_ids"], ["token_embeds"], name="token_gather", axis=0))
    else:
        inputs = [helper.make_tensor_value_info("token_onehot", TensorProto.FLOAT, [1, SEQ, VOCAB])]
        nodes.append(helper.make_node("MatMul", ["token_onehot", "token_embed"], ["token_embeds"], name="token_embed_matmul"))

    nodes.append(helper.make_node("Add", ["token_embeds", "pos_embed"], ["hidden_in"], name="add_pos_embed"))
    logits = add_decoder_block(nodes, "hidden_in")

    graph = helper.make_graph(
        nodes,
        f"tiny_lm_{variant}",
        inputs,
        [helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, SEQ, VOCAB])],
        initializers(rng),
    )
    model = helper.make_model(
        graph,
        producer_name="a733_npu_driver",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx.save(model, output_dir / f"tiny_lm_{variant}.onnx")
    write_common_metadata(output_dir, variant, tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", choices=["gather", "onehot"], default="gather")
    args = parser.parse_args()
    build_model(args.output_dir, args.variant)


if __name__ == "__main__":
    main()
