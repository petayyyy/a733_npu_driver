#!/usr/bin/env python3
"""Generate a tiny fixed-shape transformer decoder block ONNX model.

The model is intentionally small and deterministic. It starts from an embedding
tensor and produces token logits, which makes it useful for probing whether
ACUITY/VIPLite can compile and run decoder-layer compute on the A733 NPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
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


def build_model(output_dir: Path) -> None:
    rng = np.random.default_rng(733)
    seq = 4
    dim = 8
    hidden = 16
    vocab = 16

    output_dir.mkdir(parents=True, exist_ok=True)

    initializers = [
        tensor("eps", np.array([1e-5], dtype=np.float32)),
        tensor("half", np.array([0.5], dtype=np.float32)),
        tensor("one", np.array([1.0], dtype=np.float32)),
        tensor("scale_attn", np.array([1.0 / np.sqrt(dim)], dtype=np.float32)),
        tensor("inv_sqrt2", np.array([1.0 / np.sqrt(2.0)], dtype=np.float32)),
        tensor("ln1_gamma", np.ones((1, 1, dim), dtype=np.float32)),
        tensor("ln1_beta", np.zeros((1, 1, dim), dtype=np.float32)),
        tensor("ln2_gamma", np.ones((1, 1, dim), dtype=np.float32)),
        tensor("ln2_beta", np.zeros((1, 1, dim), dtype=np.float32)),
        tensor("lnf_gamma", np.ones((1, 1, dim), dtype=np.float32)),
        tensor("lnf_beta", np.zeros((1, 1, dim), dtype=np.float32)),
        tensor("causal_mask", np.triu(np.full((1, seq, seq), -10000.0, dtype=np.float32), k=1)),
    ]

    for name, shape in [
        ("wq", (dim, dim)),
        ("wk", (dim, dim)),
        ("wv", (dim, dim)),
        ("wo", (dim, dim)),
        ("w1", (dim, hidden)),
        ("w2", (hidden, dim)),
        ("w_logits", (dim, vocab)),
    ]:
        initializers.append(tensor(name, rng.normal(0.0, 0.15, size=shape).astype(np.float32)))

    nodes = []

    ln1 = add_layer_norm(nodes, "ln1", "hidden_in", "ln1_gamma", "ln1_beta", "eps")
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
            helper.make_node("Add", ["hidden_in", "attn_out"], ["resid1"], name="residual_attn"),
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

    graph = helper.make_graph(
        nodes,
        "tiny_decoder_block",
        [helper.make_tensor_value_info("hidden_in", TensorProto.FLOAT, [1, seq, dim])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, seq, vocab])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="a733_npu_driver",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    sample = rng.normal(0.0, 0.4, size=(1, seq, dim)).astype(np.float32)

    onnx.save(model, output_dir / "tiny_decoder_block.onnx")
    np.save(output_dir / "hidden_in.npy", sample)
    (output_dir / "dataset.txt").write_text("hidden_in.npy\n", encoding="ascii")
    (output_dir / "inputs_outputs.txt").write_text(
        "--inputs hidden_in --input-size-list '4,8' --outputs logits\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_model(args.output_dir)


if __name__ == "__main__":
    main()
