#!/usr/bin/env python3
"""CPU oracle for the SmolLM2 fixed-window graph used in T4.

This is intentionally a validation tool, not a deliverable inference path. It
mirrors the static ONNX graph semantics: a fixed rightmost token window, RoPE
positions 0..W-1 for that window, no KV cache, and greedy argmax from last-token
logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

import numpy as np
try:
    from tokenizers import Tokenizer as HfTokenizer
except ImportError:  # pragma: no cover - depends on host environment.
    HfTokenizer = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
BOARD_SCRIPT_DIR = SCRIPT_DIR.parent / "board"
if str(BOARD_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPT_DIR))

from smollm2_chat import SmolTokenizer  # noqa: E402

DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
QWEN2_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class SafeTensorReader:
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


class Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FallbackTokenizer:
    def __init__(self, tokenizer_json: Path) -> None:
        self._fallback = SmolTokenizer(tokenizer_json)

    @classmethod
    def from_file(cls, path: str) -> "FallbackTokenizer":
        return cls(Path(path))

    def encode(self, text: str) -> Encoded:
        return Encoded(self._fallback.encode(text))

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return self._fallback.decode(ids, skip_special=skip_special_tokens)


Tokenizer = HfTokenizer if HfTokenizer is not None else FallbackTokenizer


def parse_ids(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def render_chat(user: str, system: str | None, add_generation_prompt: bool) -> str:
    parts: list[str] = []
    if system is not None:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_qwen2_chat(
    user: str,
    system: str = QWEN2_DEFAULT_SYSTEM,
    add_generation_prompt: bool = True,
) -> str:
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * gamma


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    positions = np.arange(seq, dtype=np.float32)
    inv_freq = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float32) / half))
    freqs = np.outer(positions, inv_freq)
    angles = np.concatenate([freqs, freqs], axis=1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * cos[None, :, :]) + (rotated * sin[None, :, :])


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class FixedWindowSmolLM2:
    def __init__(self, model_dir: Path, seq_len: int) -> None:
        self.model_dir = model_dir
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.seq_len = seq_len
        self.dim = int(self.config["hidden_size"])
        self.intermediate = int(self.config["intermediate_size"])
        self.layers = int(self.config["num_hidden_layers"])
        self.n_heads = int(self.config["num_attention_heads"])
        self.n_kv_heads = int(self.config.get("num_key_value_heads", self.n_heads))
        self.head_dim = self.dim // self.n_heads
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.vocab = int(self.config["vocab_size"])
        self.eps = float(self.config.get("rms_norm_eps", 1e-5))
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))
        self.cos, self.sin = rope_tables(seq_len, self.head_dim, float(self.config.get("rope_theta", 10000.0)))
        self.mask = np.triu(np.full((self.seq_len, self.seq_len), -10000.0, dtype=np.float32), k=1)
        self.weights = self._load_weights()

    def _load_weights(self) -> dict[str, Any]:
        reader = SafeTensorReader(self.model_dir / "model.safetensors")
        try:
            weights: dict[str, Any] = {
                "embed": reader.tensor("model.embed_tokens.weight"),
                "final_norm": reader.tensor("model.norm.weight"),
                "layers": [],
            }
            for layer in range(self.layers):
                prefix = f"model.layers.{layer}"
                weights["layers"].append(
                    {
                        "attn_norm": reader.tensor(f"{prefix}.input_layernorm.weight"),
                        "mlp_norm": reader.tensor(f"{prefix}.post_attention_layernorm.weight"),
                        "q": reader.tensor(f"{prefix}.self_attn.q_proj.weight"),
                        "k": reader.tensor(f"{prefix}.self_attn.k_proj.weight"),
                        "v": reader.tensor(f"{prefix}.self_attn.v_proj.weight"),
                        "qb": reader.optional_tensor(f"{prefix}.self_attn.q_proj.bias", (self.dim,)),
                        "kb": reader.optional_tensor(
                            f"{prefix}.self_attn.k_proj.bias",
                            (self.n_kv_heads * self.head_dim,),
                        ),
                        "vb": reader.optional_tensor(
                            f"{prefix}.self_attn.v_proj.bias",
                            (self.n_kv_heads * self.head_dim,),
                        ),
                        "o": reader.tensor(f"{prefix}.self_attn.o_proj.weight"),
                        "gate": reader.tensor(f"{prefix}.mlp.gate_proj.weight"),
                        "up": reader.tensor(f"{prefix}.mlp.up_proj.weight"),
                        "down": reader.tensor(f"{prefix}.mlp.down_proj.weight"),
                    }
                )
            return weights
        finally:
            reader.close()

    def forward(self, token_window: list[int]) -> np.ndarray:
        hidden = self.weights["embed"][np.asarray(token_window, dtype=np.int64)].astype(np.float32)
        for layer in self.weights["layers"]:
            norm = rms_norm(hidden, layer["attn_norm"], self.eps)
            q = (norm @ layer["q"].T) + layer["qb"]
            k = (norm @ layer["k"].T) + layer["kb"]
            v = (norm @ layer["v"].T) + layer["vb"]
            q = q.reshape(self.seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(self.seq_len, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(self.seq_len, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)
            q = apply_rope(q, self.cos, self.sin)
            k = apply_rope(k, self.cos, self.sin)
            k = np.repeat(k, self.kv_repeat, axis=0)
            v = np.repeat(v, self.kv_repeat, axis=0)
            scores = (q @ np.swapaxes(k, 1, 2)) * self.scale
            probs = softmax(scores + self.mask[None, :, :], axis=-1)
            ctx = (probs @ v).transpose(1, 0, 2).reshape(self.seq_len, self.dim)
            hidden = hidden + (ctx @ layer["o"].T)

            norm = rms_norm(hidden, layer["mlp_norm"], self.eps)
            gate = norm @ layer["gate"].T
            up = norm @ layer["up"].T
            hidden = hidden + ((silu(gate) * up) @ layer["down"].T)

        final = rms_norm(hidden, self.weights["final_norm"], self.eps)
        return final[-1] @ self.weights["embed"].T


def topk(logits: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return [(int(i), float(logits[i])) for i in idx]


def resolve_prompt(args: argparse.Namespace, tokenizer: Tokenizer) -> tuple[str, list[int]]:
    if args.prompt_ids:
        ids = parse_ids(args.prompt_ids)
        return tokenizer.decode(ids, skip_special_tokens=False), ids
    if args.text is not None:
        text = args.text
    else:
        if args.chat_format == "qwen2":
            system = QWEN2_DEFAULT_SYSTEM if args.default_system or args.system is None else args.system
            text = render_qwen2_chat(args.chat_user, system, not args.no_generation_prompt)
        else:
            system = DEFAULT_SYSTEM if args.default_system else args.system
            text = render_chat(args.chat_user, system, not args.no_generation_prompt)
    return text, tokenizer.encode(text).ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("work/models/smollm2-135m-instruct"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt-ids")
    group.add_argument("--text")
    group.add_argument("--chat-user")
    parser.add_argument("--chat-format", choices=["smollm2", "qwen2"], default="smollm2")
    parser.add_argument("--system")
    parser.add_argument("--default-system", action="store_true")
    parser.add_argument("--no-generation-prompt", action="store_true")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--pad-token", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    prompt_text, prompt_ids = resolve_prompt(args, tokenizer)
    model = FixedWindowSmolLM2(args.model_dir, args.seq_len)

    tokens = list(prompt_ids)
    if len(tokens) < args.seq_len:
        tokens = [args.pad_token] * (args.seq_len - len(tokens)) + tokens
    else:
        tokens = tokens[-args.seq_len :]

    records: list[dict[str, Any]] = []
    print("prompt_text:")
    print(prompt_text)
    print("initial_tokens:")
    print(" ".join(str(token) for token in tokens))
    for step in range(args.steps):
        window = tokens[-args.seq_len :]
        logits = model.forward(window)
        next_token = int(np.argmax(logits))
        best = topk(logits, args.top_k)
        records.append({"step": step, "window": window, "next": next_token, "topk": best})
        print(
            f"step={step} next={next_token} top{args.top_k}="
            + ",".join(f"{idx}:{value:.6f}" for idx, value in best)
        )
        tokens.append(next_token)

    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    print("final_tokens:")
    print(" ".join(str(token) for token in tokens))
    print("decoded:")
    print(decoded)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "prompt_text": prompt_text,
                    "initial_tokens": tokens[: args.seq_len],
                    "final_tokens": tokens,
                    "decoded": decoded,
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )


if __name__ == "__main__":
    main()
