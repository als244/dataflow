"""llama3-dense weight bridge: engine packed bytes -> ``reference_models.Llama3``.

Projection weights transpose ``(in, out) -> (out, in)``; the embedding /
LM-head tables ``(vocab, d)`` and the 1-D norm gains load directly. The
byte-identity gate compares the loaded parameters against the unpacked
field values.
"""
from __future__ import annotations

import torch

from reference_models.llama3 import Llama3, Llama3Config

from ..bridge_common import assert_state_dict_byte_identical, load_state_dict_strict, transposed


def reference_config(cfg) -> Llama3Config:
    """Build the isolated reference config from a ``ShapedLlamaConfig``."""
    from .model import derive_dims

    dims = derive_dims(cfg)
    return Llama3Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, d_ff=cfg.d_ff, vocab_size=cfg.vocab_size,
        rope_base=dims.rope_base,
    )


def build_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Llama3:
    """A reference model at the engine's storage dtype (all-bf16), on
    ``device``. Weights are then loaded via ``load_engine_init``."""
    return Llama3(reference_config(cfg)).to(device=device, dtype=dtype)


def to_state_dict(dims, n_layers: int, get_bytes) -> dict:
    """Reference ``state_dict`` from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        weight_layout,
    )

    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()            # (vocab, d) direct
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(n_layers):
        w = weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"blocks.{i}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"blocks.{i}.attn.wq.weight"] = transposed(w["wq"])
        sd[f"blocks.{i}.attn.wk.weight"] = transposed(w["wk"])
        sd[f"blocks.{i}.attn.wv.weight"] = transposed(w["wv"])
        sd[f"blocks.{i}.attn.wo.weight"] = transposed(w["wo"])
        sd[f"blocks.{i}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"blocks.{i}.mlp.w1.weight"] = transposed(w["w1"])
        sd[f"blocks.{i}.mlp.w3.weight"] = transposed(w["w3"])
        sd[f"blocks.{i}.mlp.w2.weight"] = transposed(w["w2"])
    return sd


def load_engine_init(model: Llama3, dims, n_layers: int, get_bytes) -> Llama3:
    """Load the engine's packed init into ``model`` (strict; raises on any
    key/shape mismatch)."""
    return load_state_dict_strict(model, to_state_dict(dims, n_layers, get_bytes))


def assert_byte_identical(model: Llama3, dims, n_layers: int, get_bytes) -> None:
    """Gate: every reference parameter equals the engine's packed bytes
    (bit-for-bit, up to the projection transpose)."""
    assert_state_dict_byte_identical(model, to_state_dict(dims, n_layers, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Llama3:
    return build_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes) -> Llama3:
    return load_engine_init(model, dims, cfg.n_layers, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    from .model import derive_dims

    return to_state_dict(derive_dims(cfg), cfg.n_layers, get_bytes)
