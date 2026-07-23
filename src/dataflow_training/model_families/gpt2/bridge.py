"""GPT-2 weight bridge: engine packed bytes -> ``reference_models.Gpt2``.

Projection weights transpose ``(in, out) -> (out, in)``; the (vocab, d)
and (n_ctx, d) tables, biases, and LayerNorm gain/bias vectors load
directly. Tied configs load the shared table into BOTH wte and lm_head
(the twin aliases them, so the second load is an identical rewrite).
"""
from __future__ import annotations

import torch

from reference_models.gpt2 import Gpt2, Gpt2Config

from ..bridge_common import assert_state_dict_byte_identical, load_state_dict_strict, transposed


def reference_config(cfg) -> Gpt2Config:
    return Gpt2Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_ff=cfg.d_ff, vocab_size=cfg.vocab_size, n_ctx=cfg.max_seq_len,
        tied=cfg.tied_embeddings, use_bias=cfg.use_bias,
    )


def build_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Gpt2:
    return Gpt2(reference_config(cfg)).to(device=device, dtype=dtype)


def to_state_dict(dims, n_layers: int, get_bytes) -> dict:
    """Reference ``state_dict`` from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        gpt2_embed_layout,
        gpt2_head_layout,
        gpt2_weight_layout,
    )

    sd: dict[str, torch.Tensor] = {}
    ew = gpt2_embed_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["wte.weight"] = ew["w"].clone()
    sd["wpe.weight"] = ew["wpe"].clone()
    if dims.tied:
        sd["lm_head.weight"] = ew["w"].clone()
        sd["final_norm.weight"] = ew["final_norm_w"].clone()
        if dims.use_bias:
            sd["final_norm.bias"] = ew["final_norm_b"].clone()
    else:
        hw = gpt2_head_layout(dims).unpack_tensor(get_bytes("W_head"))
        sd["lm_head.weight"] = hw["w"].clone()            # (vocab, d) direct
        sd["final_norm.weight"] = hw["final_norm_w"].clone()
        if dims.use_bias:
            sd["final_norm.bias"] = hw["final_norm_b"].clone()
    for i in range(n_layers):
        w = gpt2_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"blocks.{i}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"blocks.{i}.attn.c_attn.weight"] = transposed(w["w_qkv"])
        sd[f"blocks.{i}.attn.c_proj.weight"] = transposed(w["wo"])
        sd[f"blocks.{i}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"blocks.{i}.mlp.c_fc.weight"] = transposed(w["w_fc"])
        sd[f"blocks.{i}.mlp.c_proj.weight"] = transposed(w["w_proj"])
        if dims.use_bias:
            sd[f"blocks.{i}.attn_norm.bias"] = w["attn_norm_b"].clone()
            sd[f"blocks.{i}.attn.c_attn.bias"] = w["b_qkv"].clone()
            sd[f"blocks.{i}.attn.c_proj.bias"] = w["b_o"].clone()
            sd[f"blocks.{i}.ffn_norm.bias"] = w["ffn_norm_b"].clone()
            sd[f"blocks.{i}.mlp.c_fc.bias"] = w["b_fc"].clone()
            sd[f"blocks.{i}.mlp.c_proj.bias"] = w["b_proj"].clone()
    return sd


def load_engine_init(model: Gpt2, dims, n_layers: int, get_bytes) -> Gpt2:
    """Load the engine's packed init into ``model`` (strict; raises on any
    key/shape mismatch)."""
    return load_state_dict_strict(model, to_state_dict(dims, n_layers, get_bytes))


def assert_byte_identical(model: Gpt2, dims, n_layers: int, get_bytes) -> None:
    """Gate: every reference parameter equals the engine's packed bytes
    (bit-for-bit, up to the projection transpose)."""
    assert_state_dict_byte_identical(model, to_state_dict(dims, n_layers, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Gpt2:
    return build_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes) -> Gpt2:
    return load_engine_init(model, dims, cfg.n_layers, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    from .model import derive_dims

    return to_state_dict(derive_dims(cfg), cfg.n_layers, get_bytes)
