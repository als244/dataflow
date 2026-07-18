"""qwen3-dense weight bridge: engine packed bytes -> ``reference_models.Qwen3``.

Deltas from llama3 that reach the bridge: the two per-head qk-norm gains
(``q_norm_w``/``k_norm_w``, one ``(head_dim,)`` vector each, loaded direct)
and the decoupled head_dim (``wq`` is ``(d_model, n_heads*head_dim)``; the
transpose handles it like any projection).
"""
from __future__ import annotations

import torch

from reference_models.qwen3 import Qwen3, Qwen3Config

from .common import load_state_dict_strict, transposed


def reference_qwen3_config(cfg) -> Qwen3Config:
    """Build the isolated reference config from a ``ShapedQwen3Config``."""
    from dataflow_training.model_families.qwen3 import dims_of_qwen3

    dims = dims_of_qwen3(cfg)
    return Qwen3Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size, rope_base=dims.rope_base,
    )


def build_qwen3_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen3:
    return Qwen3(reference_qwen3_config(cfg)).to(device=device, dtype=dtype)


def to_qwen3_state_dict(cfg, get_bytes) -> dict:
    """Reference qwen3 state_dict from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        qwen3_weight_layout,
    )
    from dataflow_training.model_families.qwen3 import dims_of_qwen3

    dims = dims_of_qwen3(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()            # (vocab, d) direct
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        w = qwen3_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"{p}.attn.wq.weight"] = transposed(w["wq"])
        sd[f"{p}.attn.wk.weight"] = transposed(w["wk"])
        sd[f"{p}.attn.wv.weight"] = transposed(w["wv"])
        sd[f"{p}.attn.q_norm.weight"] = w["q_norm_w"].clone()
        sd[f"{p}.attn.k_norm.weight"] = w["k_norm_w"].clone()
        sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"{p}.mlp.w1.weight"] = transposed(w["w1"])
        sd[f"{p}.mlp.w3.weight"] = transposed(w["w3"])
        sd[f"{p}.mlp.w2.weight"] = transposed(w["w2"])
    return sd


def load_qwen3_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_qwen3_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen3:
    return build_qwen3_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_qwen3_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_qwen3_state_dict(cfg, get_bytes)
