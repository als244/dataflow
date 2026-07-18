"""OLMoE weight bridge: engine packed bytes -> ``reference_models.Olmoe``.

Deltas from the dense bridges: the FULL-ROW qk-norm gains
(``q_norm_w (q_dim,)`` / ``k_norm_w (kv_dim,)``, loaded direct) and the MoE
tail — the router is an ``nn.Linear`` (transposed), while the stacked expert
weights ``w13_experts (E, d, 2F)`` / ``w2_experts (E, F, d)`` are raw
parameters already in the engine's packed orientation (loaded direct).
"""
from __future__ import annotations

import torch

from reference_models.olmoe import Olmoe, OlmoeConfig

from ..bridge_common import load_state_dict_strict, transposed


def reference_olmoe_config(cfg) -> OlmoeConfig:
    """Build the isolated reference config from a ``ShapedOlmoeConfig``."""
    return OlmoeConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        vocab_size=cfg.vocab_size, rope_base=cfg.rope_base,
    )


def build_olmoe_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Olmoe:
    return Olmoe(reference_olmoe_config(cfg)).to(device=device, dtype=dtype)


def to_olmoe_state_dict(cfg, get_bytes) -> dict:
    """Reference OLMoE state_dict from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        olmoe_weight_layout,
    )
    from .model import dims_of_olmoe

    dims = dims_of_olmoe(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        w = olmoe_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"{p}.attn.wq.weight"] = transposed(w["wq"])
        sd[f"{p}.attn.wk.weight"] = transposed(w["wk"])
        sd[f"{p}.attn.wv.weight"] = transposed(w["wv"])
        sd[f"{p}.attn.q_norm.weight"] = w["q_norm_w"].clone()   # full-row (q_dim,)
        sd[f"{p}.attn.k_norm.weight"] = w["k_norm_w"].clone()   # full-row (kv_dim,)
        sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"{p}.moe.router.weight"] = transposed(w["w_router"])
        sd[f"{p}.moe.w13_experts"] = w["w13_experts"].clone()   # (E, d, 2F) direct
        sd[f"{p}.moe.w2_experts"] = w["w2_experts"].clone()     # (E, F, d) direct
    return sd


def load_olmoe_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_olmoe_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Olmoe:
    return build_olmoe_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_olmoe_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_olmoe_state_dict(cfg, get_bytes)
