"""Qwen3-MoE weight bridge: engine packed bytes -> ``reference_models.Qwen3Moe``.

Attention maps like qwen3 (per-head qk-norm gains direct, projections
transposed). The MoE tail is ALL raw parameters in this reference — the
router ``w_router (d, E)`` and the expert stacks ``w13_experts (E, d, 2F)``
/ ``w2_experts (E, F, d)`` are already in the engine's packed orientation
and load direct (contrast olmoe, whose router is an ``nn.Linear``).
"""
from __future__ import annotations

import torch

from reference_models.qwen3moe import Qwen3Moe, Qwen3MoeConfig

from ..bridge_common import load_state_dict_strict, transposed


def reference_qwen3moe_config(cfg) -> Qwen3MoeConfig:
    """Build the isolated reference config from a ``ShapedQwen3MoeConfig``."""
    return Qwen3MoeConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        d_ff_expert=cfg.d_ff_expert, n_experts=cfg.n_experts, top_k=cfg.top_k,
        vocab_size=cfg.vocab_size, rope_base=cfg.rope_base,
    )


def build_qwen3moe_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen3Moe:
    return Qwen3Moe(reference_qwen3moe_config(cfg)).to(device=device, dtype=dtype)


def to_qwen3moe_state_dict(cfg, get_bytes) -> dict:
    """Reference Qwen3-MoE state_dict from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        qwen3moe_weight_layout,
    )
    from .model import dims_of_qwen3moe

    dims = dims_of_qwen3moe(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        w = qwen3moe_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"{p}.attn.wq.weight"] = transposed(w["wq"])
        sd[f"{p}.attn.wk.weight"] = transposed(w["wk"])
        sd[f"{p}.attn.wv.weight"] = transposed(w["wv"])
        sd[f"{p}.attn.q_norm.weight"] = w["q_norm_w"].clone()   # per-head (head_dim,)
        sd[f"{p}.attn.k_norm.weight"] = w["k_norm_w"].clone()
        sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"{p}.moe.w_router"] = w["w_router"].clone()         # raw (d, E) direct
        sd[f"{p}.moe.w13_experts"] = w["w13_experts"].clone()   # (E, d, 2F) direct
        sd[f"{p}.moe.w2_experts"] = w["w2_experts"].clone()     # (E, F, d) direct
    return sd


def load_qwen3moe_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_qwen3moe_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen3Moe:
    return build_qwen3moe_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_qwen3moe_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_qwen3moe_state_dict(cfg, get_bytes)
