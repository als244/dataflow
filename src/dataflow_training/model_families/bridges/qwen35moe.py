"""Qwen3.5-MoE weight bridge: engine packed bytes ->
``reference_models.Qwen35Moe``.

The two mixer kinds map like the dense qwen3.5 bridge (DeltaNet: qkvz/ba/out
projections transposed, depthwise conv ``(D, W) -> (D, 1, W)``, A_log /
dt_bias / gated-norm gain direct; gated attention: doubled ``wq`` and the
other projections transposed, per-head qk-norm gains direct). The MoE tail:
router + the three shared-expert projections are ``nn.Linear`` (transposed);
the routed stacks ``w13_experts -> moe.w13`` / ``w2_experts -> moe.w2`` are
raw parameters in the engine orientation (direct). Untied embeddings only
(the 35B config).
"""
from __future__ import annotations

import torch

from reference_models.qwen35moe import Qwen35Moe, Qwen35MoeConfig

from .common import load_state_dict_strict, transposed


def reference_qwen35moe_config(cfg) -> Qwen35MoeConfig:
    """Build the isolated reference config from a ``ShapedQwen35MoeConfig``."""
    return Qwen35MoeConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel,
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        vocab_size=cfg.vocab_size, routing_mode=cfg.routing_mode,
        aux_coef=cfg.aux_coef, rope_base=cfg.rope_base,
        tied_embeddings=cfg.tied_embeddings,
    )


def build_qwen35moe_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen35Moe:
    return Qwen35Moe(reference_qwen35moe_config(cfg)).to(device=device, dtype=dtype)


def to_qwen35moe_state_dict(cfg, get_bytes) -> dict:
    """Reference Qwen3.5-MoE state_dict from the engine's packed weight objects."""
    from dataflow_training.blocks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        qwen35moe_attn_weight_layout,
        qwen35moe_lin_weight_layout,
    )
    from dataflow_training.model_families.qwen35moe import dims_of_qwen35moe

    if cfg.tied_embeddings:
        raise NotImplementedError("qwen35moe is untied (the 35B config)")
    dims = dims_of_qwen35moe(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        if dims.kinds[i] == "lin":
            w = qwen35moe_lin_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.w_qkvz.weight"] = transposed(w["w_qkvz"])
            sd[f"{p}.mixer.w_ba.weight"] = transposed(w["w_ba"])
            sd[f"{p}.mixer.conv.weight"] = w["w_conv"].unsqueeze(1).contiguous()
            sd[f"{p}.mixer.A_log"] = w["A_log"].clone()
            sd[f"{p}.mixer.dt_bias"] = w["dt_bias"].clone()
            sd[f"{p}.mixer.lin_norm.weight"] = w["lin_norm_w"].clone()
            sd[f"{p}.mixer.w_out.weight"] = transposed(w["w_out"])
        else:
            w = qwen35moe_attn_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.wq.weight"] = transposed(w["wq"])       # doubled: [Q | gate]
            sd[f"{p}.mixer.wk.weight"] = transposed(w["wk"])
            sd[f"{p}.mixer.wv.weight"] = transposed(w["wv"])
            sd[f"{p}.mixer.q_norm.weight"] = w["q_norm_w"].clone()
            sd[f"{p}.mixer.k_norm.weight"] = w["k_norm_w"].clone()
            sd[f"{p}.mixer.wo.weight"] = transposed(w["wo"])
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"{p}.moe.router.weight"] = transposed(w["w_router"])
        sd[f"{p}.moe.w13"] = w["w13_experts"].clone()              # (E, d, 2F) direct
        sd[f"{p}.moe.w2"] = w["w2_experts"].clone()                # (E, F, d) direct
        if cfg.n_shared_experts:
            sd[f"{p}.moe.shared_gate.weight"] = transposed(w["w_shared_gate"])
            sd[f"{p}.moe.shared_up.weight"] = transposed(w["w_s13"])
            sd[f"{p}.moe.shared_down.weight"] = transposed(w["w_s2"])
    return sd


def load_qwen35moe_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_qwen35moe_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow_training.model_families.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen35Moe:
    return build_qwen35moe_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_qwen35moe_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_qwen35moe_state_dict(cfg, get_bytes)
