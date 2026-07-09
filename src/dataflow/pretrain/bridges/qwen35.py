"""qwen3.5-dense (hybrid) weight bridge: engine packed bytes ->
``reference_models.Qwen35``.

Projections transpose ``(in, out) -> (out, in)``; the depthwise conv
reshapes ``(D, W) -> (D, 1, W)``; 1-D params (A_log / dt_bias / norm gains)
load direct; embed/head tables are ``(vocab, d)`` direct (the tied config
packs both into ``W_embed``).
"""
from __future__ import annotations

import torch

from reference_models.qwen35 import Qwen35, Qwen35Config

from .common import load_state_dict_strict, transposed


def reference_qwen35_config(cfg) -> Qwen35Config:
    """Build the isolated qwen3.5 reference config from a ShapedQwen35Config."""
    return Qwen35Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel, d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size, rope_base=cfg.rope_base,
        tied_embeddings=cfg.tied_embeddings,
    )


def build_qwen35_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen35:
    return Qwen35(reference_qwen35_config(cfg)).to(device=device, dtype=dtype)


def to_qwen35_state_dict(cfg, get_bytes) -> dict:
    """Reference qwen3.5 state_dict from the engine's packed weight objects."""
    from dataflow.tasks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        qwen35_attn_weight_layout,
        qwen35_lin_weight_layout,
    )
    from dataflow.training.models.qwen35 import dims_of_qwen35

    dims = dims_of_qwen35(cfg)
    sd: dict[str, torch.Tensor] = {}
    if cfg.tied_embeddings:
        ew = head_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
        sd["embed.weight"] = ew["w"].clone()
        sd["lm_head.weight"] = ew["w"].clone()          # tied (shared param)
        sd["final_norm.weight"] = ew["final_norm_w"].clone()
    else:
        ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
        sd["embed.weight"] = ew["w"].clone()
        hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
        sd["lm_head.weight"] = hw["w"].clone()
        sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        if dims.kind_of(i) == "lin":
            w = qwen35_lin_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.w_qkvz.weight"] = transposed(w["w_qkvz"])
            sd[f"{p}.mixer.w_ba.weight"] = transposed(w["w_ba"])
            sd[f"{p}.mixer.conv.weight"] = w["w_conv"].unsqueeze(1).contiguous()
            sd[f"{p}.mixer.A_log"] = w["A_log"].clone()
            sd[f"{p}.mixer.dt_bias"] = w["dt_bias"].clone()
            sd[f"{p}.mixer.lin_norm.weight"] = w["lin_norm_w"].clone()
            sd[f"{p}.mixer.w_out.weight"] = transposed(w["w_out"])
            sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        else:
            w = qwen35_attn_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.wq.weight"] = transposed(w["wq"])
            sd[f"{p}.mixer.wk.weight"] = transposed(w["wk"])
            sd[f"{p}.mixer.wv.weight"] = transposed(w["wv"])
            sd[f"{p}.mixer.q_norm.weight"] = w["q_norm_w"].clone()
            sd[f"{p}.mixer.k_norm.weight"] = w["k_norm_w"].clone()
            sd[f"{p}.mixer.wo.weight"] = transposed(w["wo"])
            sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        for nm in ("w1", "w3", "w2"):
            sd[f"{p}.mlp.{nm}.weight"] = transposed(w[nm])
    return sd


def load_qwen35_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_qwen35_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow.pretrain.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Qwen35:
    return build_qwen35_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_qwen35_init(model, cfg, get_bytes)
