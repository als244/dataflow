"""DeepSeek-V3 weight bridge: engine packed bytes -> ``reference_models.Dsv3``.

MLA low-rank stacks and the two mid-stack latent norms map per layer; the
depth is MIXED (``first_k_dense`` dense-SwiGLU layers, then MoE). In this
reference the router and the shared expert are ``nn.Linear`` (transposed),
the routed stacks ``w13_experts -> ffn.w13`` / ``w2_experts -> ffn.w2`` are
raw parameters (direct), and the noaux balance bias is a BUFFER
(``ffn.router_bias``, fp32, loaded direct — state_dict covers buffers).
"""
from __future__ import annotations

import torch

from reference_models.dsv3 import Dsv3, Dsv3Config

from .common import load_state_dict_strict, transposed


def reference_dsv3_config(cfg) -> Dsv3Config:
    """Build the isolated reference config from a ``ShapedDsv3Config``."""
    return Dsv3Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim, first_k_dense=cfg.first_k_dense,
        d_ff_dense=cfg.d_ff_dense, n_experts=cfg.n_experts, top_k=cfg.top_k,
        d_ff_expert=cfg.d_ff_expert, n_group=cfg.n_group,
        topk_group=cfg.topk_group, n_shared_experts=cfg.n_shared_experts,
        d_ff_shared=cfg.d_ff_shared, vocab_size=cfg.vocab_size,
        routed_scaling=cfg.routed_scaling, rope_base=cfg.rope_base,
        aux_coef=cfg.aux_coef, bias_update_speed=cfg.bias_update_speed,
    )


def restore_fp32_islands(model: Dsv3) -> None:
    """``Module.to(dtype=bf16)`` downcasts every float tensor; the engine's
    dtype policy keeps the noaux balance bias fp32 (bf16 ulp at bias ~0.1 is
    half the 1e-3 sign-rule step) and the reference file declares it fp32 for
    the same reason — restore it after the blanket cast."""
    for module in model.modules():
        if hasattr(module, "router_bias"):
            module.router_bias.data = module.router_bias.data.float()


def build_dsv3_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Dsv3:
    model = Dsv3(reference_dsv3_config(cfg)).to(device=device, dtype=dtype)
    restore_fp32_islands(model)
    return model


def mla_attention_entries(sd: dict, p: str, w: dict) -> None:
    """The MLA attention mapping shared by every layer kind (this family's
    reference wraps each projection in ``nn.Linear``)."""
    sd[f"{p}.attn.w_q_a.weight"] = transposed(w["w_q_a"])
    sd[f"{p}.attn.q_a_norm.weight"] = w["q_a_norm_w"].clone()
    sd[f"{p}.attn.w_q_b.weight"] = transposed(w["w_q_b"])
    sd[f"{p}.attn.w_kv_a.weight"] = transposed(w["w_kv_a"])
    sd[f"{p}.attn.kv_a_norm.weight"] = w["kv_a_norm_w"].clone()
    sd[f"{p}.attn.w_kv_b.weight"] = transposed(w["w_kv_b"])
    sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])


def to_dsv3_state_dict(cfg, get_bytes) -> dict:
    """Reference DeepSeek-V3 state_dict from the engine's packed weight objects."""
    from dataflow.tasks.layouts import (
        dsv3_dense_weight_layout,
        dsv3_moe_weight_layout,
        embed_weight_layout,
        head_weight_layout,
    )
    from dataflow.training.models.dsv3 import dims_of_dsv3

    dims = dims_of_dsv3(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        dense = dims.kind_of(i) == "dense"
        layout = dsv3_dense_weight_layout if dense else dsv3_moe_weight_layout
        w = layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        mla_attention_entries(sd, p, w)
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        if dense:
            sd[f"{p}.ffn.w1.weight"] = transposed(w["w1"])
            sd[f"{p}.ffn.w3.weight"] = transposed(w["w3"])
            sd[f"{p}.ffn.w2.weight"] = transposed(w["w2"])
        else:
            sd[f"{p}.ffn.router.weight"] = transposed(w["w_router"])
            sd[f"{p}.ffn.router_bias"] = w["w_router_bias"].clone()  # buffer, fp32
            sd[f"{p}.ffn.w13"] = w["w13_experts"].clone()            # (E, d, 2F)
            sd[f"{p}.ffn.w2"] = w["w2_experts"].clone()              # (E, F, d)
            sd[f"{p}.ffn.w_s13.weight"] = transposed(w["w_s13"])
            sd[f"{p}.ffn.w_s2.weight"] = transposed(w["w_s2"])
    return sd


def load_dsv3_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_dsv3_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow.pretrain.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Dsv3:
    return build_dsv3_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_dsv3_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_dsv3_state_dict(cfg, get_bytes)
