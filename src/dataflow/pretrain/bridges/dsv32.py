"""DeepSeek-V3.2 weight bridge: engine packed bytes -> ``reference_models.Dsv32``.

dsv3's MLA mapping plus the DSA lightning indexer in EVERY layer:
``w_idx_q`` / ``w_idx_k`` are ``nn.Linear`` (transposed), the key-LayerNorm
gain/bias and the fp32 per-head weights ``w_idx_w (d, H_I)`` load direct. In
this reference the WHOLE MoE tail is raw parameters in the engine
orientation (router, expert stacks, shared expert — all direct; contrast
dsv3, whose router/shared are Linear); the balance bias is the buffer
``mlp.w_router_bias`` (fp32). The FFN attribute is ``mlp`` here (dsv3 uses
``ffn``).
"""
from __future__ import annotations

import torch

from reference_models.dsv32 import Dsv32, Dsv32Config

from .common import load_state_dict_strict, transposed


def reference_dsv32_config(cfg) -> Dsv32Config:
    """Build the isolated reference config from a ``ShapedDsv32Config``."""
    return Dsv32Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim, d_ff_dense=cfg.d_ff_dense,
        first_k_dense=cfg.first_k_dense, n_experts=cfg.n_experts,
        top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert, n_group=cfg.n_group,
        topk_group=cfg.topk_group, n_shared_experts=cfg.n_shared_experts,
        d_ff_shared=cfg.d_ff_shared, index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim, index_topk=cfg.index_topk,
        vocab_size=cfg.vocab_size, routed_scaling=cfg.routed_scaling,
        rope_base=cfg.rope_base,
    )


def restore_fp32_islands(model: Dsv32) -> None:
    """``Module.to(dtype=bf16)`` downcasts every float tensor; the engine's
    dtype policy keeps the indexer per-head weights ``w_idx_w`` and the noaux
    balance bias fp32 (and the reference file declares them fp32) — restore
    them after the blanket cast."""
    for module in model.modules():
        if hasattr(module, "w_idx_w"):
            module.w_idx_w.data = module.w_idx_w.data.float()
        if hasattr(module, "w_router_bias"):
            module.w_router_bias.data = module.w_router_bias.data.float()


def build_dsv32_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Dsv32:
    model = Dsv32(reference_dsv32_config(cfg)).to(device=device, dtype=dtype)
    restore_fp32_islands(model)
    return model


def to_dsv32_state_dict(cfg, get_bytes) -> dict:
    """Reference DeepSeek-V3.2 state_dict from the engine's packed weight objects."""
    from dataflow.tasks.layouts import (
        dsv32_dense_weight_layout,
        dsv32_moe_weight_layout,
        embed_weight_layout,
        head_weight_layout,
    )
    from dataflow.training.models.dsv32 import dims_of_dsv32

    dims = dims_of_dsv32(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        dense = dims.kinds[i] == "dense"
        layout = dsv32_dense_weight_layout if dense else dsv32_moe_weight_layout
        w = layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"{p}.attn.w_q_a.weight"] = transposed(w["w_q_a"])
        sd[f"{p}.attn.q_a_norm.weight"] = w["q_a_norm_w"].clone()
        sd[f"{p}.attn.w_q_b.weight"] = transposed(w["w_q_b"])
        sd[f"{p}.attn.w_kv_a.weight"] = transposed(w["w_kv_a"])
        sd[f"{p}.attn.kv_a_norm.weight"] = w["kv_a_norm_w"].clone()
        sd[f"{p}.attn.w_kv_b.weight"] = transposed(w["w_kv_b"])
        sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])
        # DSA lightning indexer (every layer)
        sd[f"{p}.attn.w_idx_q.weight"] = transposed(w["w_idx_q"])
        sd[f"{p}.attn.w_idx_k.weight"] = transposed(w["w_idx_k"])
        sd[f"{p}.attn.idx_k_ln_w"] = w["idx_k_ln_w"].clone()
        sd[f"{p}.attn.idx_k_ln_b"] = w["idx_k_ln_b"].clone()
        sd[f"{p}.attn.w_idx_w"] = w["w_idx_w"].clone()          # (d, H_I) fp32
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        if dense:
            sd[f"{p}.mlp.w1.weight"] = transposed(w["w1"])
            sd[f"{p}.mlp.w3.weight"] = transposed(w["w3"])
            sd[f"{p}.mlp.w2.weight"] = transposed(w["w2"])
        else:
            sd[f"{p}.mlp.w_router"] = w["w_router"].clone()          # raw (d, E)
            sd[f"{p}.mlp.w_router_bias"] = w["w_router_bias"].clone()  # buffer fp32
            sd[f"{p}.mlp.w13_experts"] = w["w13_experts"].clone()
            sd[f"{p}.mlp.w2_experts"] = w["w2_experts"].clone()
            sd[f"{p}.mlp.w_s13"] = w["w_s13"].clone()                # raw (d, 2Fs)
            sd[f"{p}.mlp.w_s2"] = w["w_s2"].clone()                  # raw (Fs, d)
    return sd


def load_dsv32_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_dsv32_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow.pretrain.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Dsv32:
    return build_dsv32_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_dsv32_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_dsv32_state_dict(cfg, get_bytes)
