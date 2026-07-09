"""GLM-5.2 weight bridge: engine packed bytes -> ``reference_models.Glm52``.

Per-layer KIND-dependent mapping (IndexShare): LEADER layers
(``indexer_types[i] == "full"``) carry the DSA lightning indexer —
``w_idx_q`` / ``w_idx_k`` Linear (transposed), LayerNorm gain/bias and the
per-head weights ``w_idx_w (d, H_I)`` direct — FOLLOWERS carry none. The
engine layouts mirror this: ``gdl``/``gml`` (leader dense/moe) include the
idx fields, ``gmf`` (follower moe) does not. In this reference the MoE
router and the shared expert are ``nn.Linear`` (transposed) while the
routed stacks stay raw (direct); the balance bias is the buffer
``ffn.w_router_bias``.
"""
from __future__ import annotations

import torch

from reference_models.glm52 import Glm52, Glm52Config

from .common import load_state_dict_strict, transposed


def reference_glm52_config(cfg) -> Glm52Config:
    """Build the isolated reference config from a ``ShapedGlm52Config``."""
    return Glm52Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim, d_ff=cfg.d_ff_dense,
        first_k_dense=cfg.first_k_dense, n_experts=cfg.n_experts,
        top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert, n_group=cfg.n_group,
        topk_group=cfg.topk_group, routed_scaling=cfg.routed_scaling,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk, indexer_types=tuple(cfg.indexer_types),
        vocab_size=cfg.vocab_size, rope_base=cfg.rope_base,
    )


def restore_fp32_islands(model: Glm52) -> None:
    """``Module.to(dtype=bf16)`` downcasts every float tensor; the engine's
    dtype policy keeps the leaders' indexer per-head weights ``w_idx_w`` and
    the noaux balance bias fp32 — restore them after the blanket cast."""
    for module in model.modules():
        if hasattr(module, "w_idx_w"):
            module.w_idx_w.data = module.w_idx_w.data.float()
        if hasattr(module, "w_router_bias"):
            module.w_router_bias.data = module.w_router_bias.data.float()


def build_glm52_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Glm52:
    model = Glm52(reference_glm52_config(cfg)).to(device=device, dtype=dtype)
    restore_fp32_islands(model)
    return model


def to_glm52_state_dict(cfg, get_bytes) -> dict:
    """Reference GLM-5.2 state_dict from the engine's packed weight objects
    (kind-dispatched: gdl / gml leaders with idx fields, gmf followers
    without)."""
    from dataflow.tasks.layouts import embed_weight_layout, head_weight_layout
    from dataflow.training.models.glm52 import _weight_layout_for, dims_of_glm52

    dims = dims_of_glm52(cfg)
    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        kind = dims.kind_of(i)                       # gdl | gml | gmf
        w = _weight_layout_for(dims, kind).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"{p}.attn.w_q_a.weight"] = transposed(w["w_q_a"])
        sd[f"{p}.attn.q_a_norm.weight"] = w["q_a_norm_w"].clone()
        sd[f"{p}.attn.w_q_b.weight"] = transposed(w["w_q_b"])
        sd[f"{p}.attn.w_kv_a.weight"] = transposed(w["w_kv_a"])
        sd[f"{p}.attn.kv_a_norm.weight"] = w["kv_a_norm_w"].clone()
        sd[f"{p}.attn.w_kv_b.weight"] = transposed(w["w_kv_b"])
        sd[f"{p}.attn.wo.weight"] = transposed(w["wo"])
        if kind in ("gdl", "gml"):                   # leader: the indexer
            sd[f"{p}.attn.w_idx_q.weight"] = transposed(w["w_idx_q"])
            sd[f"{p}.attn.w_idx_k.weight"] = transposed(w["w_idx_k"])
            sd[f"{p}.attn.idx_k_ln_w"] = w["idx_k_ln_w"].clone()
            sd[f"{p}.attn.idx_k_ln_b"] = w["idx_k_ln_b"].clone()
            sd[f"{p}.attn.w_idx_w"] = w["w_idx_w"].clone()   # (d, H_I) fp32
        sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        if kind == "gdl":
            sd[f"{p}.ffn.w1.weight"] = transposed(w["w1"])
            sd[f"{p}.ffn.w3.weight"] = transposed(w["w3"])
            sd[f"{p}.ffn.w2.weight"] = transposed(w["w2"])
        else:
            sd[f"{p}.ffn.w_router.weight"] = transposed(w["w_router"])
            sd[f"{p}.ffn.w_router_bias"] = w["w_router_bias"].clone()  # buffer
            sd[f"{p}.ffn.w13_experts"] = w["w13_experts"].clone()
            sd[f"{p}.ffn.w2_experts"] = w["w2_experts"].clone()
            sd[f"{p}.ffn.w_s13.weight"] = transposed(w["w_s13"])
            sd[f"{p}.ffn.w_s2.weight"] = transposed(w["w_s2"])
    return sd


def load_glm52_init(model, cfg, get_bytes):
    return load_state_dict_strict(model, to_glm52_state_dict(cfg, get_bytes))


# -- uniform dispatch pair (dataflow.pretrain.bridges) -------------------------

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16) -> Glm52:
    return build_glm52_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    return load_glm52_init(model, cfg, get_bytes)


def to_reference_state_dict(cfg, get_bytes) -> dict:
    """Uniform-name alias for the generic gate runners."""
    return to_glm52_state_dict(cfg, get_bytes)
