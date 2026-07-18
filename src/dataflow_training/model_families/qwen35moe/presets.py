"""qwen35moe presets: the hybrid-MoE engine-parity smoke twin at the
locked training configuration (``dataflow_training.run.presets`` owns the
locked constants)."""
from __future__ import annotations

from .model import ShapedQwen35MoeConfig


# Hybrid-MoE engine-parity smoke twin (small dims, real 50304 vocab).
# Runs LBL-OFF (aux_coef=0) like the other MoE smokes — the reference
# trains pure CE, so the engine must too.
def qwen35moe_smoke_preset() -> ShapedQwen35MoeConfig:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedQwen35MoeConfig(
        n_layers=4, d_model=256, full_attention_interval=4, n_heads=4,
        n_kv_heads=2, head_dim=64, partial_rotary_factor=0.25,
        lin_k_heads=2, lin_v_heads=4, lin_k_head_dim=32, lin_v_head_dim=32,
        lin_conv_kernel=4, n_experts=8, top_k=2, d_ff_expert=256,
        n_shared_experts=1, d_ff_shared=256, aux_coef=0.0,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def qwen35moe_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel,
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        routing_mode=cfg.routing_mode, aux_coef=cfg.aux_coef,
        rope_base=cfg.rope_base, tied_embeddings=cfg.tied_embeddings,
        vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )
