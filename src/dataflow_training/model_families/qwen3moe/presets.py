"""qwen3moe presets: the MoE engine-parity smoke twin at the locked
training configuration (``dataflow_training.run.presets`` owns the locked
constants)."""
from __future__ import annotations

from .model import ShapedQwen3MoeConfig


# MoE engine-parity smoke twin (small dims, real 50304 vocab). Runs
# LBL-OFF (aux_coef=0) like the other MoE smokes — the reference trains
# pure CE, so the engine must too.
def qwen3moe_smoke_preset() -> ShapedQwen3MoeConfig:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedQwen3MoeConfig(
        n_layers=4, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        n_experts=8, top_k=2, d_ff_expert=256, aux_coef=0.0,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def qwen3moe_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode=cfg.routing_mode, aux_coef=cfg.aux_coef,
        rope_base=cfg.rope_base, vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )
