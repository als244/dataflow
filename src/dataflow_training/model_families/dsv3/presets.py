"""dsv3 presets: the ~1.9B MoE study config and the MLA engine-parity
smoke twin, at the locked training configuration
(``dataflow_training.run.presets`` owns the locked constants)."""
from __future__ import annotations

from .model import ShapedDsv3Config


def dsv3_2b_preset(load_balance: bool = True) -> ShapedDsv3Config:
    """~1.9B-total / ~0.5B-active dsv3 (MLA + grouped noaux MoE): the
    MoE entry of the 1000-step pretraining study, same token budget as
    the llama3/qwen35 runs (seq 2048 x batch 4 x ga 8 = 64K/step).
    ``load_balance=True`` = paper-like (noaux router bias 1e-3 + small
    seq-wise aux 1e-4); False = balancing fully off."""
    from dataflow_training.run.presets import VOCAB_SIZE

    balance = (dict(aux_coef=1e-4, bias_update_speed=1e-3)
               if load_balance else
               dict(aux_coef=0.0, bias_update_speed=0.0))
    return ShapedDsv3Config(
        n_layers=14, d_model=1280, n_heads=20,
        q_lora_rank=640, kv_lora_rank=320,
        qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64,
        d_ff_dense=5120, first_k_dense=2,
        n_experts=40, top_k=4, d_ff_expert=832,
        n_group=8, topk_group=4, d_ff_shared=2560,
        vocab_size=VOCAB_SIZE, seq_len=2048, batch=4,
        grad_accum_rounds=8, **balance)


# MLA engine-parity smoke twin. LBL-OFF like the other MoE smokes
# (aux_coef=0 AND bias_update_speed=0 — no load-balance functions on
# either side).
def dsv3_smoke_preset() -> ShapedDsv3Config:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedDsv3Config(
        n_layers=4, d_model=256, n_heads=4, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_dim=32, qk_rope_dim=16, v_head_dim=32, d_ff_dense=512,
        first_k_dense=1, n_experts=8, top_k=2, d_ff_expert=128,
        n_group=4, topk_group=2, d_ff_shared=128,
        aux_coef=0.0, bias_update_speed=0.0,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def dsv3_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim, d_ff_dense=cfg.d_ff_dense,
        first_k_dense=cfg.first_k_dense, n_experts=cfg.n_experts,
        top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert, n_group=cfg.n_group,
        topk_group=cfg.topk_group, routed_scaling=cfg.routed_scaling,
        bias_update_speed=cfg.bias_update_speed, aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        rope_base=cfg.rope_base,
        vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )
