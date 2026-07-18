"""dsv32 presets: the DSA engine-parity smoke twin at the locked training
configuration (``dataflow_training.run.presets`` owns the locked
constants)."""
from __future__ import annotations

from dataflow_training.model_families.dsv3.presets import dsv3_cfg_dict

from .model import ShapedDsv32Config


# DSA engine-parity smoke twin. LBL-OFF like the other MoE smokes
# (aux_coef=0 AND bias_update_speed=0); additionally freezes the indexer
# (train_indexer=False): the reference has no KL objective, and CE
# reaches the indexer on NEITHER side (selection is detached), so both
# keep the indexer at its init while it still drives the sparse
# selection.
def dsv32_smoke_preset() -> ShapedDsv32Config:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedDsv32Config(
        n_layers=4, d_model=256, n_heads=4, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_dim=32, qk_rope_dim=16, v_head_dim=32, d_ff_dense=512,
        first_k_dense=1, n_experts=8, top_k=2, d_ff_expert=128,
        n_group=4, topk_group=2, d_ff_shared=128,
        index_n_heads=4, index_head_dim=32, index_topk=64,
        aux_coef=0.0, bias_update_speed=0.0, train_indexer=False,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def dsv32_cfg_dict(cfg) -> dict:
    d = dsv3_cfg_dict(cfg)
    d.update(
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk, sparse_mode=cfg.sparse_mode,
        train_indexer=cfg.train_indexer,
    )
    return d
