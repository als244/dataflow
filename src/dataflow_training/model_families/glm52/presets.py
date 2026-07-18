"""glm52 presets: the IndexShare engine-parity smoke twin at the locked
training configuration (``dataflow_training.run.presets`` owns the locked
constants)."""
from __future__ import annotations

from dataflow_training.model_families.dsv32.presets import dsv32_cfg_dict

from .model import ShapedGlm52Config


# IndexShare engine-parity smoke twin. LBL-OFF + frozen indexer like the
# dsv32 smoke (aux_coef=0, bias_update_speed=0, train_indexer=False);
# the 6-layer indexer_types pattern exercises full AND shared selection.
def glm52_smoke_preset() -> ShapedGlm52Config:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedGlm52Config(
        n_layers=6, d_model=256, n_heads=4, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_dim=32, qk_rope_dim=16, v_head_dim=32, d_ff_dense=512,
        first_k_dense=1, n_experts=8, top_k=2, d_ff_expert=128,
        n_group=4, topk_group=2, d_ff_shared=128,
        index_n_heads=4, index_head_dim=32, index_topk=64,
        indexer_types=("full", "full", "shared", "shared", "full", "shared"),
        aux_coef=0.0, bias_update_speed=0.0, train_indexer=False,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def glm52_cfg_dict(cfg) -> dict:
    d = dsv32_cfg_dict(cfg)
    d["indexer_types"] = list(cfg.indexer_types)
    return d
