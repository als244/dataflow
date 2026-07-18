"""qwen3 presets: the engine-parity smoke twin at the locked training
configuration (``dataflow_training.run.presets`` owns the locked
constants)."""
from __future__ import annotations

from dataflow_training.model_families.llama3.presets import SMOKE

from .model import ShapedQwen3Config


# A qwen3-dense engine-parity smoke twin of llama3's SMOKE: small dims,
# the real 50304 vocab, decoupled-head_dim family surface (per-head
# qk-norm).
def qwen3_smoke_preset() -> ShapedQwen3Config:
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedQwen3Config(
        n_layers=SMOKE["n_layers"], d_model=SMOKE["d_model"],
        n_heads=SMOKE["n_heads"], n_kv_heads=SMOKE["n_kv_heads"],
        head_dim=SMOKE["d_model"] // SMOKE["n_heads"], d_ff=SMOKE["d_ff"],
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def qwen3_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )
