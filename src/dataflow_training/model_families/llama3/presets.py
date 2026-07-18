"""Llama3 presets for the pretraining study: the scaling ladder and the
tiny real-vocab smoke config, at the locked training configuration
(``dataflow_training.run.presets`` owns the locked constants).

The ladder is llama3-shaped with head_dim 64 throughout, GQA at 4:1, and
d_ff = 4·d_model. ``rope_base`` is the ``LlamaDims`` default (500000). The
LM head is untied (a separate ``W_head``).
"""
from __future__ import annotations

from .model import ShapedLlamaConfig

# -- the scaling ladder (name -> shape) ---------------------------------------
# non-embedding params ≈ n_layers · (attn 4d² GQA-reduced + mlp 3·d·d_ff)
LADDER: dict[str, dict] = {
    "l3_125m": dict(d_model=768,  n_layers=12, n_heads=12, n_kv_heads=3, d_ff=3072),
    "l3_350m": dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=4096),
    "l3_760m": dict(d_model=1536, n_layers=24, n_heads=24, n_kv_heads=6, d_ff=6144),
    "l3_1b":   dict(d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8, d_ff=8192),
}
LADDER_NAMES = list(LADDER)

# A tiny real-vocab model for the infra/parity SMOKE test — small dims but
# vocab 50304 so it consumes real fineweb tokens through the whole pipeline
# (reference + service daemon). head_dim = 256/4 = 64.
SMOKE = dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=1024)


def preset(name: str) -> ShapedLlamaConfig:
    """A ladder ``ShapedLlamaConfig`` at the locked training config.

    The program is SINGLE-STEP (``num_steps`` defaults to 1); the driver
    calls ``run()`` ``TRAIN_STEPS`` times with W/O persisting in the store."""
    from dataflow_training.run.presets import (
        BATCH,
        GRAD_ACCUM_ROUNDS,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    p = LADDER[name]
    return ShapedLlamaConfig(
        n_layers=p["n_layers"], d_model=p["d_model"], n_heads=p["n_heads"],
        n_kv_heads=p["n_kv_heads"], d_ff=p["d_ff"], vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN, batch=BATCH, grad_accum_rounds=GRAD_ACCUM_ROUNDS,
    )


def smoke_preset() -> ShapedLlamaConfig:
    """The tiny real-vocab smoke config (single-step; fast full-pipeline
    reference-vs-service parity check)."""
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedLlamaConfig(
        n_layers=SMOKE["n_layers"], d_model=SMOKE["d_model"],
        n_heads=SMOKE["n_heads"], n_kv_heads=SMOKE["n_kv_heads"],
        d_ff=SMOKE["d_ff"], vocab_size=VOCAB_SIZE,
        seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def llama3_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, d_ff=cfg.d_ff, vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )
