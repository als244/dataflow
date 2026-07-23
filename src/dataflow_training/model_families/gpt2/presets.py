"""GPT-2 presets: the 124M speedrun-baseline shape and the tiny real-vocab
smoke config, at the locked training configuration.

``gpt2_124m`` is the llm.c/nanoGPT GPT-2 124M: 12L x 768d x 12H, d_ff
3072, vocab 50304 (50257 padded), context 1024, untied. At seq_len 1024 x
batch 8 a round is 8192 tokens (the repo's locked round size); the
speedrun-equivalent 0.5M-token step is ``--ga-rounds 64``.
"""
from __future__ import annotations

from .model import ShapedGpt2Config

GPT2_SEQ_LEN = 1024      # GPT-2 context window (wpe rows)
GPT2_BATCH = 8           # 8 x 1024 = 8192 tokens / round


def gpt2_preset() -> ShapedGpt2Config:
    """GPT-2 124M at the locked training config (single-step program; the
    driver calls ``run()`` per step with W/O persisting in the store)."""
    from dataflow_training.run.presets import GRAD_ACCUM_ROUNDS, VOCAB_SIZE

    return ShapedGpt2Config(
        n_layers=12, d_model=768, n_heads=12, d_ff=3072,
        vocab_size=VOCAB_SIZE, seq_len=GPT2_SEQ_LEN, batch=GPT2_BATCH,
        max_seq_len=GPT2_SEQ_LEN, grad_accum_rounds=GRAD_ACCUM_ROUNDS,
    )


def gpt2_smoke_preset() -> ShapedGpt2Config:
    """Tiny real-vocab smoke config (single-step; fast full-pipeline
    reference-vs-service parity check). head_dim = 256/4 = 64."""
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedGpt2Config(
        n_layers=4, d_model=256, n_heads=4, d_ff=1024,
        vocab_size=VOCAB_SIZE, seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def gpt2_cfg_dict(cfg) -> dict:
    d = dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_ff=cfg.d_ff, vocab_size=cfg.vocab_size, seq_len=cfg.seq_len,
        batch=cfg.batch, grad_accum_rounds=cfg.grad_accum_rounds,
        num_steps=cfg.num_steps, tied_embeddings=cfg.tied_embeddings,
        use_bias=cfg.use_bias,
    )
    d["max_seq_len"] = cfg.max_seq_len
    return d
