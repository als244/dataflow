"""qwen35 (Qwen3.5-dense hybrid) presets: the ~300M study config and the
tiny smoke twin, at the locked training configuration
(``dataflow_training.run.presets`` owns the locked constants)."""
from __future__ import annotations

from .model import ShapedQwen35Config

# Qwen3.5-dense (hybrid) preset. The pytorch reference's delta-rule is a
# sequential loop over the sequence (+ per-step autograd state), so a short
# seq_len keeps the 1000-step comparison tractable; batch is cheap for the
# loop (parallel). ~300M hybrid: 9 Gated-DeltaNet + 3 gated-attention layers.
QWEN35_SEQ_LEN = 512
QWEN35_BATCH = 4              # 512 x 4 = 2048 tokens / round
QWEN35_GRAD_ACCUM_ROUNDS = 4  # 4 x 2048 = 8192 tokens / step


def qwen35_preset() -> ShapedQwen35Config:
    from dataflow_training.run.presets import VOCAB_SIZE

    return ShapedQwen35Config(
        n_layers=12, d_model=1024, full_attention_interval=4,
        n_heads=8, n_kv_heads=2, head_dim=128, partial_rotary_factor=0.25,
        lin_k_heads=4, lin_v_heads=8, lin_k_head_dim=128, lin_v_head_dim=128,
        lin_conv_kernel=4, d_ff=4096, vocab_size=VOCAB_SIZE,
        seq_len=QWEN35_SEQ_LEN, batch=QWEN35_BATCH,
        grad_accum_rounds=QWEN35_GRAD_ACCUM_ROUNDS,
    )


def qwen35_smoke_preset() -> ShapedQwen35Config:
    """Tiny qwen3.5-dense hybrid smoke: 3 delta-rule + 1 full-attention
    layer (interval 4), the same attention geometry as the qwen35moe
    smoke, dense SwiGLU tail."""
    from dataflow_training.run.presets import (
        SMOKE_BATCH,
        SMOKE_GRAD_ACCUM_ROUNDS,
        SMOKE_SEQ_LEN,
        VOCAB_SIZE,
    )

    return ShapedQwen35Config(
        n_layers=4, d_model=256, full_attention_interval=4, n_heads=4,
        n_kv_heads=2, head_dim=64, partial_rotary_factor=0.25,
        lin_k_heads=2, lin_v_heads=4, lin_k_head_dim=32, lin_v_head_dim=32,
        lin_conv_kernel=4, d_ff=1024, vocab_size=VOCAB_SIZE,
        seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


def qwen35_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel, d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
        tied_embeddings=cfg.tied_embeddings, rope_base=cfg.rope_base,
    )
