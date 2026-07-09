"""Llama3-shaped model presets for the pretraining study + the locked
training configuration.

The training config is fixed (Shein 2026-07-09): sequences of length 2048
(uniform, no ragged packing), 4 per round → 8192 tokens/round, 8 grad-accum
rounds → 65,536 (~64K) tokens/step, gpt2 vocab padded to 50304. The scaling
ladder reuses this exact config; only the model shape changes.

The ladder is llama3-shaped with head_dim 64 throughout, GQA at 4:1, and
d_ff = 4·d_model. ``rope_base`` is the ``LlamaDims`` default (500000). The
LM head is untied (a separate ``W_head``).
"""
from __future__ import annotations

from dataflow.training.models.llama3 import ShapedLlamaConfig

# -- locked training config ---------------------------------------------------
SEQ_LEN = 2048
BATCH = 4                    # 4 × 2048 = 8192 tokens / round
GRAD_ACCUM_ROUNDS = 8        # 8 × 8192 = 65,536 tokens / step
VOCAB_SIZE = 50304
# Driver loop count = number of run() invocations. The PROGRAM stays
# single-step (num_steps=1, the ShapedLlamaConfig default): the daemon holds
# W/O in the store and evolves them in place across run() calls. Unrolling
# the steps into the program instead builds a ~300k-task monster.
TRAIN_STEPS = 1000

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
SMOKE_SEQ_LEN = 256
SMOKE_BATCH = 2              # 512 tokens / round
SMOKE_GRAD_ACCUM_ROUNDS = 2  # 1024 tokens / step
SMOKE_STEPS = 100           # driver loop count for the parity smoke gate


def preset(name: str) -> ShapedLlamaConfig:
    """A ladder ``ShapedLlamaConfig`` at the locked training config.

    The program is SINGLE-STEP (``num_steps`` defaults to 1); the driver
    calls ``run()`` ``TRAIN_STEPS`` times with W/O persisting in the store."""
    p = LADDER[name]
    return ShapedLlamaConfig(
        n_layers=p["n_layers"], d_model=p["d_model"], n_heads=p["n_heads"],
        n_kv_heads=p["n_kv_heads"], d_ff=p["d_ff"], vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN, batch=BATCH, grad_accum_rounds=GRAD_ACCUM_ROUNDS,
    )


def resolve_preset(name: str):
    """Preset name -> config, across families ('qwen35' -> the hybrid preset,
    otherwise the llama3 ladder)."""
    if name in ("qwen35", "q35"):
        return qwen35_preset()
    return preset(name)


def smoke_preset() -> ShapedLlamaConfig:
    """The tiny real-vocab smoke config (single-step; fast full-pipeline
    reference-vs-service parity check)."""
    return ShapedLlamaConfig(
        n_layers=SMOKE["n_layers"], d_model=SMOKE["d_model"],
        n_heads=SMOKE["n_heads"], n_kv_heads=SMOKE["n_kv_heads"],
        d_ff=SMOKE["d_ff"], vocab_size=VOCAB_SIZE,
        seq_len=SMOKE_SEQ_LEN, batch=SMOKE_BATCH,
        grad_accum_rounds=SMOKE_GRAD_ACCUM_ROUNDS,
    )


# Qwen3.5-dense (hybrid) preset. The pytorch reference's delta-rule is a
# sequential loop over the sequence (+ per-step autograd state), so a short
# seq_len keeps the 1000-step comparison tractable; batch is cheap for the
# loop (parallel). ~300M hybrid: 9 Gated-DeltaNet + 3 gated-attention layers.
QWEN35_SEQ_LEN = 512
QWEN35_BATCH = 4              # 512 x 4 = 2048 tokens / round
QWEN35_GRAD_ACCUM_ROUNDS = 4  # 4 x 2048 = 8192 tokens / step


def qwen35_preset() -> "object":
    from dataflow.training.models.qwen35 import ShapedQwen35Config

    return ShapedQwen35Config(
        n_layers=12, d_model=1024, full_attention_interval=4,
        n_heads=8, n_kv_heads=2, head_dim=128, partial_rotary_factor=0.25,
        lin_k_heads=4, lin_v_heads=8, lin_k_head_dim=128, lin_v_head_dim=128,
        lin_conv_kernel=4, d_ff=4096, vocab_size=VOCAB_SIZE,
        seq_len=QWEN35_SEQ_LEN, batch=QWEN35_BATCH,
        grad_accum_rounds=QWEN35_GRAD_ACCUM_ROUNDS,
    )


def resolver_family(cfg) -> str:
    return {"ShapedLlamaConfig": "llama3",
            "ShapedQwen35Config": "qwen35"}[type(cfg).__name__]


def _llama3_cfg_dict(cfg) -> dict:
    return dict(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, d_ff=cfg.d_ff, vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len, batch=cfg.batch,
        grad_accum_rounds=cfg.grad_accum_rounds, num_steps=cfg.num_steps,
    )


def _qwen35_cfg_dict(cfg) -> dict:
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


def cfg_dict(cfg) -> dict:
    """JSON-able config for the wire resolver spec. The daemon rebuilds
    ``config_type(**cfg)`` (bridge.resolver_for), so every field here must be
    a constructor kwarg; omitted fields take their defaults (all-bf16
    ``dtypes``, ``opt_policy='adamw'``, ``seq_lens=None`` uniform) — the SAME
    defaults the planned program used, so the rebuilt dims match."""
    if type(cfg).__name__ == "ShapedQwen35Config":
        return _qwen35_cfg_dict(cfg)
    return _llama3_cfg_dict(cfg)


def param_counts(cfg: ShapedLlamaConfig) -> dict:
    """(embed, head, blocks, total, non_embedding) parameter counts."""
    embed = cfg.embed_params
    head = cfg.head_params
    blocks = cfg.n_layers * cfg.block_params
    total = embed + head + blocks
    return dict(embed=embed, head=head, blocks=blocks, total=total,
                non_embedding=blocks)


def tokens_per_step(cfg: ShapedLlamaConfig) -> int:
    return cfg.tokens * cfg.grad_accum_rounds
