"""Qwen3-dense shaped program generator.

The chain STRUCTURE (task/object naming, grad-accum mutation pattern,
recompute rewrites, interleaved optimizer placement, replayable
final_locations) is family-generic and lives in ``build_shaped_llama3`` —
this module provides a Qwen3 config exposing the same size/cost surface
(duck-typed) and relabels the result. Qwen3-vs-llama differences that reach
this layer: qk-norm weights in the block parameter count, head_dim decoupled
from d_model/n_heads (q projection is n_heads*head_dim wide), rope theta
1e6, larger vocab, untied embed/head (already separate objects here).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .shaped_llama3 import ShapedHardware, build_shaped_llama3


@dataclass(frozen=True)
class ShapedQwen3Config:
    n_layers: int = 36
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 12288
    vocab_size: int = 151_936
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer_placement: str = "interleaved"

    @property
    def tokens(self) -> int:
        return self.seq_len * self.batch

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # -- parameter counts (duck-typed for the shared chain builder) -----------
    @property
    def block_params(self) -> int:
        d, q, kv, ff = self.d_model, self.q_dim, self.kv_dim, self.d_ff
        attn = d * q + 2 * d * kv + q * d
        mlp = 3 * d * ff
        norms = 2 * d + 2 * self.head_dim  # block norms + q/k norms
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def saved_ctx_width(self) -> int:
        """Rough elements/token saved per block (roofline seed only — the
        lowering replaces every size with the packed layout's exact bytes):
        qm/attn_out (2q), km/v (2kv), h_mid (d), x1/x3 (2ff), plus the fp32
        rstds and lse counted as 2 bf16-equivalent elements each."""
        d, q, kv, ff = self.d_model, self.q_dim, self.kv_dim, self.d_ff
        h, kvh = self.n_heads, self.n_kv_heads
        return 2 * q + 2 * kv + d + 2 * ff + 2 * (2 + h + kvh) + 2 * h

    @classmethod
    def tiny(cls) -> "ShapedQwen3Config":
        return cls(
            n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
            d_ff=512, vocab_size=512, seq_len=64, batch=1,
        )

    @classmethod
    def qwen3_8b(cls, *, seq_len: int = 4096, batch: int = 1,
                 grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedQwen3Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def build_shaped_qwen3(
    cfg: ShapedQwen3Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    label = name or (
        f"qwen3-shaped-{cfg.n_layers}L-d{cfg.d_model}-s{cfg.seq_len}-b{cfg.batch}"
        f"-r{cfg.grad_accum_rounds}-steps{cfg.num_steps}"
    )
    program = build_shaped_llama3(
        cfg, hw=hw, fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=label,
    )
    return replace(program, metadata={**program.metadata, "family": "qwen3-shaped"})
