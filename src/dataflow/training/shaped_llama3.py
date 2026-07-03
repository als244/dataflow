"""Llama3 family: shaped config + kind spec for the generic builder.

The chain grammar, costs plumbing, and Program assembly are family-generic
and live in ``shaped_program.build_shaped_program`` — this module only
declares WHAT llama3 is: its config (shapes + param counts) and its single
dense-transformer layer kind (``roofline_block_kind_spec``). Every family
follows this same contract (see shaped_qwen3 / shaped_qwen35); llama3 is
not special.

Naming conventions of the emitted chain (family-invariant, owned by the
generic builder):

    tokens_{s}_{r}, targets_{s}_{r}      int32 token/target ids (initial, backing)
    W_embed, W_{i}, W_head               parameters (initial, backing)
    O_embed, O_{i}, O_head               AdamW state, 2x params (initial, backing)
    y_embed_{s}_{r}, y_{s}_{r}_{i}       block outputs (activations)
    A_{s}_{r}_{i}                        saved backward context per block
    logits_{s}_{r}, dlogits_{s}_{r}      head output / loss gradient
    dy_*                                 activation gradients
    dW_embed_{s}, dW_{s}_{i}, dW_head_{s}  parameter gradients (accumulated over rounds)
    loss_{s}_{r}                         scalar loss (final: backing)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import DTypePolicy

from .shaped_program import (
    LayerKindSpec,
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)


@dataclass(frozen=True)
class ShapedLlamaConfig:
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    vocab_size: int = 128_256
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    # "interleaved" (default): each optimizer task is emitted immediately
    # after the LAST mutation of its gradient (inside the final grad-accum
    # round's backward), so its state streaming (O in, W+O writebacks out)
    # overlaps the remaining backward compute. "tail" restores the legacy
    # all-optimizers-after-all-rounds order, whose transfers drain into a
    # GPU-idle PCIe phase at the end of every step (measured 1.5-2.0 s at
    # 8B/seq-1K). Task ids are identical in both modes; only order changes.
    optimizer_placement: str = "interleaved"
    # per-field dtype policy for params/grads/opt state (default: all bf16,
    # the historical convention; docs/notes/dtype-policy-design.md)
    dtypes: DTypePolicy = DTypePolicy()
    # ragged packing: explicit per-round sequence lengths (sum = tokens
    # per round); None = uniform batch x seq_len
    seq_lens: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # -- parameter counts ----------------------------------------------------
    @property
    def block_params(self) -> int:
        d, kv, ff = self.d_model, self.kv_dim, self.d_ff
        attn = d * d + d * kv + d * kv + d * d  # Wq, Wk, Wv, Wo
        mlp = 3 * d * ff                        # W1 (gate), W3 (up), W2 (down)
        norms = 2 * d
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    # -- activation widths (elements per token) --------------------------------
    @property
    def saved_ctx_width(self) -> int:
        """Elements per token saved by a block forward for its backward.

        Roughly: block input (d), attention normed input (d), q (d),
        k/v (2*kv), attention output (d), softmax lse (heads), mlp normed
        input (d), gate/up projections (2*ff). Matches the scale of real
        save-all footprints for this architecture.
        """
        d, kv, ff, h = self.d_model, self.kv_dim, self.d_ff, self.n_heads
        return 5 * d + 2 * kv + h + 2 * ff

    @classmethod
    def tiny(cls) -> "ShapedLlamaConfig":
        return cls(
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=160,
            vocab_size=512,
            seq_len=64,
            batch=1,
        )

    @classmethod
    def llama3_8b(cls, *, seq_len: int = 4096, batch: int = 1, grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedLlamaConfig":
        return cls(seq_len=seq_len, batch=batch, grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def build_shaped_llama3(
    cfg: ShapedLlamaConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
    kinds: Mapping[str, LayerKindSpec] | None = None,
    kind_of=None,
) -> Program:
    """Llama3 through the generic builder: one dense-transformer kind.

    ``kinds``/``kind_of`` remain overridable because this function doubles
    as the heterogeneous-family test harness entry (the LayerKindSpec
    machinery is exercised against llama3 shapes in
    tests/training/test_shaped_llama3.py)."""
    hw = hw or ShapedHardware()
    if kinds is None:
        kinds = {"block": roofline_block_kind_spec(cfg, hw)}
        kind_of = None
    return build_shaped_program(
        cfg, hw=hw, kinds=kinds, kind_of=kind_of, family="llama3-shaped",
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
    )
