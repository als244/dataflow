"""Qwen3 lowering: family declarations over the generic machinery.

Same contract as every family (docs/extending.md §4): a config->dims
mapping plus a ``FamilyLayouts`` declaration (qk-norm weights live in the
qwen3 block layout); the chain, sizes, and initial values are generic.
"""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Qwen3Dims,
    embed_weight_layout,
    head_weight_layout,
    qwen3_context_layout,
    qwen3_weight_layout,
)
from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_program import ShapedHardware
from .shaped_qwen3 import ShapedQwen3Config, build_shaped_qwen3


def dims_of_qwen3(cfg: ShapedQwen3Config) -> Qwen3Dims:
    return Qwen3Dims(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens,
        seq_len=cfg.seq_len,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


def family_layouts(cfg: ShapedQwen3Config) -> tuple[Qwen3Dims, FamilyLayouts]:
    dims = dims_of_qwen3(cfg)
    cl = qwen3_context_layout(dims)
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: qwen3_weight_layout(dims, layer=i),
        block_context_at=lambda i: cl,
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
    )


def lower_qwen3(
    cfg: ShapedQwen3Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_qwen3(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "qwen3-exact-v1", size_of=size_of_factory(dims, fl))


def initial_values_qwen3(program: Program, cfg: ShapedQwen3Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
