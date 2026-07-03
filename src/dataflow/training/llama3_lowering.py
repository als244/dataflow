"""Llama3 lowering: family declarations over the generic machinery.

The chain comes from ``shaped_program`` (via the family's shaped module),
sizes and initial values from ``training.lowering`` — this module only
declares WHAT llama3 is: the config->dims mapping and which packed layouts
back each weight object (docs/extending.md §4).
"""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    LlamaDims,
    context_layout,
    embed_weight_layout,
    head_weight_layout,
    weight_layout,
)
from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_llama3 import ShapedHardware, ShapedLlamaConfig, build_shaped_llama3


def dims_of(cfg: ShapedLlamaConfig) -> LlamaDims:
    return LlamaDims(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens,
        seq_len=cfg.seq_len,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


def family_layouts(cfg: ShapedLlamaConfig) -> tuple[LlamaDims, FamilyLayouts]:
    dims = dims_of(cfg)
    cl = context_layout(dims)
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: weight_layout(dims, layer=i),
        block_context_at=lambda i: cl,
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
    )


def lower_llama3(
    cfg: ShapedLlamaConfig,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_llama3(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "llama3-exact-v1", size_of=size_of_factory(dims, fl))


def initial_values(program: Program, cfg: ShapedLlamaConfig, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
