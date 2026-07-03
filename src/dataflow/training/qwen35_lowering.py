"""Qwen3.5 lowering: family declarations over the generic machinery.

Same contract as every family (docs/extending.md §4), with the two
qwen3.5 particulars declared, not special-cased: per-KIND block layouts
(DeltaNet vs gated-attention, chosen by ``dims.kind_of``) and the
embedding mode — the 9B is UNTIED (bare-table W_embed + [table |
final_norm_w] W_head); the 2B-style tied variant packs the head layout
into the single W_embed (policy-addressed as head.*).
"""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    embed_weight_layout,
    head_weight_layout,
    qwen35_attn_context_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_context_layout,
    qwen35_lin_weight_layout,
)
from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_program import ShapedHardware
from .shaped_qwen35 import ShapedQwen35Config, build_shaped_qwen35, dims_of_qwen35

_WEIGHT_BUILDERS = {
    "lin": qwen35_lin_weight_layout,
    "full": qwen35_attn_weight_layout,
}


def _a_log_init(n, gen):
    # decay magnitudes ~ U(1, 16) in log space (GDN convention)
    import torch

    return torch.empty(n).uniform_(1.0, 16.0, generator=gen).log()


def _dt_bias_init(n, gen):
    import torch

    return torch.zeros(n)


def family_layouts(cfg: ShapedQwen35Config):
    dims = dims_of_qwen35(cfg)
    ctx = {
        "lin": qwen35_lin_context_layout(dims),
        "full": qwen35_attn_context_layout(dims),
    }
    hl = head_weight_layout(dims)
    tied = cfg.tied_embeddings
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _WEIGHT_BUILDERS[dims.kind_of(i)](dims, layer=i),
        block_context_at=lambda i: ctx[dims.kind_of(i)],
        embed=hl if tied else embed_weight_layout(dims),
        head=hl,
        embed_ns="head" if tied else "embed",
        init_specials={"A_log": _a_log_init, "dt_bias": _dt_bias_init},
    )


def lower_qwen35(
    cfg: ShapedQwen35Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_qwen35(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "qwen35-exact-v1", size_of=size_of_factory(dims, fl))


def initial_values_qwen35(program: Program, cfg: ShapedQwen35Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
