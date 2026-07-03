"""Execution-grade Qwen3 lowering: shaped structure + layout-exact sizes.

Mirrors ``llama3_lowering`` through the shared helpers: the shaped chain is
family-generic; this module supplies the Qwen3 size map (packed layouts with
qk-norm weights) and the dims object executables consume.
"""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Qwen3Dims,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    qwen3_context_layout,
    qwen3_weight_layout,
)
from .llama3_lowering import apply_exact_sizes, fill_initial_values
from .shaped_llama3 import ShapedHardware
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


def _exact_sizes(cfg: ShapedQwen3Config) -> dict[str, int]:
    dims = dims_of_qwen3(cfg)
    p = dims.dtypes
    wl = qwen3_weight_layout(dims)
    el = embed_weight_layout(dims)
    hl = head_weight_layout(dims)
    cl = qwen3_context_layout(dims)
    # dW/O sized from their own field-mirrored layouts — see llama3_lowering
    return {
        "__W_block": wl.total_bytes,
        "__W_embed": el.total_bytes,
        "__W_head": hl.total_bytes,
        "__dW_block": grad_layout(wl, p).total_bytes,
        "__dW_embed": grad_layout(el, p, ns="embed").total_bytes,
        "__dW_head": grad_layout(hl, p, ns="head").total_bytes,
        "__A": cl.total_bytes,
        "__O_block": opt_state_layout(wl, p).total_bytes,
        "__O_embed": opt_state_layout(el, p, ns="embed").total_bytes,
        "__O_head": opt_state_layout(hl, p, ns="head").total_bytes,
    }


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
    return apply_exact_sizes(shaped, _exact_sizes(cfg), "qwen3-exact-v1")


def initial_values_qwen3(program: Program, cfg: ShapedQwen3Config, backend, *, seed: int = 0):
    dims = dims_of_qwen3(cfg)
    return fill_initial_values(
        program, dims, qwen3_weight_layout(dims), backend, seed=seed,
        head_layout=head_weight_layout(dims),
    )
