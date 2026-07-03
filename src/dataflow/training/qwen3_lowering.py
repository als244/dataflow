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


def _size_of_factory(cfg: ShapedQwen3Config):
    """Per-object exact sizes; per-layer block layouts (depth-dependent
    dtype policies) — see llama3_lowering._size_of_factory."""
    dims = dims_of_qwen3(cfg)
    p = dims.dtypes
    el = embed_weight_layout(dims)
    hl = head_weight_layout(dims)
    cl = qwen3_context_layout(dims)
    wl_i = [qwen3_weight_layout(dims, layer=i) for i in range(cfg.n_layers)]
    dw_i = [grad_layout(wl_i[i], p, layer=i).total_bytes for i in range(cfg.n_layers)]
    o_i = [opt_state_layout(wl_i[i], p, layer=i).total_bytes for i in range(cfg.n_layers)]
    dw_e = grad_layout(el, p, ns="embed").total_bytes
    dw_h = grad_layout(hl, p, ns="head").total_bytes
    o_e = opt_state_layout(el, p, ns="embed").total_bytes
    o_h = opt_state_layout(hl, p, ns="head").total_bytes

    def size_of(oid: str) -> int | None:
        if oid.startswith("A_"):
            return cl.total_bytes
        if oid.startswith("dW_embed"):
            return dw_e
        if oid == "W_embed":
            return el.total_bytes
        if oid.startswith("dW_head"):
            return dw_h
        if oid == "W_head":
            return hl.total_bytes
        if oid == "O_embed":
            return o_e
        if oid == "O_head":
            return o_h
        if oid.startswith("O_"):
            return o_i[int(oid.split("_")[1])]
        if oid.startswith("dW_"):
            return dw_i[int(oid.rsplit("_", 1)[1])]
        if oid.startswith("W_"):
            return wl_i[int(oid.split("_")[1])].total_bytes
        return None

    return size_of


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
    return apply_exact_sizes(shaped, {}, "qwen3-exact-v1", size_of=_size_of_factory(cfg))


def initial_values_qwen3(program: Program, cfg: ShapedQwen3Config, backend, *, seed: int = 0):
    dims = dims_of_qwen3(cfg)
    return fill_initial_values(
        program, dims, qwen3_weight_layout(dims), backend, seed=seed,
        head_layout=head_weight_layout(dims),
        block_layout_of=lambda i: qwen3_weight_layout(dims, layer=i),
    )
