"""Execution-grade llama3 lowering: shaped structure + layout-exact sizes.

`build_shaped_llama3` already emits the right task chain (naming, grad-accum
mutation pattern, recompute variants, rewrites); this module rewrites its
object sizes to the tasks layer's packed-layout bytes (the single source of
truth executables actually address) and stamps optimizer tasks with their
step index. `initial_values()` fills pinned buffers with real weights and
data so a run computes real math.

Object-size mapping (per layout):
    W_i / dW_i           weight_layout(dims).total_bytes
    W_embed / W_head / dW_embed_* / dW_head_*   embed_weight_layout
    O_i                  adamw_state_layout(weight elems)
    O_embed / O_head     adamw_state_layout(vocab*d elems)
    A_{s}_{r}_{i}        context_layout(dims).total_bytes
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from dataflow.core import ObjectSpec, OutputSpec, Program, RecomputeOption, TaskSpec
from dataflow.tasks.layouts import (
    DTypePolicy,
    LlamaDims,
    context_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    weight_layout,
)
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
    )


def _exact_sizes(cfg: ShapedLlamaConfig) -> dict[str, int]:
    dims = dims_of(cfg)
    p = dims.dtypes
    wl = weight_layout(dims)
    el = embed_weight_layout(dims)
    hl = head_weight_layout(dims)
    cl = context_layout(dims)
    # dW and O are sized from their OWN layouts (field-mirrored at grad/opt
    # dtypes); under the default all-bf16 policy the totals coincide with
    # the historical W-sized dW and flat bf16-halves O.
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


def _mapped_size(object_id: str, sizes: dict[str, int]) -> int | None:
    if object_id.startswith("A_"):
        return sizes["__A"]
    if object_id.startswith("dW_embed"):
        return sizes["__dW_embed"]
    if object_id == "W_embed":
        return sizes["__W_embed"]
    if object_id.startswith("dW_head"):
        return sizes["__dW_head"]
    if object_id == "W_head":
        return sizes["__W_head"]  # head packs [table | final_norm_w]
    if object_id == "O_embed":
        return sizes["__O_embed"]
    if object_id == "O_head":
        return sizes["__O_head"]
    if object_id.startswith("O_"):
        return sizes["__O_block"]
    if object_id.startswith("dW_"):
        return sizes["__dW_block"]
    if object_id.startswith("W_"):
        return sizes["__W_block"]
    return None


def apply_exact_sizes(
    shaped: Program, sizes: dict[str, int], lowering_tag: str, *, size_of=None,
) -> Program:
    """Rewrite a shaped program's object sizes to packed-layout truth and
    stamp optimizer tasks with their step index — shared by every family's
    lowering. ``size_of(object_id) -> bytes | None`` overrides the default
    uniform map (heterogeneous families size W_{i}/A_*_{i} by layer kind)."""

    if size_of is None:
        size_of = lambda oid: _mapped_size(oid, sizes)  # noqa: E731

    def fix_obj(o: ObjectSpec) -> ObjectSpec:
        size = size_of(o.id)
        return o if size is None else replace(o, size_bytes=size, tensor=None)

    def fix_out(o: OutputSpec) -> OutputSpec:
        size = size_of(o.id)
        return o if size is None else replace(o, size_bytes=size, tensor=None)

    def fix_task(t: TaskSpec, step: int) -> TaskSpec:
        params = dict(t.block_params)
        if t.group == "optimizer":
            params["step"] = step
        return replace(t, outputs=tuple(fix_out(o) for o in t.outputs), block_params=params)

    def step_of(task_id: str) -> int:
        # optimizer ids: optimizer_embed_{s} / optimizer_{s}_{i} / optimizer_head_{s}
        parts = task_id.split("_")
        if task_id.startswith(("optimizer_embed", "optimizer_head")):
            return int(parts[-1])
        return int(parts[1])

    new_rewrites = tuple(
        replace(
            rw,
            options=(
                RecomputeOption(
                    level=0,
                    saved_bytes=size_of(rw.object_id) or sizes.get("__A", 0),
                    recompute_us=0.0, label="save",
                ),
            ) + tuple(o for o in rw.options if o.level != 0),
        )
        for rw in shaped.recompute_rewrites
    )
    return replace(
        shaped,
        initial_objects=tuple(fix_obj(o) for o in shaped.initial_objects),
        tasks=tuple(
            fix_task(t, step_of(t.id)) if t.group == "optimizer" else fix_task(t, 0)
            for t in shaped.tasks
        ),
        recompute_rewrites=new_rewrites,
        metadata={**shaped.metadata, "lowering": lowering_tag},
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
    return apply_exact_sizes(shaped, _exact_sizes(cfg), "llama3-exact-v1")


def initial_values(program: Program, cfg: ShapedLlamaConfig, backend, *, seed: int = 0):
    """Allocate + fill pinned buffers for every initial object.

    Weights: N(0, 0.02) bf16; norm weights: ones; optimizer state: zeros;
    tokens/targets: uniform ints. Returns {object_id: Buffer} for
    Engine.execute(initial_buffers=...).
    """
    from dataflow.tasks.layouts import weight_layout as _wl

    d = dims_of(cfg)
    return fill_initial_values(
        program, d, _wl(d), backend, seed=seed, head_layout=head_weight_layout(d),
    )


def fill_weight_fields(buf, layout, gen, *, special=None) -> None:
    """Per-FIELD init at each field's OWN storage dtype: N(0, 0.02) draws,
    ``*_norm_w`` fields ones, ``special[name](n, gen)`` overrides (A_log
    etc.). Padding gaps are explicitly zeroed (deterministic bytes for
    comparators). Draw order = field order — part of run-to-run
    reproducibility, not of golden parity (goldens init FROM these bytes)."""
    import torch

    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view

    end = 0
    u8 = torch_view(buf, (buf.size_bytes,), torch.uint8)
    for f in layout.fields:
        if f.offset_bytes > end:  # zero the alignment gap before this field
            u8[end : f.offset_bytes] = 0
        end = f.offset_bytes + f.nbytes
        n = 1
        for s in f.shape:
            n *= int(s)
        v = torch_view(buf, (n,), TORCH_DTYPE_BY_NAME[f.dtype], offset_bytes=f.offset_bytes)
        if f.name.endswith("_norm_w"):
            v.fill_(1.0)
        elif special is not None and f.name in special:
            v.copy_(special[f.name](n, gen).to(v.dtype))
        else:
            v.copy_((torch.randn(n, generator=gen) * 0.02).to(v.dtype))
    if buf.size_bytes > end:
        u8[end:] = 0


def fill_initial_values(program: Program, dims, wl, backend, *, seed: int = 0,
                        head_layout=None, embed_layout=None):
    """Family-generic buffer filling: per-field weight init (see
    ``fill_weight_fields``), optimizer state zeroed, tokens/targets uniform
    ints."""
    import torch

    from dataflow.tasks.interop import torch_view

    gen = torch.Generator().manual_seed(seed)
    buffers = {}
    for spec in program.initial_objects:
        if spec.id in buffers:
            continue
        buf = backend.alloc("backing", spec.size_bytes)
        buffers[spec.id] = buf
        if spec.id.startswith("W_") and spec.id not in ("W_embed", "W_head"):
            fill_weight_fields(buf, wl, gen)
        elif spec.id == "W_embed":
            fill_weight_fields(buf, embed_layout or embed_weight_layout(dims), gen)
        elif spec.id == "W_head":
            fill_weight_fields(buf, head_layout or head_weight_layout(dims), gen)
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes,), torch.uint8).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
    return buffers
