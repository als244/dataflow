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
    LlamaDims,
    adamw_state_layout,
    context_layout,
    embed_weight_layout,
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
    )


def _exact_sizes(cfg: ShapedLlamaConfig) -> dict[str, int]:
    dims = dims_of(cfg)
    wl = weight_layout(dims)
    el = embed_weight_layout(dims)
    cl = context_layout(dims)
    # optimizer state covers EVERY byte of the packed param object (padding
    # included): the AdamW executable derives its element count from the W
    # buffer's size, so the two must be sized from the same number. (Layouts
    # whose fields all land 256-aligned have zero padding and are unaffected.)
    return {
        "__W_block": wl.total_bytes,
        "__W_embed": el.total_bytes,
        "__A": cl.total_bytes,
        "__O_block": adamw_state_layout(wl.total_bytes // 2).total_bytes,
        "__O_embed": adamw_state_layout(el.total_bytes // 2).total_bytes,
    }


def _mapped_size(object_id: str, sizes: dict[str, int]) -> int | None:
    if object_id.startswith("A_"):
        return sizes["__A"]
    if object_id in ("W_embed", "W_head") or object_id.startswith(("dW_embed", "dW_head")):
        return sizes["__W_embed"]
    if object_id in ("O_embed", "O_head"):
        return sizes["__O_embed"]
    if object_id.startswith("O_"):
        return sizes["__O_block"]
    if object_id.startswith(("W_", "dW_")):
        return sizes["__W_block"]
    return None


def apply_exact_sizes(shaped: Program, sizes: dict[str, int], lowering_tag: str) -> Program:
    """Rewrite a shaped program's object sizes to packed-layout truth and
    stamp optimizer tasks with their step index — shared by every family's
    lowering (the size MAP is family-specific; the id families are not)."""

    def fix_obj(o: ObjectSpec) -> ObjectSpec:
        size = _mapped_size(o.id, sizes)
        return o if size is None else replace(o, size_bytes=size, tensor=None)

    def fix_out(o: OutputSpec) -> OutputSpec:
        size = _mapped_size(o.id, sizes)
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
                RecomputeOption(level=0, saved_bytes=sizes["__A"], recompute_us=0.0, label="save"),
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

    return fill_initial_values(program, dims_of(cfg), _wl(dims_of(cfg)), backend, seed=seed)


def fill_initial_values(program: Program, dims, wl, backend, *, seed: int = 0):
    """Family-generic buffer filling: weights N(0, 0.02) bf16 with every
    ``*_norm_w`` layout field set to ones, optimizer state zeroed,
    tokens/targets uniform ints. The generation order is part of golden
    comparability — it follows the program's initial-object order."""
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
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
            for f in wl.fields:
                if f.name.endswith("_norm_w"):
                    start = f.offset_bytes // 2
                    flat[start : start + f.shape[0]] = 1.0
        elif spec.id in ("W_embed", "W_head"):
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
    return buffers
