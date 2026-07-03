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
    w_elems = sum(
        int(__import__("math").prod(f.shape)) for f in wl.fields
    )
    e_elems = dims.vocab_size * dims.d_model
    sizes: dict[str, int] = {}
    sizes["__W_block"] = wl.total_bytes
    sizes["__W_embed"] = el.total_bytes
    sizes["__A"] = cl.total_bytes
    sizes["__O_block"] = adamw_state_layout(w_elems).total_bytes
    sizes["__O_embed"] = adamw_state_layout(e_elems).total_bytes
    return sizes


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
    sizes = _exact_sizes(cfg)

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
        metadata={**shaped.metadata, "lowering": "llama3-exact-v1"},
    )


def initial_values(program: Program, cfg: ShapedLlamaConfig, backend, *, seed: int = 0):
    """Allocate + fill pinned buffers for every initial object.

    Weights: N(0, 0.02) bf16; norm weights: ones; optimizer state: zeros;
    tokens/targets: uniform ints. Returns {object_id: Buffer} for
    Engine.execute(initial_buffers=...).
    """
    import torch

    from dataflow.tasks.interop import torch_view
    from dataflow.tasks.layouts import weight_layout as _wl

    dims = dims_of(cfg)
    gen = torch.Generator().manual_seed(seed)
    buffers = {}
    wl = _wl(dims)
    for spec in program.initial_objects:
        if spec.location != "backing" and spec.id in buffers:
            continue
        if spec.id in buffers:
            continue
        buf = backend.alloc("backing", spec.size_bytes)
        buffers[spec.id] = buf
        if spec.id.startswith("W_") and spec.id not in ("W_embed", "W_head"):
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
            for name in ("attn_norm_w", "ffn_norm_w"):
                f = wl.field(name)
                start = f.offset_bytes // 2
                n = f.shape[0]
                flat[start : start + n] = 1.0
        elif spec.id in ("W_embed", "W_head"):
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
        elif spec.id.startswith("X_"):
            # embed-on-host staging: the host gathers W_embed rows here each
            # round (train_loop); zero until then
            torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16).zero_()
    return buffers


def host_embed_state(cfg: ShapedLlamaConfig, *, seed: int = 0):
    """CPU-resident embedding state for cfg.embed_on_host mode.

    The table itself is bf16 (model dtype); gradient accumulation and
    optimizer moments use cfg.embed_host_accum — a training-spec choice
    ("float32" or "bfloat16"), not a hardwired dtype. Seeded independently
    of initial_values' order-dependent stream so it is stable across chain
    variants."""
    import torch

    dims = dims_of(cfg)
    accum = getattr(torch, cfg.embed_host_accum)
    gen = torch.Generator().manual_seed(seed + 7919)
    w = (torch.randn(dims.vocab_size, dims.d_model, generator=gen) * 0.02).to(torch.bfloat16)
    return {
        "W": w,
        "M": torch.zeros(dims.vocab_size, dims.d_model, dtype=accum),
        "V": torch.zeros(dims.vocab_size, dims.d_model, dtype=accum),
        "dW": torch.zeros(dims.vocab_size, dims.d_model, dtype=accum),
    }


def host_embed_adamw(host, *, lr, beta1, beta2, eps, weight_decay, step) -> None:
    """CPU AdamW on the host embedding table, mirroring ops.adamw_step's
    structure (moments stored in the configured accum dtype; when that is
    bf16 the round-trip matches the GPU path bit-for-bit in structure)."""
    import torch

    w32 = host["W"].float()
    g = host["dW"].float()
    m = host["M"].float().mul_(beta1).add_(g, alpha=1 - beta1)
    v = host["V"].float().mul_(beta2).addcmul_(g, g, value=1 - beta2)
    host["M"].copy_(m.to(host["M"].dtype))
    host["V"].copy_(v.to(host["V"].dtype))
    mhat = host["M"].float() / (1 - beta1 ** step)
    vhat = host["V"].float() / (1 - beta2 ** step)
    w32 -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * w32)
    host["W"].copy_(w32.to(torch.bfloat16))
    host["dW"].zero_()
