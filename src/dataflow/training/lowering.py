"""Family-generic lowering helpers: exact sizes + initial values.

A family's *lowering* = the shaped chain (``shaped_program``) rewritten to
packed-layout byte truth, plus buffers filled with real values. Both steps
are family-invariant once the family declares its layouts:

    FamilyLayouts   which packed layout backs each weight object — per
                    LAYER for blocks (depth-dependent dtype policies make
                    W_0 and W_1 different bytes; heterogeneous families
                    differ by kind), plus the embed/head tables and any
                    special init distributions (qwen3.5's A_log/dt_bias).
    size_of_factory the shared object-id grammar (W_{i} / dW_{s}_{i} /
                    O_{i} / A_{s}_{r}_{i} / *_embed / *_head) -> bytes.
    initial_values_from_layouts
                    per-field typed init (norm weights ones, N(0, 0.02)
                    draws, specials), optimizer state zeroed, token ids.

No family writes this logic itself — a family lowering module is
``dims_of`` + a ``FamilyLayouts`` declaration + two thin wrappers
(docs/extending.md §4).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

from dataflow.core import ObjectSpec, OutputSpec, Program, RecomputeOption, TaskSpec
from dataflow.tasks.layouts import PackedLayout, grad_layout, opt_state_layout


@dataclass(frozen=True)
class FamilyLayouts:
    """Which packed layout backs each weight object of a family."""

    n_layers: int
    block_weight_at: Callable[[int], PackedLayout]   # layer -> W_{i} layout
    block_context_at: Callable[[int], PackedLayout]  # layer -> A_* layout
    embed: PackedLayout          # W_embed (tied families: the head layout)
    head: PackedLayout           # W_head (unused branches when tied)
    embed_ns: str = "embed"      # policy namespace: "head" when tied
    init_specials: Mapping[str, Callable] | None = None  # field -> (n, gen) -> tensor
    # layer -> M_{s}_{r}_{i} layout (metadata objects: never-recompute
    # forward artifacts — routing packs, selections). None/empty layout =
    # the layer has no metadata.
    block_meta_at: Callable[[int], PackedLayout] | None = None


def size_of_factory(dims, fl: FamilyLayouts):
    """Exact bytes for every object id the shaped builder emits. dW and O
    are sized from their OWN layouts (field-mirrored at grad/opt dtypes);
    block sizes are PER LAYER. Under a uniform all-bf16 policy every layer
    coincides with the historical uniform sizes."""
    p = dims.dtypes
    n = fl.n_layers
    wl_i = [fl.block_weight_at(i) for i in range(n)]
    a_i = [fl.block_context_at(i).total_bytes for i in range(n)]
    op = getattr(dims, "opt_policy", None)
    dw_i = [grad_layout(wl_i[i], p, layer=i, opt_policy=op).total_bytes
            for i in range(n)]
    m_i = ([fl.block_meta_at(i).total_bytes for i in range(n)]
           if fl.block_meta_at is not None else None)
    op = getattr(dims, "opt_policy", None)
    o_i = [opt_state_layout(wl_i[i], p, layer=i, opt_policy=op).total_bytes
           for i in range(n)]
    dw_e = grad_layout(fl.embed, p, ns=fl.embed_ns, opt_policy=op).total_bytes
    dw_h = grad_layout(fl.head, p, ns="head", opt_policy=op).total_bytes
    o_e = opt_state_layout(fl.embed, p, ns=fl.embed_ns,
                           opt_policy=op).total_bytes
    o_h = opt_state_layout(fl.head, p, ns="head",
                           opt_policy=op).total_bytes

    def size_of(oid: str) -> int | None:
        if oid.startswith("A_"):            # A_{s}_{r}_{i}
            return a_i[int(oid.rsplit("_", 1)[1])]
        if oid.startswith("M_") and m_i is not None:  # M_{s}_{r}_{i}
            return m_i[int(oid.rsplit("_", 1)[1])]
        if oid.startswith("dW_embed"):
            return dw_e
        if oid == "W_embed":
            return fl.embed.total_bytes
        if oid.startswith("dW_head"):
            return dw_h
        if oid == "W_head":
            return fl.head.total_bytes  # head packs [table | final_norm_w]
        if oid == "O_embed":
            return o_e
        if oid == "O_head":
            return o_h
        if oid.startswith("O_"):            # O_{i}
            return o_i[int(oid.split("_")[1])]
        if oid.startswith("dW_"):           # dW_{s}_{i}
            return dw_i[int(oid.rsplit("_", 1)[1])]
        if oid.startswith("W_"):            # W_{i}
            return wl_i[int(oid.split("_")[1])].total_bytes
        return None

    return size_of


def apply_exact_sizes(
    shaped: Program, lowering_tag: str, *, size_of,
) -> Program:
    """Rewrite a shaped program's object sizes to packed-layout truth and
    stamp optimizer tasks with their step index — shared by every family.
    ``size_of(object_id) -> bytes | None`` is the family's size map
    (``size_of_factory``)."""

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
                    saved_bytes=size_of(rw.object_id) or 0,
                    recompute_us=0.0, label="save",
                ),
            ) + tuple(o for o in rw.options if o.level != 0),
        )
        for rw in shaped.recompute_rewrites
    )
    objs = tuple(fix_obj(o) for o in shaped.initial_objects)
    # a fully-STATELESS optimizer assignment (e.g. sgd for every field
    # of a layer) sizes that O object to zero — drop it and scrub it
    # from its optimizer task rather than shipping a 0-byte object.
    # The same rule generalizes to GRADIENTS under frozen optimizer
    # policies (warm-up): a dW whose policy-filtered grad layout sizes
    # to zero is never created — its round-0 output, accumulate
    # mutates, and optimizer input all scrub, and tasks left
    # purposeless (embed_bwd with no outputs; an optimizer whose dW
    # vanished) drop from the chain entirely.
    dead = {o.id for o in objs
            if o.id.startswith("O_") and o.size_bytes == 0}
    for task in shaped.tasks:
        for out in task.outputs:
            if out.id.startswith("dW_") and (size_of(out.id) or 0) == 0:
                dead.add(out.id)
    if dead:
        objs = tuple(o for o in objs if o.id not in dead)
    final = {k: v for k, v in shaped.final_locations.items()
             if k not in dead}

    def scrub(t: TaskSpec) -> TaskSpec:
        touched = (set(t.inputs) | set(t.mutates)
                   | {o.id for o in t.outputs}) & dead
        if not dead or not touched:
            return t
        return replace(
            t,
            inputs=tuple(i for i in t.inputs if i not in dead),
            mutates=tuple(m for m in t.mutates if m not in dead),
            outputs=tuple(o for o in t.outputs if o.id not in dead),
        )

    def alive(t: TaskSpec) -> bool:
        if not t.outputs and not t.mutates:
            return False              # e.g. embed_bwd stripped of its dW
        if t.group == "optimizer" and not any(
                i.startswith("dW_") for i in t.inputs):
            # frozen layer/embed/head: nothing to apply
            return False
        return True

    tasks = tuple(
        scrub(fix_task(t, step_of(t.id)) if t.group == "optimizer"
              else fix_task(t, 0))
        for t in shaped.tasks
    )
    if dead:
        tasks = tuple(t for t in tasks if alive(t))
    return replace(
        shaped,
        initial_objects=objs,
        tasks=tasks,
        recompute_rewrites=new_rewrites,
        final_locations=final,
        metadata={**shaped.metadata, "lowering": lowering_tag},
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


def initial_values_from_layouts(program: Program, dims, fl: FamilyLayouts,
                                backend, *, seed: int = 0):
    """Allocate + fill pinned buffers for every initial object: per-field
    weight init at storage dtypes (``fill_weight_fields``), optimizer state
    zeroed, tokens/targets uniform ints. Returns {object_id: Buffer} for
    ``Engine.execute(initial_buffers=...)``."""
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
            layer = int(spec.id.split("_")[1])
            fill_weight_fields(buf, fl.block_weight_at(layer), gen,
                               special=fl.init_specials)
        elif spec.id == "W_embed":
            fill_weight_fields(buf, fl.embed, gen)
        elif spec.id == "W_head":
            fill_weight_fields(buf, fl.head, gen)
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes,), torch.uint8).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
    return buffers
