"""Freeze/objective surgery: FreezePlan -> specialized program.

``build_shaped_program(freeze=plan)`` dispatches here after building
the standard program; the common builder carries no freeze branches.
Two objective shapes:

CE (``plan.objective == "ce"``, some layers fully frozen):
- TRUNCATED layers (fully frozen, nothing below trains): backward task
  dropped; their dy objects never exist; their A output and its
  recompute rewrite are dropped too (no backward will ever read them —
  the forward runs context-free, exactly the recomputed-fwd path the
  block classes already support).
- PASS-THROUGH layers (fully frozen, something below trains): backward
  kept for its dgrads; it has no dW (the zero-byte scrub prunes the
  object and the launch tolerates dw=None) — guards-first, per the
  design note.
- embed_bwd survives only if the embedding trains AND dy reaches it;
  head_loss always survives under CE (a frozen head still passes
  dgrad; its dW_head prunes via the zero-byte scrub).

INDEXER-KL (``plan.objective == "indexer_kl"``, the dense warm-up):
- NO head/targets/CE, NO dy anywhere; ``loss_{s}_{r}`` keeps its id
  but is the KL accumulator — CREATED by the first contributing
  backward in chain order (plan.loss_contributors), accumulated by the
  rest. See the warm-up notes; this path is the former
  warmup_program.to_indexer_warmup, byte-for-byte.
"""
from __future__ import annotations

from dataclasses import replace

from dataflow.core import OutputSpec, Program, TaskSpec, TensorMeta


def _sri_of(task_id: str) -> tuple[int, int, int]:
    parts = task_id.split("_")            # block_bwd_{s}_{r}_{i} etc.
    return int(parts[-3]), int(parts[-2]), int(parts[-1])


def to_frozen_form(program: Program, plan) -> Program:
    if plan.objective == "indexer_kl":
        return _indexer_kl_form(program, plan)
    return _ce_frozen_form(program, plan)


# ---------------------------------------------------------------- CE

def _ce_frozen_form(program: Program, plan) -> Program:
    n = plan.n_layers
    dead: set[str] = set()
    drop_tasks: set[str] = set()

    for t in program.tasks:
        if t.id.startswith("block_bwd_"):
            s, r, i = _sri_of(t.id)
            if not plan.emit_bwd[i]:
                drop_tasks.add(t.id)
                dead.update(o.id for o in t.outputs)
            # NOTE: kept backwards keep ALL their outputs — the boundary
            # bwd still writes its dy into a consumer-less (disposable)
            # object. Stripping outputs from kept tasks would break the
            # positional launch contract (outputs[0] IS the dx buffer).
        elif t.id.startswith("block_recompute_"):
            s, r, i = _sri_of(t.id)
            if not plan.save_ctx[i]:
                drop_tasks.add(t.id)
        elif t.id.startswith("embed_bwd") and not plan.embed_trainable:
            drop_tasks.add(t.id)
            dead.update(o.id for o in t.outputs)

    # context objects nobody will read (their layer has no backward)
    ctx_dead: set[str] = set()
    for t in program.tasks:
        if t.id.startswith("block_fwd_"):
            s, r, i = _sri_of(t.id)
            if not plan.save_ctx[i]:
                ctx_dead.update(o.id for o in t.outputs
                                if o.id.startswith("A_"))
    dead |= ctx_dead

    tasks: list[TaskSpec] = []
    for t in program.tasks:
        if t.id in drop_tasks:
            continue
        inputs = tuple(x for x in t.inputs if x not in dead)
        outputs = tuple(o for o in t.outputs if o.id not in dead)
        mutates = tuple(m for m in t.mutates if m not in dead)
        if (inputs, outputs, mutates) != (t.inputs, t.outputs, t.mutates):
            t = replace(t, inputs=inputs, outputs=outputs, mutates=mutates)
        tasks.append(t)

    rewrites = tuple(rw for rw in program.recompute_rewrites
                     if rw.object_id not in ctx_dead)
    final = {k: v for k, v in program.final_locations.items()
             if k not in dead}
    return replace(program, tasks=tuple(tasks),
                   recompute_rewrites=rewrites, final_locations=final)


# ------------------------------------------------------- indexer KL

def _indexer_kl_form(program: Program, plan) -> Program:
    contributors = frozenset(plan.loss_contributors)
    drop_prefixes = ("head_loss_", "embed_bwd_", "optimizer_head",
                     "optimizer_embed")
    dead_objects: set[str] = set()
    for t in program.tasks:
        if t.id.startswith(drop_prefixes):
            dead_objects.update(o.id for o in t.outputs)
    for t in program.tasks:
        for o in t.outputs:
            if o.id.startswith(("dy_",)):
                dead_objects.add(o.id)
    dead_objects = {o for o in dead_objects if not o.startswith("loss_")}

    loss_created: set[tuple[int, int]] = set()
    tasks: list[TaskSpec] = []
    for t in program.tasks:
        if t.id.startswith(drop_prefixes):
            continue
        if not t.id.startswith("block_bwd_"):
            tasks.append(t)
            continue
        s, r, i = _sri_of(t.id)
        lid = f"loss_{s}_{r}"
        inputs = tuple(x for x in t.inputs if x not in dead_objects)
        outputs = tuple(o for o in t.outputs if o.id not in dead_objects)
        mutates = tuple(m for m in t.mutates if m not in dead_objects)
        if i in contributors:
            if (s, r) not in loss_created:
                outputs = outputs + (OutputSpec(
                    id=lid, size_bytes=4, role="output",
                    tensor=TensorMeta(dtype="fp32", shape=(1,))),)
                loss_created.add((s, r))
            else:
                inputs = inputs + (lid,)
                mutates = mutates + (lid,)
        tasks.append(replace(t, inputs=inputs, outputs=outputs,
                             mutates=mutates))

    initial = tuple(o for o in program.initial_objects
                    if not o.id.startswith("targets_"))
    final = {k: v for k, v in program.final_locations.items()
             if k not in dead_objects}
    return replace(program, tasks=tuple(tasks), initial_objects=initial,
                   final_locations=final)
