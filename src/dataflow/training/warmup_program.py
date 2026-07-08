"""Dense warm-up program surgery: the indexer-only objective.

The warm-up stage's TRAINING OBJECTIVE is the indexer KL alone — the
paper's L_I (per-layer) / L^I_multi (IndexShare groups). Cross-entropy
trains nothing (every consumer is frozen; the indexer is severed from
CE by the detach seam) and with the main model frozen it isn't even
informative monitoring. This module rewrites a STANDARD shaped program
into the specialized warm-up form:

- NO head: the fused head_loss task, its ``targets_{s}_{r}`` inputs and
  the (t, vocab)-scale GEMMs are gone. ``W_head`` remains an (untouched)
  initial object — it is model state, not a participant.
- NO dy chain: block backwards take ``(A, x, W[, M..., dM])`` and emit
  no ``dy_*``; ``embed_bwd`` (whose sole purpose is dW_embed) and the
  embed/head optimizer tasks are dropped.
- LOSS = THE OBJECTIVE: ``loss_{s}_{r}`` keeps its id (train_loop and
  every bench reader stay unchanged) but is now the KL accumulator —
  CREATED by the first KL-contributing backward in chain order and
  accumulate-mutated by the rest. Contributors are every layer that is
  not a group FOLLOWER (dsv32: all layers; glm52: leaders), matching
  where the engine can actually form the group-centroid KL value.

``build_shaped_program(indexer_only_objective=True)`` dispatches here;
the common builder carries no warm-up branches (Shein's call: the
special case lives in its own file).
"""
from __future__ import annotations

from dataclasses import replace

from dataflow.core import OutputSpec, Program, TaskSpec, TensorMeta


def _sr_of(task_id: str) -> tuple[int, int]:
    # block_bwd_{s}_{r}_{i}
    parts = task_id.split("_")
    return int(parts[2]), int(parts[3])


def to_indexer_warmup(program: Program, *, followers: frozenset[int]) -> Program:
    """Rewrite a standard shaped program into the indexer-only warm-up
    form (see module docstring). ``followers`` are the layer indices
    that consume another layer's metadata (group followers) — they
    deposit into dM and contribute NO direct loss term."""
    drop_prefixes = ("head_loss_", "embed_bwd_", "optimizer_head",
                     "optimizer_embed")
    dead_objects: set[str] = set()
    for t in program.tasks:
        if t.id.startswith(drop_prefixes):
            dead_objects.update(o.id for o in t.outputs)
    # dy objects are block-bwd outputs; collect for input scrubbing
    for t in program.tasks:
        for o in t.outputs:
            if o.id.startswith(("dy_",)):
                dead_objects.add(o.id)
    # the loss objects are re-owned below, not dead
    dead_objects = {o for o in dead_objects if not o.startswith("loss_")}

    loss_created: set[tuple[int, int]] = set()
    tasks: list[TaskSpec] = []
    for t in program.tasks:
        if t.id.startswith(drop_prefixes):
            continue
        if not t.id.startswith("block_bwd_"):
            tasks.append(t)
            continue
        s, r = _sr_of(t.id)
        lid = f"loss_{s}_{r}"
        inputs = tuple(i for i in t.inputs if i not in dead_objects)
        outputs = tuple(o for o in t.outputs if o.id not in dead_objects)
        mutates = tuple(m for m in t.mutates if m not in dead_objects)
        layer = t.block_params.get("layer")
        if layer not in followers:
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
