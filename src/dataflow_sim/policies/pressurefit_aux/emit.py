"""Interval-to-trigger emission for PressureFit."""
from __future__ import annotations

from dataclasses import replace

from dataflow_sim.policies.pressurefit_aux.types import (
    _PrefetchAssignments,
    _TransitionPlan,
)
from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger


def _emit_chain(
    bare: TaskChain,
    transition_plan: _TransitionPlan,
    prefetch_assignments: _PrefetchAssignments,
    *,
    coalesce_clean_gaps: bool = False,
) -> TaskChain:
    task_count = len(bare.tasks)
    releases: list[list[str]] = [[] for _ in range(task_count)]
    offloads: list[list[str]] = [[] for _ in range(task_count)]
    prefetches = [list(oids) for oids in prefetch_assignments]
    pre_placed = set(transition_plan.preplaced)

    for oid, transition in transition_plan.departures:
        fire_task = transition.departure_after_task
        assert fire_task is not None
        if transition.departure_action == "offload":
            offloads[fire_task].append(oid)
        elif transition.departure_action == "release":
            releases[fire_task].append(oid)

    for i in range(task_count):
        if releases[i]:
            releases[i] = list(dict.fromkeys(releases[i]))
        if offloads[i]:
            offloads[i] = list(dict.fromkeys(offloads[i]))

        # Releasing a clean fast copy and immediately restoring that same
        # object from its already-current backing copy is exactly a no-op.
        # Normalize that schedule-level zero-width gap to continuous
        # residency.  Offload/prefetch conflicts are not equivalent: the
        # inbound source does not become current until the writeback ends.
        clean_noops = (
            set(releases[i]) & set(prefetches[i])
            if coalesce_clean_gaps
            else set()
        )
        if clean_noops:
            releases[i] = [oid for oid in releases[i] if oid not in clean_noops]
            prefetches[i] = [oid for oid in prefetches[i] if oid not in clean_noops]
        conflicts = set(offloads[i]) & set(prefetches[i])
        if conflicts:
            raise RuntimeError(
                "pressurefit internal error: transition plan emitted same-task "
                f"offload/prefetch for {sorted(conflicts)!r} after task {i}"
            )

    backing_objs = {o.id: o for o in bare.initial_memory if o.location == "backing"}
    new_initial = list(bare.initial_memory)
    for oid in sorted(pre_placed):
        src = backing_objs[oid]
        # Preserve descriptive schema metadata; `type` never participates in
        # PressureFit placement, ranking, or scheduling decisions.
        new_initial.append(Object(
            id=src.id,
            size=src.size,
            location="fast",
            type=src.type,
        ))

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        new_tasks.append(replace(
            task,
            releases_after=releases[i],
            offload_after=[TransferTrigger(obj_id=o) for o in offloads[i]],
            prefetch_after=[TransferTrigger(obj_id=o) for o in prefetches[i]],
        ))

    return replace(bare, initial_memory=new_initial, tasks=new_tasks)
