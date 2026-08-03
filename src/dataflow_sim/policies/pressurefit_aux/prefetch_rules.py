"""PressureFit prefetch-boundary selection rules."""
from __future__ import annotations

import math
from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _pressure_start,
)
from dataflow_sim.policies.pressurefit_aux.types import (
    _PrefetchAssignments,
    _PrefetchRuleKind,
    _PrefetchWindow,
    _TransitionPlan,
)


@dataclass(frozen=True, slots=True)
class _PrefetchJob:
    oid: str
    earliest: int
    latest: int
    deadline: int
    tau: int
    first_use: int
    # Interval entry boundary `a`. The analytic model reserves the object's
    # bytes from its preceding lead boundary; firing the trigger earlier
    # materializes bytes at boundaries the model never charged.
    entry_a: int
    size: int


def _prefetch_job(
    oid: str,
    window: _PrefetchWindow,
    facts: _Facts,
    inbound_bw: int | None,
    rule: _PrefetchRuleKind,
) -> _PrefetchJob | None:
    first_use = window.first_use_task
    if first_use is None:
        return None

    earliest = window.earliest_after_task
    latest = window.latest_after_task
    if rule == "interval-entry":
        latest = max(
            earliest,
            min(latest, max(0, window.entry_boundary - 1)),
        )
    tau = (
        max(1, math.ceil(facts.sizes[oid] / inbound_bw))
        if inbound_bw is not None and inbound_bw > 0
        else 0
    )
    return _PrefetchJob(
        oid=oid,
        earliest=earliest,
        latest=latest,
        deadline=facts.task_start[first_use],
        tau=tau,
        first_use=first_use,
        entry_a=window.entry_boundary,
        size=facts.sizes[oid],
    )


def _latest_safe_fire(job: _PrefetchJob, facts: _Facts) -> int:
    """Return the latest individually feasible enqueue boundary."""
    for task_index in range(job.latest, job.earliest - 1, -1):
        if facts.task_end[task_index] + job.tau <= job.deadline:
            return task_index
    return job.latest


def _assign_prefetch_jobs(
    jobs: list[_PrefetchJob],
    facts: _Facts,
    *,
    pool: list[int] | None = None,
    cap: int | None = None,
    extra_pressure: list[int] | None = None,
    prefetch_headroom: bool = True,
) -> tuple[list[list[str]], list[dict[str, int]]]:
    """Pack inbound jobs backward from their deadlines as one FIFO queue.

    When `pool` and `cap` are given, packing is pressure-aware: a job may
    not fire earlier than compute pressure allows. Firing on task `t`
    materializes destination bytes before the modeled lead boundary; the
    clamp slides the
    trigger later until every newly covered boundary still satisfies the
    strict capacity inequality, and commits the accepted coverage to `pool`
    so subsequent jobs see it.

    The clamp deliberately charges from the trigger boundary, not from a
    queue-aware estimate of the actual transfer start. A queue-aware variant
    (charge from `max(enqueue, stream cursor)`) was measured and rejected:
    it interpolates between unclamped packing and this clamp — inheriting
    unclamped packing's repair divergence on long tight chains while losing
    this clamp's conservative-rescuer wins — instead of dominating either.
    """
    prefetches: list[list[str]] = [[] for _ in range(facts.n)]
    prefetch_order: list[dict[str, int]] = [dict() for _ in range(facts.n)]
    if not jobs:
        return prefetches, prefetch_order
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)

    next_start = math.inf
    assignments: list[tuple[int, _PrefetchJob]] = []
    for job in sorted(jobs, key=lambda j: (j.deadline, j.latest, j.oid), reverse=True):
        latest_finish = (
            job.deadline if math.isinf(next_start)
            else min(job.deadline, next_start)
        )
        desired_start = latest_finish - job.tau
        for t in range(job.latest, job.earliest - 1, -1):
            if facts.task_end[t] <= desired_start:
                fire = t
                break
        else:
            fire = job.earliest
        if pool is not None and cap is not None:
            fire = _pressure_clamped_fire(
                job,
                fire,
                pool,
                cap,
                extra_pressure,
                facts,
                prefetch_headroom=prefetch_headroom,
            )
        assignments.append((fire, job))
        next_start = max(facts.task_end[fire], desired_start)

    for fire, job in assignments:
        prefetches[fire].append(job.oid)
        prefetch_order[fire][job.oid] = job.first_use
    return prefetches, prefetch_order


def _apply_prefetch_rule(
    transition_plan: _TransitionPlan,
    facts: _Facts,
    inbound_bw: int | None,
    rule: _PrefetchRuleKind,
    *,
    pool: list[int] | None = None,
    cap: int | None = None,
    extra_pressure: list[int] | None = None,
    prefetch_headroom: bool = True,
) -> _PrefetchAssignments:
    """Choose one enqueue boundary for every prefetch transition."""
    packs_fifo = rule in ("packed-fifo", "packed-fit")
    direct: list[list[str]] = [[] for _ in range(facts.n)]
    direct_order: list[dict[str, int]] = [dict() for _ in range(facts.n)]
    packed_jobs: list[_PrefetchJob] = []

    for oid, transition in transition_plan.prefetches:
        assert transition.prefetch is not None
        job = _prefetch_job(
            oid,
            transition.prefetch,
            facts,
            inbound_bw,
            rule,
        )
        if job is not None and packs_fifo:
            packed_jobs.append(job)
            continue
        fire = (
            _latest_safe_fire(job, facts)
            if job is not None
            else transition.prefetch.latest_after_task
        )
        direct[fire].append(oid)
        direct_order[fire][oid] = job.first_use if job is not None else facts.n

    prefetches, prefetch_order = _assign_prefetch_jobs(
        packed_jobs,
        facts,
        pool=pool,
        cap=cap,
        extra_pressure=extra_pressure,
        prefetch_headroom=prefetch_headroom,
    )
    for task_idx, oids in enumerate(direct):
        if oids:
            prefetches[task_idx].extend(oids)
            prefetch_order[task_idx].update(direct_order[task_idx])
        if prefetches[task_idx]:
            prefetches[task_idx] = sorted(
                dict.fromkeys(prefetches[task_idx]),
                key=lambda oid: (prefetch_order[task_idx].get(oid, facts.n), oid),
            )
    return tuple(tuple(oids) for oids in prefetches)


def _pressure_clamped_fire(
    job: _PrefetchJob,
    fire: int,
    pool: list[int],
    cap: int,
    extra_pressure: list[int],
    facts: _Facts,
    *,
    prefetch_headroom: bool = True,
) -> int:
    """Slide `fire` later until its newly covered boundaries fit the cap."""
    model_entry = _pressure_start(
        job.oid,
        job.entry_a,
        facts,
        prefetch_headroom=prefetch_headroom,
    )
    if fire >= model_entry:
        return fire
    clamped = fire
    for x in range(model_entry - 1, fire - 1, -1):
        idx = x + 1
        if (
            pool[idx]
            + job.size
            + facts.next_outputs[idx]
            + extra_pressure[idx]
            > cap
        ):
            clamped = x + 1
            break
    for x in range(clamped, model_entry):
        pool[x + 1] += job.size
    return clamped
