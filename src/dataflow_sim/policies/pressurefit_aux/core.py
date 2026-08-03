"""Shared interval model for PressureFit."""
from __future__ import annotations

import math
from collections import defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from dataflow_sim.policies._common import _object_sizes
from dataflow_sim.policies.pressurefit_aux.types import (
    _ResidencyAnchor,
    _ResidencySpan,
)
from dataflow_sim.core.schema import TaskChain


@dataclass(frozen=True, slots=True)
class _Facts:
    n: int
    sizes: dict[str, int]
    producer: dict[str, int]
    uses: dict[str, tuple[int, ...]]
    mutators: dict[str, tuple[int, ...]]
    backing_ids: set[str]
    compute_ids: set[str]
    final_locations: dict[str, str]
    task_index: dict[str, int]
    task_start: list[int]
    task_end: list[int]
    next_outputs: list[int]
    inbound_bandwidth: int | None
    outbound_bandwidth: int | None


def _build_facts(chain: TaskChain) -> _Facts:
    sizes = _object_sizes(chain)
    n = len(chain.tasks)
    producer = {o.id: -1 for o in chain.initial_memory}
    for i, task in enumerate(chain.tasks):
        for out in task.outputs:
            producer[out.id] = i

    uses: dict[str, list[int]] = defaultdict(list)
    mutators: dict[str, set[int]] = defaultdict(set)
    for i, task in enumerate(chain.tasks):
        for inp in task.inputs:
            uses[inp].append(i)
        for oid in task.mutates_inputs:
            mutators[oid].add(i)

    task_start: list[int] = []
    task_end: list[int] = []
    t = 0
    for task in chain.tasks:
        task_start.append(t)
        t += task.runtime
        task_end.append(t)

    next_outputs = [0] * (n + 1)
    for b in range(-1, n - 1):
        task = chain.tasks[b + 1]
        next_outputs[b + 1] = sum(
            out.size for out in task.outputs if out.location == "fast"
        )

    return _Facts(
        n=n,
        sizes=sizes,
        producer=producer,
        uses={k: tuple(sorted(v)) for k, v in uses.items()},
        mutators={k: tuple(sorted(v)) for k, v in mutators.items()},
        backing_ids={o.id for o in chain.initial_memory if o.location == "backing"},
        compute_ids={o.id for o in chain.initial_memory if o.location == "fast"},
        final_locations=dict(chain.final_locations),
        task_index={task.id: index for index, task in enumerate(chain.tasks)},
        task_start=task_start,
        task_end=task_end,
        next_outputs=next_outputs,
        inbound_bandwidth=chain.bandwidth_from_slow,
        outbound_bandwidth=chain.bandwidth_to_slow,
    )


def _transfer_time(size: int, bandwidth: int | None) -> int:
    if bandwidth is None or bandwidth <= 0:
        return 0
    return max(1, math.ceil(size / bandwidth))


def _anchor_records(oid: str, facts: _Facts) -> tuple[_ResidencyAnchor, ...]:
    """Return every mandatory fast-residency point for ``oid``.

    Final fast residency is a first-class liveness anchor.  Omitting it was
    the root modeling error behind the retained-state PressureFit failure.
    """
    out: set[_ResidencyAnchor] = set()
    if oid in facts.compute_ids:
        out.add(_ResidencyAnchor(-1, "initial", None))
    p = facts.producer.get(oid, -1)
    if p >= 0:
        out.add(_ResidencyAnchor(p, "producer", p))
    for u in facts.uses.get(oid, []):
        out.add(_ResidencyAnchor(u - 1, "use", u))
    if facts.final_locations.get(oid) == "fast":
        out.add(_ResidencyAnchor(max(-1, facts.n - 1), "final_fast", None))
    return tuple(sorted(out, key=lambda anchor: (anchor.boundary, anchor.kind)))


def _anchors(oid: str, facts: _Facts) -> list[int]:
    """Return unique anchor boundaries for the reducer's hot path."""
    return sorted({anchor.boundary for anchor in _anchor_records(oid, facts)})


def _pressure_start(
    oid: str,
    span_start: int,
    facts: _Facts,
    *,
    prefetch_headroom: bool = True,
) -> int:
    """Return the boundary charged by one analytical pressure view."""
    producer = facts.producer.get(oid, -1)
    if prefetch_headroom and span_start > -1 and span_start != producer:
        return span_start - 1
    return span_start


def _pool_size(
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    *,
    prefetch_headroom: bool = True,
) -> list[int]:
    # Difference-array range addition preserves the exact inclusive boundary
    # totals while reducing seed construction from O(total span length) to
    # O(spans + boundaries).
    delta = [0] * (facts.n + 2)
    for oid, ivs in intervals.items():
        for a, b in ivs:
            start = max(
                -1,
                _pressure_start(
                    oid,
                    a,
                    facts,
                    prefetch_headroom=prefetch_headroom,
                ),
            ) + 1
            end = min(facts.n - 1, b) + 1
            if start > end:
                continue
            delta[start] += facts.sizes[oid]
            delta[end + 1] -= facts.sizes[oid]
    pool = [0] * (facts.n + 1)
    running = 0
    for index in range(facts.n + 1):
        running += delta[index]
        pool[index] = running
    return pool


def _fire_task_for_interval(
    oid: str,
    a: int,
    b: int,
    facts: _Facts,
) -> int | None:
    candidate = -1
    p = facts.producer.get(oid, -1)
    if p >= 0 and a <= p <= b:
        candidate = p
    uses = facts.uses.get(oid, ())
    hi = bisect_right(uses, b + 1)
    if hi and uses[hi - 1] >= a + 1:
        candidate = max(candidate, uses[hi - 1])
    if candidate >= 0:
        return min(facts.n - 1, candidate)
    # Initial residency is an anchor even when a split segment contains no
    # producer/use. TaskChain triggers fire only after tasks, so task 0 is the
    # earliest legal departure point.
    if a == -1 and facts.n:
        return 0
    return None


def _first_use_in_interval(
    oid: str,
    a: int,
    b: int,
    facts: _Facts,
) -> int | None:
    uses = facts.uses.get(oid, ())
    pos = bisect_left(uses, a + 1)
    if pos < len(uses) and uses[pos] <= b + 1:
        return uses[pos]
    return None


def _departing_before_next(
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    idx: int,
    *,
    prefetch_headroom: bool = True,
) -> int:
    boundary = idx - 1
    if boundary < 0 or boundary >= facts.n - 1:
        return 0
    total = 0
    for oid, ivs in intervals.items():
        for a, b in ivs:
            if not (
                _pressure_start(
                    oid,
                    a,
                    facts,
                    prefetch_headroom=prefetch_headroom,
                )
                <= boundary
                <= b
            ):
                continue
            if _fire_task_for_interval(oid, a, b, facts) == boundary:
                total += facts.sizes[oid]
    return total


def _nonblocking_arrivals_before_next(
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    idx: int,
    *,
    prefetch_headroom: bool = True,
) -> int:
    boundary = idx - 1
    if boundary < 0 or boundary >= facts.n - 1:
        return 0
    total = 0
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for a, b in ivs:
            if (
                a <= -1
                or a == p
                or _pressure_start(
                    oid,
                    a,
                    facts,
                    prefetch_headroom=prefetch_headroom,
                )
                != boundary
            ):
                continue
            first_use = _first_use_in_interval(oid, a, b, facts)
            if first_use is None or first_use > boundary + 1:
                total += facts.sizes[oid]
    return total


def _modeled_boundary_need(
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    idx: int,
    pool: list[int] | None = None,
    *,
    prefetch_headroom: bool = True,
) -> int:
    if pool is None:
        pool = _pool_size(
            facts,
            intervals,
            prefetch_headroom=prefetch_headroom,
        )
    return (
        pool[idx]
        - _departing_before_next(
            facts,
            intervals,
            idx,
            prefetch_headroom=prefetch_headroom,
        )
        - _nonblocking_arrivals_before_next(
            facts,
            intervals,
            idx,
            prefetch_headroom=prefetch_headroom,
        )
        + facts.next_outputs[idx]
    )
