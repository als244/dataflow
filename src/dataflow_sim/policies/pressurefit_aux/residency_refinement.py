"""Implementation-specific refinements of a pressure-fit residency plan."""
from __future__ import annotations

import math
from typing import NamedTuple

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _pool_size,
    _pressure_start,
)
from dataflow_sim.policies.pressurefit_aux.transitions import _build_object_transitions
from dataflow_sim.policies.pressurefit_aux.types import _IntervalSet, _ResidencySpan


class _LeadTimeRequest(NamedTuple):
    deadline: int
    oid: str
    span_index: int
    transfer_time: int
    current_start: int
    earliest_start: int


def _lead_time_requests(
    facts: _Facts,
    intervals: _IntervalSet,
    inbound_bw: int,
    *,
    coalesce_clean_gaps: bool,
) -> list[_LeadTimeRequest]:
    requests: list[_LeadTimeRequest] = []
    for oid, spans in intervals.items():
        transitions = _build_object_transitions(
            oid,
            spans,
            facts,
            coalesce_clean_gaps=coalesce_clean_gaps,
        )
        for span_index, transition in enumerate(transitions):
            start = transition.span.start
            window = transition.prefetch
            if window is None or start <= 0 or window.first_use_task is None:
                continue
            if window.earliest_after_task >= start:
                continue
            requests.append(_LeadTimeRequest(
                deadline=facts.task_start[window.first_use_task],
                oid=oid,
                span_index=span_index,
                transfer_time=max(1, math.ceil(facts.sizes[oid] / inbound_bw)),
                current_start=start,
                earliest_start=window.earliest_after_task,
            ))
    return requests


def _ideal_entry(
    request: _LeadTimeRequest,
    next_start_time: float,
    task_ends: list[int],
) -> int:
    ideal_start_time = max(
        0,
        min(request.deadline, next_start_time) - request.transfer_time,
    )
    ideal_fire = -1
    for task_index, end_time in enumerate(task_ends):
        if end_time > ideal_start_time:
            break
        ideal_fire = task_index
    return max(ideal_fire + 1, request.earliest_start)


def _first_fitting_entry(
    request: _LeadTimeRequest,
    ideal_entry: int,
    facts: _Facts,
    pool: list[int],
    extra_pressure: list[int],
    cap: int,
    *,
    prefetch_headroom: bool,
) -> tuple[int, int]:
    old_pressure_start = _pressure_start(
        request.oid,
        request.current_start,
        facts,
        prefetch_headroom=prefetch_headroom,
    )
    size = facts.sizes[request.oid]
    for attempted_entry in range(ideal_entry, request.current_start):
        new_pressure_start = _pressure_start(
            request.oid,
            attempted_entry,
            facts,
            prefetch_headroom=prefetch_headroom,
        )
        if all(
            pool[boundary + 1]
            + size
            + facts.next_reservations[boundary + 1]
            + extra_pressure[boundary + 1]
            <= cap
            for boundary in range(new_pressure_start, old_pressure_start)
        ):
            return attempted_entry, old_pressure_start
    return request.current_start, old_pressure_start


def _extend_inbound_lead_time(
    facts: _Facts,
    intervals: _IntervalSet,
    cap: int | None,
    inbound_bw: int | None,
    extra_pressure: list[int] | None = None,
    *,
    prefetch_headroom: bool = True,
    coalesce_clean_gaps: bool = False,
) -> None:
    """Move prefetch interval entries left when strict capacity permits."""
    if cap is None or inbound_bw is None or inbound_bw <= 0:
        return
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)

    requests = _lead_time_requests(
        facts,
        intervals,
        inbound_bw,
        coalesce_clean_gaps=coalesce_clean_gaps,
    )
    if not requests:
        return

    pool = _pool_size(
        facts,
        intervals,
        prefetch_headroom=prefetch_headroom,
    )
    next_start_time = math.inf
    for request in sorted(
        requests,
        key=lambda item: (-item.deadline, item.oid, item.span_index),
    ):
        spans = intervals.get(request.oid)
        if (
            spans is None
            or request.span_index >= len(spans)
            or spans[request.span_index].start != request.current_start
        ):
            continue

        ideal_entry = _ideal_entry(request, next_start_time, facts.task_end)
        if ideal_entry >= request.current_start:
            next_start_time = facts.task_end[request.current_start - 1]
            continue

        chosen_entry, old_pressure_start = _first_fitting_entry(
            request,
            ideal_entry,
            facts,
            pool,
            extra_pressure,
            cap,
            prefetch_headroom=prefetch_headroom,
        )
        if chosen_entry >= request.current_start:
            next_start_time = facts.task_end[request.current_start - 1]
            continue
        new_pressure_start = _pressure_start(
            request.oid,
            chosen_entry,
            facts,
            prefetch_headroom=prefetch_headroom,
        )
        size = facts.sizes[request.oid]
        for boundary in range(new_pressure_start, old_pressure_start):
            pool[boundary + 1] += size
        end = spans[request.span_index].end
        spans[request.span_index] = _ResidencySpan(chosen_entry, end)
        next_start_time = facts.task_end[chosen_entry - 1]
