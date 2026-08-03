"""Translate PressureFit residency spans into legal executable transitions.

This is the single source of truth shared by reduction, inbound scheduling,
and trigger emission.  In particular, an initial-only span has a real
departure after task 0 even though it contains no producer or use event, and a
later span cannot be prefetched until a preceding departure has fired.
"""
from __future__ import annotations

from bisect import bisect_left

from dataflow_sim.policies.pressurefit_aux.core import _Facts, _fire_task_for_interval
from dataflow_sim.policies.pressurefit_aux.types import (
    _BoundTransition,
    _IntervalSet,
    _PrefetchWindow,
    _ResidencySpan,
    _ResidencyTransition,
    _TransitionPlan,
)


class _InvalidResidencyTransition(ValueError):
    """A span set has no legal TaskChain trigger realization."""


def _has_event_in_span(events: tuple[int, ...], span: _ResidencySpan) -> bool:
    """Return whether task event ``e`` has anchor ``e - 1`` in ``span``."""
    if not events:
        return False
    lo = bisect_left(events, span.start + 1)
    return lo < len(events) and events[lo] <= span.end + 1


def _first_use(oid: str, span: _ResidencySpan, facts: _Facts) -> int | None:
    uses = facts.uses.get(oid, ())
    pos = bisect_left(uses, span.start + 1)
    if pos < len(uses) and uses[pos] <= span.end + 1:
        return uses[pos]
    return None


def _build_object_transitions(
    oid: str,
    spans: list[_ResidencySpan] | list[tuple[int, int]],
    facts: _Facts,
    *,
    coalesce_clean_gaps: bool = False,
) -> list[_ResidencyTransition]:
    """Build a complete legal transition plan for one object's spans."""
    normalized = sorted(_ResidencySpan(*span) for span in spans)
    if not normalized:
        return []

    transitions: list[_ResidencyTransition] = []
    backing_current = oid in facts.backing_ids
    producer = facts.producer.get(oid, -1)
    previous_departure: int | None = None
    previous_departure_was_release = False

    for index, span in enumerate(normalized):
        is_last = index == len(normalized) - 1
        final_location = facts.final_locations.get(oid)

        if index == 0 and span.start == -1:
            arrival = (
                "preplaced"
                if oid in facts.backing_ids and oid not in facts.compute_ids
                else "initial"
            )
            prefetch = None
        elif index == 0 and producer >= 0 and span.start == producer:
            arrival = "produced"
            prefetch = None
            # A newly produced value invalidates any stale copy with the same
            # logical object identifier.
            backing_current = False
        else:
            arrival = "prefetch"
            first_use = _first_use(oid, span, facts)
            earliest = 0 if previous_departure is None else previous_departure + 1
            if coalesce_clean_gaps and previous_departure_was_release:
                earliest = previous_departure
            if first_use is not None:
                latest = first_use - 1
            elif is_last and final_location == "fast":
                latest = max(0, span.start)
            else:
                latest = max(0, min(facts.n - 1, span.start))
            if facts.n == 0 or earliest > latest:
                raise _InvalidResidencyTransition(
                    f"object {oid!r} span {span} has no legal prefetch window "
                    f"after task {earliest - 1} and before task {latest + 1}"
                )
            prefetch = _PrefetchWindow(
                earliest_after_task=earliest,
                latest_after_task=latest,
                first_use_task=first_use,
                entry_boundary=span.start,
            )

        mutated = _has_event_in_span(facts.mutators.get(oid, ()), span)
        if mutated:
            backing_current = False

        departure = _fire_task_for_interval(
            oid, span.start, span.end, facts,
        )
        if is_last and final_location == "fast":
            action = "retain"
            departure = None
        elif departure is None:
            raise _InvalidResidencyTransition(
                f"object {oid!r} span {span} has no legal departure task"
            )
        elif (not is_last) or final_location == "backing":
            action = "release" if backing_current else "offload"
        else:
            action = "release"

        transitions.append(
            _ResidencyTransition(
                span=span,
                arrival=arrival,
                prefetch=prefetch,
                departure_after_task=departure,
                departure_action=action,
                mutated=mutated,
            )
        )
        if action == "offload":
            backing_current = True
        previous_departure = departure
        previous_departure_was_release = action == "release"

    return transitions


def _split_transition_cost(
    oid: str,
    spans: list[_ResidencySpan],
    interval_index: int,
    left_end: int | None,
    right_start: int | None,
    facts: _Facts,
    *,
    backing_current_before: bool | None = None,
    previous_departure_before: int | None = None,
) -> int | None:
    """Return transfer-stream cost when a proposed split is executable.

    The reducer calls this in its hottest candidate-ranking loop.  It performs
    the same backing-validity and trigger-window checks as
    :func:`_build_object_transitions`, but only walks the prefix needed to
    evaluate the new gap and allocates no temporary transition plan.
    """
    original = _ResidencySpan(*spans[interval_index])
    producer = facts.producer.get(oid, -1)
    if backing_current_before is None:
        state = _backing_state_before_spans(oid, spans, facts)
        backing_current, previous_departure = state[interval_index]
    else:
        backing_current = backing_current_before
        previous_departure = previous_departure_before

    stream_cost = 0
    if left_end is not None:
        left = _ResidencySpan(original.start, left_end)
        if interval_index == 0 and producer >= 0 and left.start == producer:
            backing_current = False
        if _has_event_in_span(facts.mutators.get(oid, ()), left):
            backing_current = False
        departure = _fire_task_for_interval(
            oid, left.start, left.end, facts,
        )
        if departure is None:
            return None
        stream_cost = 0 if backing_current else 1
        backing_current = True
        previous_departure = departure
    elif not backing_current:
        # Removing the leading piece is only possible when a recoverable
        # backing copy already exists; there is no pre-task offload trigger.
        return None

    if right_start is None:
        return stream_cost

    right = _ResidencySpan(right_start, original.end)
    first_use = _first_use(oid, right, facts)
    earliest = 0 if previous_departure is None else previous_departure + 1
    is_final_span = interval_index == len(spans) - 1
    if first_use is not None:
        latest = first_use - 1
    elif is_final_span and facts.final_locations.get(oid) == "fast":
        latest = max(0, right.start)
    else:
        latest = max(0, min(facts.n - 1, right.start))
    if facts.n == 0 or earliest > latest:
        return None
    return stream_cost


def _backing_state_before_spans(
    oid: str,
    spans: list[_ResidencySpan],
    facts: _Facts,
) -> list[tuple[bool, int | None]]:
    """Return ``(backing current, prior departure)`` before each span."""
    state: list[tuple[bool, int | None]] = []
    backing_current = oid in facts.backing_ids
    previous_departure: int | None = None
    producer = facts.producer.get(oid, -1)
    last_index = len(spans) - 1
    for index, raw_span in enumerate(spans):
        span = _ResidencySpan(*raw_span)
        if index == 0 and producer >= 0 and span.start == producer:
            backing_current = False
        state.append((backing_current, previous_departure))
        if _has_event_in_span(facts.mutators.get(oid, ()), span):
            backing_current = False
        departure = _fire_task_for_interval(
            oid, span.start, span.end, facts,
        )
        if index < last_index and not backing_current:
            backing_current = True
        previous_departure = departure
    return state


def _build_transitions(
    intervals: _IntervalSet,
    facts: _Facts,
    *,
    coalesce_clean_gaps: bool = False,
) -> _TransitionPlan:
    """Build deterministic executable transitions for the whole plan."""
    preplaced: list[str] = []
    prefetches: list[_BoundTransition] = []
    departures: list[_BoundTransition] = []
    for oid in sorted(intervals):
        object_transitions = _build_object_transitions(
            oid,
            intervals[oid],
            facts,
            coalesce_clean_gaps=coalesce_clean_gaps,
        )
        for transition in object_transitions:
            bound = _BoundTransition(oid, transition)
            if transition.arrival == "preplaced":
                preplaced.append(oid)
            elif transition.arrival == "prefetch":
                prefetches.append(bound)
            if transition.departure_action != "retain":
                departures.append(bound)
    return _TransitionPlan(
        preplaced=tuple(preplaced),
        prefetches=tuple(prefetches),
        departures=tuple(departures),
    )
