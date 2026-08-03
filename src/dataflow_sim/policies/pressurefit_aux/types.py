"""Shared PressureFit types.

The public policy still consumes and returns :class:`TaskChain`.  These small
private records make the planner's boundary and transition semantics explicit
without exposing a second public planning API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple


_CutScoreKind = Literal["min-stall", "min-transfer"]
_PrefetchRuleKind = Literal[
    "packed-fifo",
    "packed-fit",
    "interval-entry",
    "latest-safe",
]


@dataclass(frozen=True, slots=True)
class _ResidencySpec:
    """One pressure view, CutScore choice, and stopping rule."""

    name: str
    prefetch_headroom: bool
    cut_score: _CutScoreKind
    continue_headroom_cuts: bool = True


@dataclass(frozen=True, slots=True)
class _PrefetchRuleSpec:
    """One exact rule name plus its boundary heuristic and normalization."""

    name: str
    kind: _PrefetchRuleKind
    coalesce_clean_gaps: bool = False


class _ResidencySpan(NamedTuple):
    """Inclusive fast-residency span in PressureFit boundary coordinates.

    Boundaries run from ``-1`` (before task 0) through ``n - 1`` (after the
    final task).  A named tuple deliberately preserves the compact storage and
    tuple operations used by the hot reducer while preventing anonymous
    ``(a, b)`` values from crossing module boundaries undocumented.
    """

    start: int
    end: int


_AnchorKind = Literal["initial", "producer", "use", "final_fast"]


@dataclass(frozen=True, slots=True)
class _ResidencyAnchor:
    """One mandatory point at which an object must be fast-resident."""

    boundary: int
    kind: _AnchorKind
    task: int | None


_ArrivalKind = Literal["initial", "preplaced", "produced", "prefetch"]
_DepartureAction = Literal["retain", "release", "offload"]


@dataclass(frozen=True, slots=True)
class _PrefetchWindow:
    """Legal trigger window for one later fast-residency span."""

    earliest_after_task: int
    latest_after_task: int
    first_use_task: int | None
    entry_boundary: int


@dataclass(frozen=True, slots=True)
class _ResidencyTransition:
    """Executable arrival/departure semantics for one residency span."""

    span: _ResidencySpan
    arrival: _ArrivalKind
    prefetch: _PrefetchWindow | None
    departure_after_task: int | None
    departure_action: _DepartureAction
    mutated: bool


class _BoundTransition(NamedTuple):
    """One object id paired with one of its residency transitions."""

    oid: str
    transition: _ResidencyTransition


@dataclass(frozen=True, slots=True)
class _TransitionPlan:
    """Complete executable transition decisions for one residency plan.

    The categorized representation contains exactly what rule application and
    emission consume. Absence from ``departures`` represents retention.
    """

    preplaced: tuple[str, ...]
    prefetches: tuple[_BoundTransition, ...]
    departures: tuple[_BoundTransition, ...]


_PrefetchAssignments = tuple[tuple[str, ...], ...]


_IntervalSet = dict[str, list[_ResidencySpan]]
