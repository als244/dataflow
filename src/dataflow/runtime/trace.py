"""Execution trace, shape-compatible with the simulator's EventLog.

`intervals` uses the simulator's TaskInterval vocabulary (task_id, start,
end, track) including its transfer naming scheme ("from_slow:obj",
"from_slow:obj#1" for repeats), so parity is a direct multiset comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Interval:
    task_id: str
    start: float
    end: float
    track: str  # "compute" | "from_slow" | "to_slow"


@dataclass(frozen=True)
class TraceEvent:
    t: float
    kind: str  # reserve|live|release|enqueue|deferred|transfer_start|transfer_end|mutate
    object_id: str | None = None
    task_id: str | None = None
    detail: str = ""


@dataclass
class RunTrace:
    intervals: list[Interval] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    memory_trace: list[tuple[float, int]] = field(default_factory=list)  # (t_us, used_fast)
    peak_fast_bytes: int = 0

    def makespan_us(self) -> float:
        return max((iv.end for iv in self.intervals), default=0.0)


@dataclass(frozen=True)
class ParityDiff:
    missing: tuple[Any, ...]      # in sim, not in runtime
    extra: tuple[Any, ...]        # in runtime, not in sim
    peak_sim: int
    peak_runtime: int

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra and self.peak_sim == self.peak_runtime


def compare_to_sim_eventlog(trace: RunTrace, event_log: Any, *, time_tol: float = 0.0) -> ParityDiff:
    """Compare runtime intervals + peak against a sim EventLog.

    With time_tol == 0 the comparison is exact (both sides compute float
    microseconds through the same formulas). A nonzero tolerance buckets
    times to that precision before comparing.
    """

    def norm(t: float) -> float:
        if time_tol <= 0:
            return t
        return round(t / time_tol) * time_tol

    sim = {(iv.task_id, iv.track, norm(iv.start), norm(iv.end)) for iv in event_log.task_intervals}
    ours = {(iv.task_id, iv.track, norm(iv.start), norm(iv.end)) for iv in trace.intervals}
    return ParityDiff(
        missing=tuple(sorted(sim - ours)),
        extra=tuple(sorted(ours - sim)),
        peak_sim=event_log.peak_fast_memory_bytes,
        peak_runtime=trace.peak_fast_bytes,
    )
