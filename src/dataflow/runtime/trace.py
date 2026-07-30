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


@dataclass(frozen=True)
class DispatchRecord:
    """One task's dispatch anatomy on the strict-paced dispatcher, host
    microseconds on the run clock (the same clock the intervals use):
    when the previous task retired (pacing), when every input was
    host-observed live, when the fast outputs were reserved, and when
    the executable's enqueue finished. The stall fields name what the
    dispatcher last waited ON whenever a phase took time: input objects
    not yet live, output ids whose placed offsets were busy, and
    whether ledger capacity itself blocked — so a timeline gap before a
    task reads directly as its cause (input dependency vs output
    space vs host turnaround)."""
    task_id: str
    pacing_done: float
    inputs_live: float
    outputs_reserved: float
    launched: float
    stalled_inputs: tuple[str, ...] = ()
    stalled_outputs: tuple[str, ...] = ()
    ledger_blocked: bool = False


@dataclass
class RunTrace:
    intervals: list[Interval] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    memory_trace: list[tuple[float, int]] = field(default_factory=list)  # (t_us, used_fast)
    peak_fast_bytes: int = 0
    dispatch: list[DispatchRecord] = field(default_factory=list)

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


def trace_to_dict(trace: RunTrace) -> dict:
    """Wire form of a RunTrace: measured intervals + memory trace +
    peak, JSON-clean. Events are omitted (debug-volume; the interval
    timeline is what sim comparison and the webapp consume)."""
    return {
        "intervals": [[iv.task_id, iv.track, iv.start, iv.end]
                      for iv in trace.intervals],
        "memory_trace": [[t, used] for t, used in trace.memory_trace],
        "peak_fast_bytes": trace.peak_fast_bytes,
        "makespan_us": trace.makespan_us(),
        "dispatch": [[d.task_id, d.pacing_done, d.inputs_live,
                      d.outputs_reserved, d.launched,
                      list(d.stalled_inputs), list(d.stalled_outputs),
                      d.ledger_blocked]
                     for d in trace.dispatch],
    }
