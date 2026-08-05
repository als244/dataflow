"""Diagnostics data structures for PressureFit."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PressureFitCandidateDiagnostic:
    """Observable outcome of one internal planner candidate."""
    name: str
    status: str
    selected: bool
    makespan_us: float | None
    wall_time_s: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PressureFitDiagnostics:
    planning_time_s: float
    task_count: int
    object_count: int
    program_memory_capacity: int | None
    fast_memory_capacity: int | None
    max_task_workspace_bytes: int
    program_leeway_bytes: int
    candidate_count: int
    valid_candidate_count: int
    selected_candidate: str
    selected_makespan_us: float
    candidates: list[PressureFitCandidateDiagnostic]


def _candidate_diagnostic(
    name: str,
    *,
    status: str,
    wall_time_s: float = 0.0,
    makespan_us: float | None = None,
    error: str | None = None,
) -> PressureFitCandidateDiagnostic:
    return PressureFitCandidateDiagnostic(
        name=name,
        status=status,
        selected=False,
        makespan_us=makespan_us,
        wall_time_s=wall_time_s,
        error=error,
    )
