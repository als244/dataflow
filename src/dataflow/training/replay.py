"""Replay-fidelity: the scheduler-quality metric.

Re-simulate a program with every MEASURED duration (tasks and transfers)
installed as overrides; the gap between the real makespan and that replay's
makespan isolates pure scheduling/dispatch overhead from all cost-model
error. This is the number the engine gates quote.
"""
from __future__ import annotations

from dataclasses import replace

from dataflow.core import Program, TransferDirective
from dataflow.core.convert import to_sim_chain
from dataflow.runtime.trace import RunTrace


def replay_program(program: Program, trace: RunTrace) -> Program:
    """The program with measured durations from `trace` as overrides."""
    measured_compute = {
        iv.task_id: iv.end - iv.start for iv in trace.intervals if iv.track == "compute"
    }
    seen: dict[tuple[str, str], int] = {}
    measured_xfer: dict[tuple[str, str, int], float] = {}
    for iv in sorted(
        (iv for iv in trace.intervals if iv.track != "compute"), key=lambda iv: iv.start
    ):
        obj = iv.task_id.split(":", 1)[1].split("#", 1)[0]
        n = seen.get((iv.track, obj), 0)
        seen[(iv.track, obj)] = n + 1
        measured_xfer[(iv.track, obj, n)] = iv.end - iv.start

    fired: dict[tuple[str, str], int] = {}

    def override(direction: str, object_id: str) -> float:
        n = fired.get((direction, object_id), 0)
        fired[(direction, object_id)] = n + 1
        return measured_xfer[(direction, object_id, n)]

    new_tasks = tuple(
        replace(
            t,
            runtime_us=measured_compute[t.id],
            offload_after=tuple(
                TransferDirective(object_id=x.object_id, runtime_us=override("to_slow", x.object_id))
                for x in t.offload_after
            ),
            prefetch_after=tuple(
                TransferDirective(object_id=x.object_id, runtime_us=override("from_slow", x.object_id))
                for x in t.prefetch_after
            ),
        )
        for t in program.tasks
    )
    return replace(program, tasks=new_tasks)


def replay_gap_pct(program: Program, trace: RunTrace, real_makespan_us: float) -> float:
    from dataflow_sim.engine.simulator import run as sim_run

    log = sim_run(to_sim_chain(replay_program(program, trace)), snapshots=False)
    replay_makespan = max(iv.end for iv in log.task_intervals)
    return (real_makespan_us - replay_makespan) / replay_makespan * 100
