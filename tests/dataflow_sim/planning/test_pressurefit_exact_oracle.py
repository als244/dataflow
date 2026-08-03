"""Independent exhaustive-oracle checks for tiny PressureFit chains.

Tests:
- test_exact_oracle_exposes_pressurefit_approximation_gap: exhaustive annotation enumeration proves the retained three-task canary has a 4-us optimum while PressureFit returns its documented 5-us bounded-heuristic plan.
"""
from __future__ import annotations

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.engine.simulator import run
from dataflow_sim.policies.pressurefit import plan_pressurefit_policy
from tools.bench.pressurefit_exact_oracle import find_exact_plan


def _final_fast_gap() -> TaskChain:
    return TaskChain(
        initial_memory=[
            Object("retained", 61, location="fast"),
            Object("later", 61, location="fast"),
        ],
        tasks=[
            Task("task0", inputs=["retained"], outputs=[], runtime=1),
            Task(
                "task1",
                inputs=[],
                outputs=[OutputAlloc("temporary", 61)],
                runtime=1,
            ),
            Task("task2", inputs=["later"], outputs=[], runtime=1),
        ],
        final_locations={"retained": "fast"},
        fast_memory_capacity=122,
        backing_memory_capacity=1_000,
        bandwidth_from_slow=100,
        bandwidth_to_slow=100,
    )


def test_exact_oracle_exposes_pressurefit_approximation_gap():
    bare = _final_fast_gap()
    exact = find_exact_plan(bare, max_assignments=300_000)
    planned, diagnostics = plan_pressurefit_policy(bare, preplace="task0")

    assert exact.assignment_count == 262_144
    assert exact.valid_plan_count == 80
    assert exact.best_makespan_us == 4
    assert exact.peak_fast_memory_bytes == 122
    assert [item.obj_id for item in exact.best_chain.tasks[0].offload_after] == [
        "retained"
    ]
    assert [item.obj_id for item in exact.best_chain.tasks[1].prefetch_after] == [
        "retained"
    ]

    # The oracle is diagnostic, not a hidden replacement policy. Preserve the
    # current selected chain during the planning-time-only optimization pass.
    assert diagnostics.selected_makespan_us == 5
    assert max(
        interval.end for interval in run(planned, snapshots=False).task_intervals
    ) == 5
