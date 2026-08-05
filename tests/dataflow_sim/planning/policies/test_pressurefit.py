"""PressureFit policy: initial placement, PrefetchRule candidates, and triggers.

Tests:
- test_pressure_initial_placement_skips_hidden_future_use: pressure initial placement pre-places a task-0 input but not an object whose first inbound hides behind an intervening task.
- test_pressurefit_prefetches_late_object_instead_of_preplacing: the applied policy keeps the early object compute-resident and reaches the late object via a prefetch, not pre-placement.
- test_pressurefit_runs_training_chain_at_moderate_cap: a 5-layer training chain annotated at a moderate cap runs with the first forward and first backward tasks on the compute track.
- test_pressurefit_keeps_capacity_tight_candidate_with_equal_span_geometry: residency geometry is not incorrectly deduplicated across differing pressure/repair semantics.
- test_clean_same_task_restore_is_an_explicit_coalesced_candidate: only a clean release plus immediate restore is normalized to continuous residency.
- test_pressurefit_diagnostics_describe_selected_candidate: the plan diagnostics report exactly one selected valid candidate whose makespan equals the simulated makespan.
- test_pressurefit_evaluates_all_prefetch_rules_when_lower_bound_is_unmet: when no candidate reaches the compute lower bound, every residency strategy is raced across all eight named PrefetchRule variants and the selected makespan is the valid minimum.
- test_pressurefit_can_restrict_prefetch_rule_portfolio: an exact PrefetchRule subset evaluates only that rule for every residency strategy.
- test_pressurefit_stops_after_verified_compute_lower_bound: a simulator-verified compute-only makespan stops the remaining candidate race without changing the optimum.
- test_pressurefit_can_release_disposable_mutation_after_final_use: a disposable mutated buffer with no final-location constraint is released, not offloaded, after its last use.
- test_pressurefit_preserves_final_backing_mutation_writeback: a mutated buffer pinned to backing is offloaded (written back), not released, after its last use.
- test_pressurefit_uses_timing_relief_when_static_boundary_is_impossible: when no static boundary fits, the policy offloads then prefetches the oversized activation to relieve pressure by timing.
- test_packed_fifo_clamps_prefetch_fire_to_pressure: deadline packing that would fire a prefetch into full boundaries is clamped to a later task where the destination bytes fit.
- test_pressurefit_extends_prefetch_intervals_under_strict_cap: under a strict cap, inbound lead-time extension widens each object's residency interval leftward.
- test_preplace_task0_limits_initial_fast_to_task0_inputs: preplace=task0 restricts initial fast memory to task 0's inputs and brings the rest in as prefetches, unlike greedy.
- test_pressurefit_models_final_fast_and_initial_only_departure: terminal liveness and an initial-only split produce legal offload/prefetch triggers at the exact feasible cap.
- test_pressurefit_places_backing_only_object_for_terminal_fast_state: an unused backing object required fast at exit is prefetched and retained.
- test_pressurefit_retains_unused_produced_terminal_fast_object: an unused produced object required fast at exit is retained without release or offload.
- test_pressurefit_reuses_clean_backing_copy_after_first_writeback: a produced object is written back once and later clean departures use release.
- test_pressurefit_rewrites_backing_copy_after_mutation: mutating a re-prefetched object invalidates backing and forces another writeback.
- test_pressurefit_plans_objects_after_global_workspace_reserve: PressureFit
  subtracts the maximum task workspace once, plans only object residency, and
  restores the original task workspace metadata for final simulation.
- test_pressurefit_fails_fast_on_policy_independent_task_footprint: planning
  identifies every task whose object-only lower bound exceeds the reduced
  object capacity before running any residency heuristic.
- test_pressurefit_rejects_workspace_and_leeway_larger_than_budget: the
  object-only projection rejects when maximum workspace plus fixed leeway
  consumes the complete program budget.
- test_preplace_rejects_unknown_mode: an unknown preplace mode raises ValueError.
"""
from __future__ import annotations

import pytest

from dataflow_sim.policies._common import _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx
from dataflow_sim.policies.pressurefit import (
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.policies.pressurefit_aux.candidate import _realize_plan
from dataflow_sim.policies.pressurefit_aux.core import _build_facts
from dataflow_sim.policies.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.policies.pressurefit_aux.residency_refinement import (
    _extend_inbound_lead_time,
)
from dataflow_sim.policies.pressurefit_aux.seeds import (
    _initial_residency,
    _pressure_initial_placement,
)
from dataflow_sim.policies.pressurefit_aux.types import (
    _PrefetchRuleSpec,
    _ResidencySpan,
    _ResidencySpec,
)
from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.engine.simulator import run
from chain_fixtures import build_bare_training_chain


_TEST_RESIDENCY = _ResidencySpec("test", True, "min-stall")


def _realize_for_test(
    bare: TaskChain,
    facts,
    intervals,
    rule="latest-safe",
    *,
    coalesce_clean_gaps=False,
) -> TaskChain:
    return _realize_plan(
        bare,
        facts,
        intervals,
        _TEST_RESIDENCY,
        _PrefetchRuleSpec("test", rule, coalesce_clean_gaps),
        [0] * (facts.n + 1),
    )


def _initial_choice_chain() -> TaskChain:
    return TaskChain(
        initial_memory=[
            Object(id="early", size=10, location="backing", type="weight"),
            Object(id="late", size=10, location="backing", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=["early"], outputs=[], runtime=50),
            Task(id="t1", inputs=[], outputs=[], runtime=100),
            Task(id="t2", inputs=["late"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=15,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )


def test_pressure_initial_placement_skips_hidden_future_use():
    bare = _initial_choice_chain()
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)

    placement = _pressure_initial_placement(
        bare, bare.fast_memory_capacity, sizes, uses_by_task,
    )

    assert "early" in placement  # task-0 input, no prior trigger slot
    assert "late" not in placement  # its first inbound hides behind t1


def test_pressurefit_prefetches_late_object_instead_of_preplacing():
    bare = _initial_choice_chain()
    annotated = apply_pressurefit_policy(bare)
    run(annotated)

    initial_compute = {o.id for o in annotated.initial_memory if o.location == "fast"}
    assert "early" in initial_compute
    assert "late" not in initial_compute
    assert any(
        trig.obj_id == "late"
        for task in annotated.tasks
        for trig in task.prefetch_after
    )


def test_pressurefit_plans_objects_after_global_workspace_reserve():
    bare = TaskChain(
        initial_memory=[
            Object(id="early", size=4, location="backing"),
            Object(id="late", size=5, location="backing"),
        ],
        tasks=[
            Task(
                id="scratch-heavy",
                inputs=["early"],
                outputs=[],
                runtime=10,
                workspace_bytes=3,
            ),
            Task(
                id="object-heavy",
                inputs=["late"],
                outputs=[],
                runtime=1,
                workspace_bytes=1,
            ),
        ],
        fast_memory_capacity=9,
        backing_memory_capacity=20,
        bandwidth_from_slow=5,
        bandwidth_to_slow=5,
    )

    annotated, diagnostics = plan_pressurefit_policy(
        bare,
        preplace="task0",
        program_leeway_bytes=1,
    )
    log = run(annotated, snapshots=False)

    assert diagnostics.program_memory_capacity == 9
    assert diagnostics.fast_memory_capacity == 5
    assert diagnostics.max_task_workspace_bytes == 3
    assert diagnostics.program_leeway_bytes == 1
    assert annotated.fast_memory_capacity == 8
    assert [task.workspace_bytes for task in annotated.tasks] == [3, 1]
    assert log.peak_fast_memory_bytes == 7
    assert {obj.id for obj in annotated.initial_memory if obj.location == "fast"} == {
        "early"
    }
    assert any(
        trigger.obj_id == "late"
        for task in annotated.tasks
        for trigger in task.prefetch_after
    )


def test_pressurefit_rejects_workspace_and_leeway_larger_than_budget():
    bare = TaskChain(
        initial_memory=[],
        tasks=[Task(id="task", inputs=[], outputs=[], runtime=1, workspace_bytes=8)],
        fast_memory_capacity=10,
    )

    with pytest.raises(ValueError, match="maximum task workspace plus program leeway"):
        apply_pressurefit_policy(bare, program_leeway_bytes=3)


def test_pressurefit_fails_fast_on_policy_independent_task_footprint():
    bare = TaskChain(
        initial_memory=[
            Object(id="state", size=6, location="backing"),
            Object(id="other", size=1, location="backing"),
        ],
        tasks=[
            Task(id="feasible", inputs=["other"], outputs=[], runtime=1),
            Task(
                id="impossible-update",
                inputs=["state"],
                mutates_inputs=["state"],
                outputs=[OutputAlloc(id="fresh", size=3)],
                runtime=1,
                workspace_bytes=2,
            ),
            Task(
                id="second-impossible",
                inputs=["state"],
                outputs=[OutputAlloc(id="fresh-again", size=4)],
                runtime=1,
                workspace_bytes=3,
            ),
        ],
        fast_memory_capacity=10,
        backing_memory_capacity=20,
        bandwidth_from_slow=5,
        bandwidth_to_slow=5,
    )

    with pytest.raises(ValueError) as caught:
        apply_pressurefit_policy(bare, preplace="task0")
    message = str(caught.value)
    assert "2 task(s) cannot satisfy fast memory need" in message
    assert (
        "task 'impossible-update' requires 6 bytes of inputs + 3 bytes of "
        "compute outputs + 0 bytes of workspace = 9 > fast_memory_capacity=7"
        in message
    )
    assert (
        "task 'second-impossible' requires 6 bytes of inputs + 4 bytes of "
        "compute outputs + 0 bytes of workspace = 10 > fast_memory_capacity=7"
        in message
    )


def test_pressurefit_runs_training_chain_at_moderate_cap():
    bare = build_bare_training_chain(L=5)
    annotated = apply_pressurefit_policy(bare, fast_memory_capacity=800)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "f_0" in compute_ids
    assert "b_0" in compute_ids


def test_pressurefit_keeps_capacity_tight_candidate_with_equal_span_geometry():
    bare = build_bare_training_chain(L=2)
    annotated, diagnostics = plan_pressurefit_policy(
        bare,
        fast_memory_capacity=224,
    )
    log = run(annotated, snapshots=False)

    assert diagnostics.selected_candidate.startswith("tight-")
    assert diagnostics.selected_makespan_us == 114
    assert log.peak_fast_memory_bytes == 224


def test_clean_same_task_restore_is_an_explicit_coalesced_candidate():
    bare = TaskChain(
        initial_memory=[Object(id="state", size=10, location="backing")],
        tasks=[
            Task(id="use-early", inputs=["state"], outputs=[], runtime=10),
            Task(id="gap", inputs=[], outputs=[], runtime=10),
            Task(id="use-late", inputs=["state"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=20,
        backing_memory_capacity=20,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )
    facts = _build_facts(bare)
    intervals = {
        "state": [
            _ResidencySpan(-1, -1),
            _ResidencySpan(1, 1),
        ],
    }

    ordinary = _realize_for_test(
        bare,
        facts,
        intervals,
        rule="latest-safe",
    )
    coalesced = _realize_for_test(
        bare,
        facts,
        intervals,
        rule="latest-safe",
        coalesce_clean_gaps=True,
    )

    run(ordinary, snapshots=False)
    run(coalesced, snapshots=False)
    assert ordinary.tasks[0].releases_after == ["state"]
    assert ordinary.tasks[1].prefetch_after[0].obj_id == "state"
    assert "state" not in coalesced.tasks[0].releases_after
    assert all(
        trigger.obj_id != "state"
        for task in coalesced.tasks
        for trigger in task.prefetch_after
    )
def test_pressurefit_diagnostics_describe_selected_candidate():
    bare = build_bare_training_chain(L=5)
    annotated, diagnostics = plan_pressurefit_policy(bare, fast_memory_capacity=800)
    log = run(annotated)
    makespan = max(iv.end for iv in log.task_intervals)

    assert diagnostics.selected_makespan_us == makespan
    assert diagnostics.valid_candidate_count > 0
    selected = [c for c in diagnostics.candidates if c.selected]
    assert len(selected) == 1
    assert selected[0].name == diagnostics.selected_candidate
    assert selected[0].status == "valid"


def test_pressurefit_evaluates_all_prefetch_rules_when_lower_bound_is_unmet():
    bare = build_bare_training_chain(L=10)
    _annotated, diagnostics = plan_pressurefit_policy(bare, fast_memory_capacity=500)

    expected_prefetch_rules = {
        "packed-fifo",
        "packed-fit",
        "interval-entry",
        "latest-safe",
        "packed-fifo-coalesced",
        "packed-fit-coalesced",
        "interval-entry-coalesced",
        "latest-safe-coalesced",
    }
    prefetch_rules_by_residency: dict[str, set[str]] = {}
    for candidate in diagnostics.candidates:
        residency, prefetch_rule = candidate.name.split("/", 1)
        prefetch_rules_by_residency.setdefault(residency, set()).add(prefetch_rule)
    assert set(prefetch_rules_by_residency) == {
        "headroom-stall",
        "headroom-transfer",
        "tight-stall",
        "tight-transfer",
        "relaxed-stall",
    }
    assert all(
        prefetch_rules == expected_prefetch_rules
        for prefetch_rules in prefetch_rules_by_residency.values()
    )
    assert diagnostics.candidate_count == 40
    # The selected plan is the fastest valid one.
    valid = [c for c in diagnostics.candidates if c.status == "valid"]
    assert diagnostics.selected_makespan_us == min(c.makespan_us for c in valid)


def test_pressurefit_can_restrict_prefetch_rule_portfolio():
    bare = build_bare_training_chain(L=10)
    _annotated, diagnostics = plan_pressurefit_policy(
        bare,
        fast_memory_capacity=500,
        prefetch_rules=("latest-safe",),
    )

    assert diagnostics.candidate_count == 5
    assert all(
        candidate.name.endswith("/latest-safe")
        for candidate in diagnostics.candidates
    )


def test_pressurefit_stops_after_verified_compute_lower_bound():
    bare = build_bare_training_chain(L=5)
    annotated, diagnostics = plan_pressurefit_policy(
        bare,
        fast_memory_capacity=800,
    )

    assert diagnostics.selected_makespan_us == sum(
        task.runtime for task in bare.tasks
    )
    assert diagnostics.selected_candidate == "headroom-stall/packed-fifo"
    assert diagnostics.candidate_count == 1
    assert diagnostics.valid_candidate_count == 1
    assert run(annotated, snapshots=False).peak_fast_memory_bytes <= 800


def test_pressurefit_can_release_disposable_mutation_after_final_use():
    bare = TaskChain(
        initial_memory=[
            Object(id="buf", size=10, location="backing", type="other"),
        ],
        tasks=[
            Task(
                id="mut",
                inputs=["buf"],
                outputs=[OutputAlloc(id="out", size=1, location="fast")],
                runtime=1,
                mutates_inputs=["buf"],
            ),
        ],
        fast_memory_capacity=32,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert annotated.tasks[0].mutates_inputs == ["buf"]
    assert "buf" in annotated.tasks[0].releases_after
    assert not any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_preserves_final_backing_mutation_writeback():
    bare = TaskChain(
        initial_memory=[
            Object(id="buf", size=10, location="backing", type="other"),
        ],
        tasks=[
            Task(
                id="mut",
                inputs=["buf"],
                outputs=[OutputAlloc(id="out", size=1, location="fast")],
                runtime=1,
                mutates_inputs=["buf"],
            ),
        ],
        final_locations={"buf": "backing"},
        fast_memory_capacity=32,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert annotated.tasks[0].mutates_inputs == ["buf"]
    assert "buf" not in annotated.tasks[0].releases_after
    assert any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_uses_timing_relief_when_static_boundary_is_impossible():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=1, location="fast", type="activation"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["x"],
                outputs=[
                    OutputAlloc(id="A", size=60, location="fast"),
                    OutputAlloc(id="y", size=1, location="fast"),
                ],
                runtime=10,
            ),
            Task(
                id="t1",
                inputs=["y"],
                outputs=[OutputAlloc(id="B", size=60, location="fast")],
                runtime=10,
            ),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["A"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=100,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert any(trig.obj_id == "A" for trig in annotated.tasks[0].offload_after)
    assert any(
        trig.obj_id == "A"
        for task in annotated.tasks[1:3]
        for trig in task.prefetch_after
    )
    run(annotated)


def test_packed_fifo_clamps_prefetch_fire_to_pressure():
    """Deadline packing must not fire a prefetch into boundaries whose
    modeled bytes leave no room for the transfer's destination."""
    bare = TaskChain(
        initial_memory=[
            # `hog` pins boundaries -1..1 (anchors at every gap), leaving no
            # room for x's 30 bytes before boundary 2.
            Object(id="hog", size=90, location="fast", type="other"),
            Object(id="x", size=30, location="backing", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t1", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t2", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t3", inputs=[], outputs=[], runtime=10),
            Task(id="t4", inputs=["x"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )
    facts = _build_facts(bare)
    intervals = _initial_residency(facts, initial_compute=set())
    _reduce_to_fit(facts, intervals, bare.fast_memory_capacity)
    assert intervals["x"] == [(3, 3)]

    unclamped = _realize_for_test(
        bare,
        facts,
        intervals,
        rule="packed-fifo",
    )
    assert "x" in [t.obj_id for t in unclamped.tasks[0].prefetch_after]

    annotated = _realize_for_test(bare, facts, intervals, rule="packed-fit")
    prefetch_by_task = {
        task.id: [trig.obj_id for trig in task.prefetch_after]
        for task in annotated.tasks
    }
    # Deadline packing alone fires x on t0 (tau=30, deadline=40), but
    # boundaries 0..1 hold 90/100 bytes; the clamp slides the trigger to t2.
    assert prefetch_by_task["t0"] == []
    assert prefetch_by_task["t1"] == []
    assert "x" in prefetch_by_task["t2"]
    run(annotated)


def test_pressurefit_extends_prefetch_intervals_under_strict_cap():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=25, location="backing", type="other"),
            Object(id="y", size=25, location="backing", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=[], outputs=[], runtime=10),
            Task(id="t1", inputs=[], outputs=[], runtime=10),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["x"], outputs=[], runtime=10),
            Task(id="t4", inputs=["y"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=50,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )
    facts = _build_facts(bare)
    intervals = _initial_residency(facts, initial_compute=set())
    _reduce_to_fit(facts, intervals, bare.fast_memory_capacity)

    assert intervals["x"] == [(2, 2)]
    assert intervals["y"] == [(3, 3)]

    _extend_inbound_lead_time(
        facts, intervals, bare.fast_memory_capacity, bare.bandwidth_from_slow,
    )

    assert intervals["x"] == [(0, 2)]
    assert intervals["y"] == [(1, 3)]


def test_pressurefit_models_final_fast_and_initial_only_departure():
    bare = TaskChain(
        initial_memory=[
            Object(id="retained", size=61, location="fast"),
            Object(id="later", size=61, location="fast"),
        ],
        tasks=[
            Task(id="task0", inputs=["retained"], outputs=[], runtime=1),
            Task(
                id="task1",
                inputs=[],
                outputs=[OutputAlloc(id="temporary", size=61, location="fast")],
                runtime=1,
            ),
            Task(id="task2", inputs=["later"], outputs=[], runtime=1),
        ],
        final_locations={"retained": "fast"},
        fast_memory_capacity=122,
        backing_memory_capacity=1_000,
        bandwidth_from_slow=100,
        bandwidth_to_slow=100,
    )

    annotated, diagnostics = plan_pressurefit_policy(bare, preplace="task0")
    log = run(annotated, snapshots=False)

    assert log.peak_fast_memory_bytes == 122
    assert diagnostics.valid_candidate_count >= 4
    assert all(candidate.status == "valid" for candidate in diagnostics.candidates)
    assert any(
        trigger.obj_id == "later"
        for trigger in annotated.tasks[0].offload_after
    )
    assert any(
        trigger.obj_id == "later"
        for trigger in annotated.tasks[1].prefetch_after
    )
    assert not any(
        trigger.obj_id == "retained"
        for task in annotated.tasks
        for trigger in task.offload_after
    )
    assert "retained" not in annotated.tasks[-1].releases_after

    with pytest.raises(ValueError, match="initial_memory|cannot reduce|infeasible"):
        apply_pressurefit_policy(bare, fast_memory_capacity=121, preplace="task0")


def test_pressurefit_places_backing_only_object_for_terminal_fast_state():
    bare = TaskChain(
        initial_memory=[Object(id="state", size=10, location="backing")],
        tasks=[
            Task(id="task0", inputs=[], outputs=[], runtime=1),
            Task(id="task1", inputs=[], outputs=[], runtime=1),
        ],
        final_locations={"state": "fast"},
        fast_memory_capacity=10,
        backing_memory_capacity=20,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare, preplace="task0")
    log = run(annotated, snapshots=False)

    assert log.peak_fast_memory_bytes == 10
    assert any(
        trigger.obj_id == "state"
        for task in annotated.tasks
        for trigger in task.prefetch_after
    )
    assert all("state" not in task.releases_after for task in annotated.tasks)


def test_pressurefit_retains_unused_produced_terminal_fast_object():
    bare = TaskChain(
        initial_memory=[],
        tasks=[
            Task(
                id="produce",
                inputs=[],
                outputs=[OutputAlloc(id="result", size=10, location="fast")],
                runtime=1,
            ),
            Task(id="tail", inputs=[], outputs=[], runtime=1),
        ],
        final_locations={"result": "fast"},
        fast_memory_capacity=10,
    )

    annotated = apply_pressurefit_policy(bare)
    run(annotated, snapshots=False)

    assert all("result" not in task.releases_after for task in annotated.tasks)
    assert all(
        trigger.obj_id != "result"
        for task in annotated.tasks
        for trigger in task.offload_after
    )


def _repeated_produced_object_chain(*, mutate_middle_use: bool) -> TaskChain:
    return TaskChain(
        initial_memory=[],
        tasks=[
            Task(
                id="produce",
                inputs=[],
                outputs=[OutputAlloc(id="x", size=10, location="fast")],
                runtime=1,
            ),
            Task(
                id="temporary-1",
                inputs=[],
                outputs=[OutputAlloc(id="tmp1", size=10, location="fast")],
                runtime=1,
            ),
            Task(id="idle-1", inputs=[], outputs=[], runtime=1),
            Task(
                id="use-1",
                inputs=["x"],
                outputs=[],
                runtime=1,
                mutates_inputs=["x"] if mutate_middle_use else [],
            ),
            Task(
                id="temporary-2",
                inputs=[],
                outputs=[OutputAlloc(id="tmp2", size=10, location="fast")],
                runtime=1,
            ),
            Task(id="idle-2", inputs=[], outputs=[], runtime=1),
            Task(id="use-2", inputs=["x"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=10,
        backing_memory_capacity=100,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )


def test_pressurefit_reuses_clean_backing_copy_after_first_writeback():
    annotated = apply_pressurefit_policy(
        _repeated_produced_object_chain(mutate_middle_use=False),
    )
    run(annotated, snapshots=False)

    x_offloads = [
        task.id
        for task in annotated.tasks
        if any(trigger.obj_id == "x" for trigger in task.offload_after)
    ]
    assert x_offloads == ["produce"]
    assert "x" in annotated.tasks[3].releases_after


def test_pressurefit_rewrites_backing_copy_after_mutation():
    annotated = apply_pressurefit_policy(
        _repeated_produced_object_chain(mutate_middle_use=True),
    )
    run(annotated, snapshots=False)

    x_offloads = [
        task.id
        for task in annotated.tasks
        if any(trigger.obj_id == "x" for trigger in task.offload_after)
    ]
    assert x_offloads == ["produce", "use-1"]


def _spacious_chain() -> TaskChain:
    # capacity fits everything: greedy pre-places both weights, task0 only w0
    return TaskChain(
        initial_memory=[
            Object(id="w0", size=10, location="backing", type="weight"),
            Object(id="w1", size=10, location="backing", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=["w0"], outputs=[], runtime=50),
            Task(id="t1", inputs=["w1"], outputs=[], runtime=50),
        ],
        fast_memory_capacity=100,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )


def test_preplace_task0_limits_initial_fast_to_task0_inputs():
    bare = _spacious_chain()
    greedy = apply_pressurefit_policy(bare)
    task0 = apply_pressurefit_policy(bare, preplace="task0")
    run(greedy)
    run(task0)

    greedy_fast = {o.id for o in greedy.initial_memory if o.location == "fast"}
    task0_fast = {o.id for o in task0.initial_memory if o.location == "fast"}
    assert "w1" in greedy_fast  # spare capacity -> greedy pre-places it
    assert task0_fast == {"w0"}  # honest mode: only task 0's needs
    # w1 still arrives, but as a planned (charged, overlappable) prefetch
    assert any(
        trig.obj_id == "w1"
        for task in task0.tasks
        for trig in task.prefetch_after
    )


def test_preplace_rejects_unknown_mode():
    with pytest.raises(ValueError, match="preplace"):
        apply_pressurefit_policy(_spacious_chain(), preplace="everything")
