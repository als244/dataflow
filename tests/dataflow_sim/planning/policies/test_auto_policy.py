"""Tests for the belady_reactive auto-policy.

Working envelope:
  L=3: fast_memory_capacity >= 500 or None
  L=5: fast_memory_capacity >= 500 or None
  L=10: fast_memory_capacity >= 500 or None

Tests:
- test_next_use_after_returns_first_use_at_or_after: _next_use_after returns the first use at or after a time, or infinity when none (including unknown objects).
- test_compute_uses_collects_input_timestamps: _compute_uses records each object's input-use timestamps from the ideal starts.
- test_initial_placement_must_place_T1_inputs: initial placement must place the first compute task's backing inputs; the already-resident input is excluded.
- test_initial_placement_raises_when_widest_T1_too_big: initial placement raises widest-task infeasibility when the first task cannot fit under the cap.
- test_auto_policy_L3_works_at_loose_caps: the belady_reactive policy runs an L=3 chain across None and 1200..500 caps with the first forward and backward tasks present.
- test_auto_policy_L5_works_down_to_cap_500: belady_reactive runs an L=5 chain down to cap=500 with the first backward task present.
- test_auto_policy_L10_works_down_to_cap_500: belady_reactive runs an L=10 chain down to cap=500 with the first backward task present.
- test_iterative_refinement_recovers_valid_plan_at_tight_cap: at L=10 cap=600 iterative refinement yields a valid run with makespan under 400.
- test_roundtrip_enumeration_finds_wide_gap_weights: round-trip enumeration finds candidates for the wide-gap forward weights W_0 and W_1.
- test_per_task_use_events_collapse_duplicates: per-task use events collapse duplicate references so W_0 records exactly its three ordered uses.
- test_initial_placement_leaves_slack_for_widest_task: slack-aware initial placement leaves room for the widest single-task footprint under a tight cap.
- test_auto_policy_L3_zero_stalls_at_unlimited: at unlimited cap the L=3 schedule has no gaps between consecutive compute tasks.
- test_auto_policy_L3_unlimited_emits_no_transfers_without_final_locations: at unlimited cap with no final-location constraints the policy emits no prefetches or offloads.
- test_auto_policy_L3_unlimited_honors_final_backing_locations: a final-backing constraint yields exactly one write-back offload per pinned gradient and no prefetches.
- test_auto_policy_emits_releases_at_tight_cap: at a tight-but-feasible cap the policy emits at least one release trigger.
- test_releases_weight_instead_of_offloading_when_backing_copy_exists: weights with a byte-identical backing copy are released, never offloaded, when their compute bytes must be freed.
- test_smart_initial_placement_defers_to_leave_room_for_outputs: smart initial placement defers late-first-use backing objects at a tight cap but places all of them at a loose cap.
- test_smart_initial_placement_at_loose_cap_eliminates_forward_stalls: with enough room, smart initial placement runs all five forward tasks back-to-back with zero stall.
- test_auto_writes_back_final_backing_objects: objects listed in final_locations end live on backing after the run.
- test_activation_offload_fires_eagerly_at_production: a forced activation offload fires within one compute task of its production, not at the deadline.
- test_auto_at_least_as_fast_as_sliding_window_at_unlimited_cap: at unlimited cap the auto-policy makespan is at most the sliding-window makespan.
"""
from dataclasses import replace

import pytest

from dataflow_sim.policies._common import (
    _compute_ideal_starts,
    _compute_uses,
    _next_use_after,
    _object_sizes,
)
from dataflow_sim.policies.roundtrip_planner import _initial_placement
from dataflow_sim.policies.belady_reactive import (
    apply_belady_reactive_policy as apply_auto_policy,
)
from dataflow_sim.policies.sliding_window import apply_sliding_window_policy
from dataflow_sim.engine.simulator import run
from chain_fixtures import build_bare_training_chain


# ---------- next-use / reference-stream helpers ----------

def test_next_use_after_returns_first_use_at_or_after():
    uses = {"a": [0, 10, 20], "b": [5, 15]}
    assert _next_use_after(uses, "a", 0) == 0
    assert _next_use_after(uses, "a", 5) == 10
    assert _next_use_after(uses, "a", 21) == float("inf")
    assert _next_use_after(uses, "b", 5) == 5
    assert _next_use_after(uses, "b", 16) == float("inf")
    assert _next_use_after(uses, "missing", 0) == float("inf")


def test_compute_uses_collects_input_timestamps():
    bare = build_bare_training_chain(L=2)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # input is used by f_0 at t=0
    assert uses["input"] == [0]
    # W_0 used by f_0 (t=0) and by r_0/b_0 (later)
    assert 0 in uses["W_0"]


# ---------- initial placement ----------

def test_initial_placement_must_place_T1_inputs():
    bare = build_bare_training_chain(L=3)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # T_1 = f_0; inputs are input, W_0
    placement = _initial_placement(bare, fast_memory_capacity=2000, uses=uses, sizes=sizes)
    assert "W_0" in placement  # input is already compute-resident, so excluded


def test_initial_placement_raises_when_widest_T1_too_big():
    bare = build_bare_training_chain(L=3, weight_size=64, input_size=16)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # T_1 needs input(16) + W_0(64) on compute + outputs A_0(32) + y_0(32) reserved = 144
    # Initial pool also has input already on compute (16 bytes)
    # If capacity is 30, can't fit even input
    with pytest.raises(ValueError, match="widest-task infeasibility"):
        _initial_placement(bare, fast_memory_capacity=30, uses=uses, sizes=sizes)


# ---------- End-to-end / working envelope ----------

@pytest.mark.parametrize("cap", [None, 1200, 1000, 800, 600, 500])
def test_auto_policy_L3_works_at_loose_caps(cap):
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, fast_memory_capacity=cap)
    log = run(annotated)  # must not raise
    # All compute tasks should appear in intervals
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "f_0" in compute_ids
    assert "b_0" in compute_ids


@pytest.mark.parametrize("cap", [None, 1200, 1000, 800, 600, 500])
def test_auto_policy_L5_works_down_to_cap_500(cap):
    """belady_reactive extends L=5 down to cap=500."""
    bare = build_bare_training_chain(L=5)
    annotated = apply_auto_policy(bare, fast_memory_capacity=cap)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "b_0" in compute_ids


@pytest.mark.parametrize("cap", [None, 1500, 1000, 800, 600, 500])
def test_auto_policy_L10_works_down_to_cap_500(cap):
    """belady_reactive extends L=10 down to cap=500."""
    bare = build_bare_training_chain(L=10)
    annotated = apply_auto_policy(bare, fast_memory_capacity=cap)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "b_0" in compute_ids


def test_iterative_refinement_recovers_valid_plan_at_tight_cap():
    """The policy's iterative refinement: at L=10 cap=600 the initial planning
    overshoots capacity at some prefetches; refinement shifts the prefetch
    earlier until the simulator accepts."""
    bare = build_bare_training_chain(L=10)
    annotated = apply_auto_policy(bare, fast_memory_capacity=600)
    log = run(annotated)
    makespan = max(iv.end for iv in log.task_intervals)
    # Should be a valid run (no exceptions); makespan within reasonable bound.
    assert makespan < 400


def test_roundtrip_enumeration_finds_wide_gap_weights():
    """roundtrip_planner's gap enumeration should find candidate round-trips for forward
    weights (W_0..W_2) since they have large gaps between f_i and r_i/b_i."""
    from dataflow_sim.policies._common import (
        _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx,
    )
    from dataflow_sim.policies.roundtrip_planner import _enumerate_roundtrips
    bare = build_bare_training_chain(L=3)
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)
    candidates = _enumerate_roundtrips(bare, sizes, uses_by_task, ideal)
    obj_ids = {c.obj_id for c in candidates}
    # W_0 and W_1 have wide gaps between forward and backward. W_2's gap
    # (f_2 end → r_2 start = 2 ticks) is too narrow for an 8+8 round-trip.
    assert "W_0" in obj_ids
    assert "W_1" in obj_ids


def test_per_task_use_events_collapse_duplicates():
    """A task's input list may reference the same object once; we should
    record a single use event per (task, obj) pair."""
    from dataflow_sim.policies._common import _compute_ideal_starts, _object_uses_by_task_idx
    bare = build_bare_training_chain(L=2)
    ideal = _compute_ideal_starts(bare)
    by_task = _object_uses_by_task_idx(bare, ideal)
    # W_0 used by f_0 (task 0), r_0, b_0 — exactly three events.
    w0_events = by_task["W_0"]
    assert len(w0_events) == 3
    assert [e.task_idx for e in w0_events] == sorted(e.task_idx for e in w0_events)


def test_initial_placement_leaves_slack_for_widest_task():
    """The slack-aware initial placement should leave room equal to the widest
    single-task footprint so future prefetches/cascade have headroom."""
    from dataflow_sim.policies.roundtrip_planner import _initial_placement
    from dataflow_sim.policies._common import _compute_ideal_starts, _compute_uses, _object_sizes
    bare = build_bare_training_chain(L=3)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # At cap=400 (which is < 2 × widest 224), slack should keep initial small.
    placement = _initial_placement(bare, fast_memory_capacity=400, uses=uses, sizes=sizes)
    initial_bytes = 16 + sum(sizes[oid] for oid in placement)  # input + placements
    # widest = 224, so cap - widest = 176. Initial pool (after T_1 outputs reserved
    # = 64) should fit in 400 - 224 = 176 bytes free. With must_place W_0 (64)
    # + input (16) = 80, headroom for greedy is small.
    assert initial_bytes <= 400 - 224 + 80  # widest-task slack + T_1 pinned


def test_auto_policy_L3_zero_stalls_at_unlimited():
    """At unlimited capacity, both policies should be stall-free."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, fast_memory_capacity=None)
    log = run(annotated)
    compute = sorted(
        [iv for iv in log.task_intervals if iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    # No gaps between consecutive compute tasks
    for prev, cur in zip(compute, compute[1:]):
        assert cur.start <= prev.end, f"stall: {prev.task_id}→{cur.task_id}"


def test_auto_policy_L3_unlimited_emits_no_transfers_without_final_locations():
    """At unlimited capacity and with no terminal placement constraints, the
    policy should not emit transfers for disposable mutated intermediates."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, fast_memory_capacity=None)
    n_pre = sum(len(t.prefetch_after) for t in annotated.tasks)
    n_off = sum(len(t.offload_after) for t in annotated.tasks)
    assert n_pre == 0
    assert n_off == 0


def test_auto_policy_L3_unlimited_honors_final_backing_locations():
    """A final-backing constraint, not object type, creates required writebacks."""
    bare0 = build_bare_training_chain(L=3)
    final_locations = {
        o.id: "backing" for o in bare0.initial_memory
        if o.location == "backing" and o.type == "gradient"
    }
    bare = replace(bare0, final_locations=final_locations)
    annotated = apply_auto_policy(bare, fast_memory_capacity=None)
    n_pre = sum(len(t.prefetch_after) for t in annotated.tasks)
    assert n_pre == 0

    offloaded = [tr.obj_id for t in annotated.tasks for tr in t.offload_after]
    from collections import Counter
    c = Counter(offloaded)
    for g in final_locations:
        assert c[g] == 1, f"gradient {g} offloaded {c[g]} times (want 1)"
    extra = set(offloaded) - set(final_locations)
    assert not extra, f"unexpected offloads at unlimited cap: {sorted(extra)}"


def test_auto_policy_emits_releases_at_tight_cap():
    """At a tight-but-feasible cap, the policy should emit at least some triggers."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, fast_memory_capacity=800)
    n_rel = sum(len(t.releases_after) for t in annotated.tasks)
    assert n_rel >= 1


# ---------- equivalence at generous capacity ----------

def test_releases_weight_instead_of_offloading_when_backing_copy_exists():
    """W_i is backing-initial and never mutated (workload contract). When the
    planner needs to free its compute bytes between fwd-use and bwd-use, the
    backing copy is byte-identical, so a release (instant, no to_slow) is
    correct; an offload would waste to_slow bandwidth re-writing identical
    bytes."""
    bare = build_bare_training_chain(L=5)
    annotated = apply_auto_policy(bare, fast_memory_capacity=600)
    # Collect every per-task to_slow trigger by object.
    offloaded_objs = set()
    for task in annotated.tasks:
        for trig in task.offload_after:
            offloaded_objs.add(trig.obj_id)
    # No W_i should be offloaded — releases handle them.
    w_offloads = {o for o in offloaded_objs if o.startswith("W_")}
    assert not w_offloads, f"belady_reactive wastefully offloaded weights with backing copies: {sorted(w_offloads)}"


def test_smart_initial_placement_defers_to_leave_room_for_outputs():
    """The smart initial placement should DEFER backing objects whose
    pre-placement would push pessimistic-bps over cap at some boundary,
    leaving room for task outputs (activations) that accumulate over
    forward. Smart init must not pre-place every dW_i + W_head + dW_head
    just because their SUM fits under cap; it must leave room for
    activations that accumulate later during forward."""
    from dataflow_sim.policies.belady_reactive import _smart_initial_placement
    from dataflow_sim.policies._common import (
        _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx,
    )
    bare = build_bare_training_chain(
        L=5, input_size=50, weight_size=100, activation_size=200,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_from_slow=50, bandwidth_to_slow=50,
    )
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, _compute_ideal_starts(bare))

    # At loose cap, smart init places everything backing with a use.
    loose = _smart_initial_placement(bare, 100_000, sizes, uses_by_task)
    all_backing_with_use = {
        o.id for o in bare.initial_memory
        if o.location == "backing" and uses_by_task.get(o.id)
    }
    assert loose == all_backing_with_use

    # At tight cap where SUM of weights+grads alone fits but adding
    # activations would overflow, smart init defers the backing objects whose
    # first-use is LATEST (backward grads, head, dW_head). At least one of
    # {W_head, dW_head, dW_4, dW_3, dW_2, dW_1, dW_0} must be deferred.
    tight = _smart_initial_placement(bare, 1200, sizes, uses_by_task)
    deferred = all_backing_with_use - tight
    late_use_objs = {"W_head", "dW_head"} | {f"dW_{i}" for i in range(5)}
    assert deferred & late_use_objs, (
        f"smart init didn't defer any backward-only object at tight cap; "
        f"placed={sorted(tight)}, deferred={sorted(deferred)}"
    )


def test_smart_initial_placement_at_loose_cap_eliminates_forward_stalls():
    """With cap big enough that smart init has room to fit everything live
    during forward (without pre-placing things that aren't needed until
    backward), forward tasks should run back-to-back with no stall."""
    bare = build_bare_training_chain(
        L=5, input_size=50, weight_size=100, activation_size=200,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_from_slow=50, bandwidth_to_slow=50,
    )
    # Cap that comfortably holds: 5 W + 5 A + y + input + reserved next-task
    # outputs + a couple of dW prefetches during forward.
    annotated = apply_auto_policy(bare, fast_memory_capacity=2500)
    log = run(annotated)
    f_compute = sorted(
        [iv for iv in log.task_intervals
         if iv.task_id.startswith("f_") and iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    assert len(f_compute) == 5
    for i in range(1, 5):
        gap = f_compute[i].start - f_compute[i - 1].end
        assert gap == 0, (
            f"forward stall between f_{i-1} and f_{i}: gap={gap}; "
            f"smart initial placement should have reserved room for "
            f"accumulating activations"
        )


def test_auto_writes_back_final_backing_objects():
    """Objects listed in final_locations must end on backing with live bytes."""
    bare0 = build_bare_training_chain(L=5)
    final_locations = {
        o.id: "backing" for o in bare0.initial_memory
        if o.location == "backing" and o.type == "gradient"
    }
    bare = replace(bare0, final_locations=final_locations)
    annotated = apply_auto_policy(bare, fast_memory_capacity=None)
    log = run(annotated)
    final = log.events[-1].snapshot
    for g in final_locations:
        backing_entry = next(
            (m for m in final.memory if m.id == g and m.location == "backing"),
            None,
        )
        assert backing_entry is not None, f"gradient {g!r} not on backing at end"
        assert backing_entry.state == "live", (
            f"{g!r} state={backing_entry.state}, want 'live'"
        )


def test_activation_offload_fires_eagerly_at_production():
    """When belady_reactive decides an activation (no backing source) must be
    offloaded, the offload must fire at the EARLIEST safe boundary (right
    after production) while to_slow is idle, not at the latest boundary that
    still meets the deadline."""
    # Tight cap that forces activation offloads.
    bare = build_bare_training_chain(
        L=8, input_size=50, weight_size=100, activation_size=500,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_from_slow=50, bandwidth_to_slow=50,
    )
    # Cap chosen so A_0 MUST be offloaded, else the assertion below no-ops.
    # At cap=2000 the L=8 backward-needed activations cycle off-compute during
    # forward.
    annotated = apply_auto_policy(bare, fast_memory_capacity=2000)
    log = run(annotated)
    f_0 = next(iv for iv in log.task_intervals if iv.task_id == "f_0")
    # Look for A_0's to_slow start.
    a0_to_slow = next(
        (iv for iv in log.task_intervals
         if iv.track == "to_slow" and iv.task_id.split(":", 1)[1].startswith("A_0")),
        None,
    )
    assert a0_to_slow is not None, (
        "A_0 wasn't offloaded at cap=2000; tighten further or check policy "
        "behaviour — the eagerness assertion below needs a real to_slow to inspect"
    )
    # A_0 must start within ONE compute task of production (not "as late as
    # the deadline allows", which would be many tasks later).
    fwd_runtime = f_0.end - f_0.start
    delay = a0_to_slow.start - f_0.end
    assert delay < fwd_runtime, (
        f"A_0 to_slow delayed by {delay} units after production (f_0 ends at "
        f"{f_0.end}); expected < {fwd_runtime} (one task) since to_slow is "
        f"idle and earliest-safe boundary should be picked"
    )


def test_auto_at_least_as_fast_as_sliding_window_at_unlimited_cap():
    """At unlimited capacity, the auto-policy should be at least as fast as
    sliding-window because it doesn't emit unnecessary triggers (the
    sliding-window's tail-end `dW_*` offloads extend the makespan past the
    last compute task end)."""
    bare = build_bare_training_chain(L=3)
    sliding = apply_sliding_window_policy(bare, window_size=2, fast_memory_capacity=None)
    auto = apply_auto_policy(bare, fast_memory_capacity=None)
    sliding_dur = max(iv.end for iv in run(sliding).task_intervals)
    auto_dur = max(iv.end for iv in run(auto).task_intervals)
    assert auto_dur <= sliding_dur, f"auto={auto_dur} > sliding={sliding_dur}"
