"""Adversarial unit chains: each exercises one tricky scheduling semantic,
compared against the simulator where the sim accepts the chain, plus
engine-specific assertions (trace events, error types).

All chains use explicit per-trigger runtime overrides so no bandwidth is
needed, keeping the arithmetic trivial to verify by hand.

Tests:
- test_deferred_prefetch_waits_for_offload: a prefetch fired during the same object's in-flight offload defers, then runs, with the expected interval timing.
- test_blocked_queue_head_starts_when_bytes_free: a prefetch that will not fit blocks only the transfer queue while compute still dispatches, and starts the instant a release frees bytes.
- test_mutation_offload_overwrites_backing_in_place: offloading a mutated object over its live backing copy overwrites in place without charging extra backing bytes.
- test_re_prefetch_gets_distinct_interval_name: re-prefetching a re-offloaded object produces a distinct interval name alongside the first.
- test_capacity_deadlock_raises: a task whose output can never fit raises DeadlockError, and the sim errors on the same chain.
- test_release_of_non_live_object_errors: releasing an object not live in fast raises ExecutionError.
- test_initial_memory_over_capacity_errors: initial objects exceeding fast capacity raise LedgerError.
- test_stale_final_location_detected: mutating an object without offloading leaves backing stale and raises a final-location ExecutionError.
- test_ledger_inversion_parity_baseline: under exact sim timing the inversion chain completes with zero pressure evictions.
- test_ledger_inversion_eviction_valve: under distorted timing the valve evicts the farthest-next-use resident, reloads it, and completes within budget.
- test_ledger_inversion_without_valve_deadlocks: disabling the valve under the same distorted timing deadlocks (negative control).
- test_annotate_rename_rewrites_nvtx_only: the annotate_rename callback rewrites only NVTX push names, leaving trace and plan ids plan-relative.
"""
import pytest

from dataflow.core import ObjectSpec, OutputSpec, Program, TaskSpec, TransferDirective
from dataflow.core.convert import to_sim_chain
from dataflow.runtime import DeadlockError, Engine, ExecutionError, compare_to_sim_eventlog
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.ledger import LedgerError


def run_engine(program):
    return Engine(FakeBackend()).execute(program)


def run_both(program):
    from dataflow_sim.engine.simulator import run as sim_run

    result = run_engine(program)
    log = sim_run(to_sim_chain(program), snapshots=False)
    diff = compare_to_sim_eventlog(result.trace, log)
    assert diff.ok, f"missing={diff.missing}, extra={diff.extra}, peak {diff.peak_sim} vs {diff.peak_runtime}"
    return result, log


def test_deferred_prefetch_waits_for_offload():
    """Prefetch fired while the same object's offload is in flight must wait
    for the offload, then run — sim's transfer_deferred semantics."""
    program = Program(
        name="deferred-prefetch",
        initial_objects=(ObjectSpec(id="x", size_bytes=100, location="fast"),),
        tasks=(
            TaskSpec(
                id="t0", inputs=("x",), runtime_us=10.0,
                offload_after=(TransferDirective(object_id="x", runtime_us=50.0),),
            ),
            TaskSpec(
                id="t1", runtime_us=5.0,
                prefetch_after=(TransferDirective(object_id="x", runtime_us=7.0),),
            ),
            TaskSpec(id="t2", inputs=("x",), runtime_us=3.0),
        ),
        fast_memory_capacity=1_000,
    )
    result, _ = run_both(program)
    kinds = [e.kind for e in result.trace.events]
    assert "transfer_deferred" in kinds
    # offload [10,60]; deferred prefetch starts at 60, ends 67; t2 runs [67,70]
    by_id = {(iv.task_id, iv.track): iv for iv in result.trace.intervals}
    assert by_id[("to_slow:x", "to_slow")].end == 60.0
    assert by_id[("from_slow:x", "from_slow")].start == 60.0
    assert by_id[("t2", "compute")].start == 67.0


def test_blocked_queue_head_starts_when_bytes_free():
    """A prefetch whose destination doesn't fit must wait (blocking only the
    transfer queue, never compute dispatch) and start exactly when a release
    frees bytes."""
    program = Program(
        name="blocked-head",
        initial_objects=(
            ObjectSpec(id="big", size_bytes=800, location="fast"),
            ObjectSpec(id="w", size_bytes=500, location="backing"),
        ),
        tasks=(
            TaskSpec(
                id="t0", inputs=("big",), runtime_us=10.0,
                prefetch_after=(TransferDirective(object_id="w", runtime_us=4.0),),
            ),
            # t1 does NOT need w; it must dispatch while w's prefetch is blocked
            TaskSpec(
                id="t1", inputs=("big",), runtime_us=20.0,
                releases_after=("big",),
            ),
            TaskSpec(id="t2", inputs=("w",), runtime_us=5.0),
        ),
        fast_memory_capacity=1_000,
    )
    result, _ = run_both(program)
    by_id = {(iv.task_id, iv.track): iv for iv in result.trace.intervals}
    # t1 computes [10,30] concurrently with the blocked prefetch; release at 30
    assert by_id[("t1", "compute")].start == 10.0
    assert by_id[("from_slow:w", "from_slow")].start == 30.0  # unblocked by release
    assert by_id[("t2", "compute")].start == 34.0


def test_mutation_offload_overwrites_backing_in_place():
    """Offloading a mutated object with an existing live backing copy must
    overwrite it (no extra backing bytes) and refresh its version."""
    program = Program(
        name="mutate-overwrite",
        initial_objects=(
            ObjectSpec(id="w", size_bytes=100, location="fast"),
            ObjectSpec(id="w", size_bytes=100, location="backing"),
        ),
        tasks=(
            TaskSpec(
                id="opt", inputs=("w",), mutates=("w",), runtime_us=10.0,
                offload_after=(TransferDirective(object_id="w", runtime_us=5.0),),
            ),
        ),
        final_locations={"w": "backing"},
        fast_memory_capacity=1_000,
        backing_memory_capacity=100,  # would fail if the offload double-charged
    )
    result, _ = run_both(program)
    assert result.final_location_violations == ()


def test_re_prefetch_gets_distinct_interval_name():
    program = Program(
        name="re-prefetch",
        initial_objects=(ObjectSpec(id="x", size_bytes=100, location="fast"),),
        tasks=(
            TaskSpec(
                id="t0", inputs=("x",), runtime_us=10.0,
                offload_after=(TransferDirective(object_id="x", runtime_us=5.0),),
            ),
            TaskSpec(
                id="t1", runtime_us=20.0,
                prefetch_after=(TransferDirective(object_id="x", runtime_us=5.0),),
            ),
            TaskSpec(id="t2", inputs=("x",), runtime_us=5.0, releases_after=("x",)),
            TaskSpec(
                id="t3", runtime_us=8.0,
                prefetch_after=(TransferDirective(object_id="x", runtime_us=5.0),),
            ),
            TaskSpec(id="t4", inputs=("x",), runtime_us=5.0),
        ),
        fast_memory_capacity=1_000,
    )
    result, _ = run_both(program)
    names = {iv.task_id for iv in result.trace.intervals if iv.track == "from_slow"}
    assert names == {"from_slow:x", "from_slow:x#1"}


def test_capacity_deadlock_raises():
    """A task whose outputs can never fit must produce a clear deadlock error
    (the sim raises its own error on the same chain)."""
    program = Program(
        name="deadlock",
        initial_objects=(ObjectSpec(id="a", size_bytes=900, location="fast"),),
        tasks=(
            TaskSpec(
                id="t0", inputs=("a",), runtime_us=5.0,
                outputs=(OutputSpec(id="b", size_bytes=500),),
            ),
        ),
        fast_memory_capacity=1_000,
    )
    with pytest.raises(DeadlockError, match="waiting to reserve"):
        run_engine(program)
    from dataflow_sim.engine.simulator import run as sim_run

    with pytest.raises(ValueError):
        sim_run(to_sim_chain(program), snapshots=False, validate=False)


def test_release_of_non_live_object_errors():
    program = Program(
        name="bad-release",
        initial_objects=(ObjectSpec(id="x", size_bytes=100, location="backing"),),
        tasks=(TaskSpec(id="t0", runtime_us=5.0, releases_after=("x",)),),
        fast_memory_capacity=1_000,
    )
    with pytest.raises(ExecutionError, match="cannot release"):
        run_engine(program)


def test_initial_memory_over_capacity_errors():
    program = Program(
        name="init-over",
        initial_objects=(ObjectSpec(id="a", size_bytes=2_000, location="fast"),),
        tasks=(),
        fast_memory_capacity=1_000,
    )
    with pytest.raises(LedgerError, match="over-commit"):
        run_engine(program)


def test_stale_final_location_detected():
    """Mutating w without offloading leaves backing stale — the engine must
    flag the final_locations violation."""
    program = Program(
        name="stale-final",
        initial_objects=(
            ObjectSpec(id="w", size_bytes=100, location="fast"),
            ObjectSpec(id="w", size_bytes=100, location="backing"),
        ),
        tasks=(TaskSpec(id="opt", inputs=("w",), mutates=("w",), runtime_us=10.0),),
        final_locations={"w": "backing"},
        fast_memory_capacity=1_000,
    )
    with pytest.raises(ExecutionError, match="stale"):
        run_engine(program)


def _inversion_program() -> Program:
    """Sim-valid chain whose ONE-IN-FLIGHT h2d serialization protects t2's
    reservation under sim timing: when `a`'s offload frees 30 bytes at t=33,
    the h2d engine is still busy with `x` (until t=43), so blocked-head `far`
    cannot take the bytes and stalled t2 reserves first. Under distorted
    timing (h2d much faster than d2h), `x` finishes long before `a`'s
    offload, the idle h2d engine grabs the freed bytes for `far` ahead of
    t2's re-check, and `far`'s release depends on t3 > t2: a ledger
    lifetime inversion no waiting can resolve."""
    return Program(
        name="ledger-inversion",
        initial_objects=(
            ObjectSpec(id="a", size_bytes=30, location="fast"),
            ObjectSpec(id="x", size_bytes=40, location="backing"),
            ObjectSpec(id="far", size_bytes=60, location="backing"),
        ),
        tasks=(
            TaskSpec(id="t0", runtime_us=1.0),
            TaskSpec(
                id="t1", inputs=("a",), runtime_us=2.0,
                offload_after=(TransferDirective(object_id="a", runtime_us=30.0),),
                prefetch_after=(
                    TransferDirective(object_id="x", runtime_us=40.0),
                    TransferDirective(object_id="far", runtime_us=60.0),
                ),
            ),
            TaskSpec(
                id="t2", runtime_us=1000.0,
                outputs=(OutputSpec(id="w2", size_bytes=50),),
                releases_after=("w2",),
            ),
            TaskSpec(id="t3", inputs=("x", "far"), runtime_us=1.0),
        ),
        fast_memory_capacity=100,
        final_locations={"a": "backing"},
    )


def test_ledger_inversion_parity_baseline():
    """Under exact (sim) timing the chain completes with no evictions —
    the valve must never fire where parity holds."""
    result, _ = run_both(_inversion_program())
    assert result.pressure_evictions == 0


def test_ledger_inversion_eviction_valve():
    """Distorted timing (h2d 1000x faster) reorders admissions into a
    quiescent ledger deadlock; the valve evicts the farthest-next-use clean
    resident, reloads it before use, and the run completes within budget."""
    backend = FakeBackend(
        time_scale=lambda kind, us: us / 1000.0 if kind == "h2d" else us
    )
    result = Engine(backend).execute(_inversion_program())
    assert result.pressure_evictions >= 1
    assert result.peak_fast_bytes <= 100  # budget contract held throughout
    kinds = [e.kind for e in result.trace.events]
    assert "pressure_evict" in kinds
    evicted = [e.object_id for e in result.trace.events if e.kind == "pressure_evict"]
    assert evicted == ["far"]  # Belady: farthest next use, largest


def test_ledger_inversion_without_valve_deadlocks(monkeypatch):
    """Negative control: same distorted timing minus the valve must be the
    quiescent deadlock the valve exists to break."""
    from dataflow.runtime.engine import _RunState

    monkeypatch.setattr(_RunState, "_evict_for_pressure", lambda self, exclude: False)
    backend = FakeBackend(
        time_scale=lambda kind, us: us / 1000.0 if kind == "h2d" else us
    )
    with pytest.raises(DeadlockError, match="t2"):
        Engine(backend).execute(_inversion_program())


def test_annotate_rename_rewrites_nvtx_only():
    """NVTX display names get the caller's rename (global step substituted);
    trace intervals and plan ids stay plan-relative (they key replay-gap
    matching and the pinned-buffer registry)."""
    from dataflow.runtime.device.annotate import RecordingAnnotator

    program = Program(
        name="rename",
        initial_objects=(ObjectSpec(id="x", size_bytes=64, location="backing"),),
        tasks=(
            TaskSpec(
                id="t_0_1", runtime_us=5.0,
                prefetch_after=(TransferDirective(object_id="x", runtime_us=3.0),),
            ),
            TaskSpec(id="t_0_2", inputs=("x",), runtime_us=5.0),
        ),
        fast_memory_capacity=1_000,
    )
    backend = FakeBackend()
    backend.annotator = RecordingAnnotator()
    result = Engine(backend).execute(
        program,
        annotate_rename=lambda name: name.replace("_0_", "_7_"),
    )
    pushes = [n for kind, n in backend.annotator.events if kind == "push"]
    assert "t_7_1" in pushes and "t_7_2" in pushes
    assert "from_slow:x" in pushes  # no step field: unchanged
    assert not any("_0_" in n for n in pushes)
    # trace keeps plan-relative ids
    trace_ids = {iv.task_id for iv in result.trace.intervals}
    assert {"t_0_1", "t_0_2", "from_slow:x"} <= trace_ids
