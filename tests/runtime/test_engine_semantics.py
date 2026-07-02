"""Adversarial unit chains: each exercises one tricky scheduling semantic,
compared against the simulator where the sim accepts the chain, plus
engine-specific assertions (trace events, error types).

All chains use explicit per-trigger runtime overrides so no bandwidth is
needed, keeping the arithmetic trivial to verify by hand.
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
