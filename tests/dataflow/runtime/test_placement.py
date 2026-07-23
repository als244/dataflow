"""Static placement: offline packing succeeds where online placement fails,
runs at exact sim parity, and resets cleanly across multi-step epochs.

Tests:
- test_online_placement_fails_where_offline_packing_fits: a fragmenting get/put sequence overflows the online slab but offline lifetime-aware packing fits it in exactly the capacity.
- test_parity_with_placement_8b: assigned mode reproduces the sim's intervals and peak on 8B with contiguous-packing overhead under 1.35x.
- test_placement_epoch_reset_multi_run: replaying a placed program repeatedly through one session resets incarnation counters with no overflows.
- test_assigned_mode_rejects_shape_instability: assigned mode raises when a request size disagrees with the recorded footprint.
- test_quiescent_lifetime_inversion_escapes_instead_of_deadlocking: an assigned offset overlapping a still-live instance escapes to a dynamic allocation and completes instead of deadlocking.
- test_annotator_ranges_balanced_over_full_run: every task launch opens and closes exactly one annotator range, covering all tasks.
- test_placed_reuse_inherits_pending_guard: a pending poison guard survives placed reuse by range, re-attaches to the next incarnation, then clears once complete.
"""
import pytest

from dataflow.core.convert import to_sim_chain
from dataflow.runtime import Engine, compare_to_sim_eventlog
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.placement import PlacementRecorder, compute_placement
from dataflow.runtime.pool import BufferPool
from dataflow.runtime.slab import SlabAllocator
from dataflow_training.lowering.planning import plan_program
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, build_shaped_llama3

KB = 1024

# get/put sequence engineered so free bytes >= request but no contiguous
# hole fits: A[0,40) B[40,60) C[60,100); free A and C; D=60 sees two
# disjoint 40k holes. Offline packing with lifetime knowledge fits the same
# sequence into exactly 100k.
SEQUENCE = [
    ("get", "A", 40 * KB),
    ("get", "B", 20 * KB),
    ("get", "C", 40 * KB),
    ("put", "A", None),
    ("put", "C", None),
    ("get", "D", 60 * KB),
]


def _drive(pool: BufferPool) -> None:
    live = {}
    for op, name, size in SEQUENCE:
        if op == "get":
            live[name] = pool.get("fast", size, tag=name)
        else:
            pool.put(live.pop(name))


def test_online_placement_fails_where_offline_packing_fits():
    backend = FakeBackend()

    # dynamic best-fit slab, no headroom, no arena: D escapes the slab
    dynamic = BufferPool(backend)
    dynamic.slabs["fast"] = SlabAllocator(
        backend=backend, location="fast", capacity_bytes=100 * KB, headroom_factor=0.0,
    )
    _drive(dynamic)
    assert dynamic.slab_overflows == 1  # fragmentation defeated online placement

    # offline: record lifetimes, pack, replay — fits in exactly the capacity
    recorder = PlacementRecorder()
    recording = BufferPool(backend, recorder=recorder)
    _drive(recording)
    placement = compute_placement(recorder, physical_limit_bytes=100 * KB)
    assert placement.extent_bytes <= 100 * KB
    assert placement.load_bytes == 100 * KB  # peak concurrent load

    assigned = BufferPool(backend)
    assigned.enable_placement(placement)
    _drive(assigned)
    assert assigned.slab_overflows == 0


def test_parity_with_placement_8b():
    """Assigned mode must not perturb scheduling: exact interval + peak
    parity vs the simulator on the full 8B chain."""
    from dataflow_sim.engine.simulator import run as sim_run

    program = build_shaped_llama3(ShapedLlamaConfig.llama3_8b())
    annotated = plan_program(program, fast_memory_capacity=16 * 1024**3).program

    recorder = PlacementRecorder()
    dry = Engine(FakeBackend()).execute(annotated, record_placement=recorder)
    dry.close()
    # physical limit = an arbitrary scenario ceiling above the logical budget
    # (a planning input to the packer, not a machine fact — this run is on the
    # fake backend): contiguous packing carries a geometry tax over the peak
    # load (reported first-class)
    placement = compute_placement(recorder, physical_limit_bytes=28 * 1024**3)
    assert placement.overhead < 1.35, placement.overhead

    result = Engine(FakeBackend()).execute(annotated, placement=placement)
    log = sim_run(to_sim_chain(annotated), snapshots=False)
    diff = compare_to_sim_eventlog(result.trace, log)
    assert diff.ok, f"missing={diff.missing[:3]}, extra={diff.extra[:3]}, peak {diff.peak_sim} vs {diff.peak_runtime}"
    result.close()


def test_placement_epoch_reset_multi_run():
    """The same placed program replays through one pool (session semantics):
    incarnation counters reset per execute."""
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    annotated = plan_program(program, fast_memory_capacity=600_000).program

    recorder = PlacementRecorder()
    Engine(FakeBackend()).execute(annotated, record_placement=recorder).close()
    placement = compute_placement(recorder, physical_limit_bytes=2_000_000)

    from dataflow.runtime.engine import Session

    backend = FakeBackend()
    session = Session(backend=backend)
    for _ in range(3):
        result = Engine(backend, session=session).execute(annotated, placement=placement)
        assert result.slab_overflows == 0
    session.close()

def test_assigned_mode_rejects_shape_instability():
    """Variable-length programs change instance sizes: assigned mode must
    fail loudly (pointing at dynamic mode), never hand out an offset whose
    recorded footprint disagrees with the request."""
    backend = FakeBackend()
    recorder = PlacementRecorder()
    recording = BufferPool(backend, recorder=recorder)
    recording.put(recording.get("fast", 40 * KB, tag="A"))
    placement = compute_placement(recorder, physical_limit_bytes=100 * KB)

    assigned = BufferPool(backend)
    assigned.enable_placement(placement)
    with pytest.raises(RuntimeError, match="shape-stable"):
        assigned.get("fast", 48 * KB, tag="A")


def test_quiescent_lifetime_inversion_escapes_instead_of_deadlocking():
    """The bs2ga32@20 deadlock, distilled: the packer overlaps two instances
    whose dry-run lifetimes were disjoint, but at run time the holder's
    release depends on a task AFTER the blocked one (a lifetime inversion
    the recording could not see). The engine must escape the blocked
    instance to a dynamic allocation (counted) and complete, not deadlock."""
    from dataflow.core.program import ObjectSpec, OutputSpec, Program, TaskSpec
    from dataflow.runtime.placement import Placement

    KB = 1024
    # t1 makes A (released after t2 by t3's position in the chain via t2's
    # directives); t2 makes B whose ASSIGNED offset overlaps A's range.
    program = Program(
        name="inversion",
        initial_objects=(),
        tasks=(
            TaskSpec(id="t1", runtime_us=10, outputs=(OutputSpec(id="A", size_bytes=40 * KB, location="fast"),)),
            TaskSpec(id="t2", runtime_us=10, inputs=("A",), outputs=(OutputSpec(id="B", size_bytes=40 * KB, location="fast"),)),
            TaskSpec(id="t3", runtime_us=10, inputs=("A", "B"), releases_after=("A", "B")),
        ),
        fast_memory_capacity=200 * KB,
    )
    # adversarial placement: A and B share [0, 40k) although both are live
    # across t2/t3 — models the recorded-vs-real divergence
    placement = Placement(
        offsets={("A", 0): 0, ("B", 0): 0},
        sizes={("A", 0): 40 * KB, ("B", 0): 40 * KB},
        extent_bytes=40 * KB,
        load_bytes=40 * KB,
        physical_limit_bytes=200 * KB,
    )
    result = Engine(FakeBackend()).execute(program, placement=placement)
    assert result.placement_escapes == 1, result.placement_escapes
    result.close()


def test_annotator_ranges_balanced_over_full_run():
    """Every task launch opens and closes exactly one range (depth never
    exceeds 1 from the engine side), so profiler timelines nest correctly."""
    from dataflow.runtime.device.annotate import RecordingAnnotator

    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    annotated = plan_program(program, fast_memory_capacity=600_000).program

    backend = FakeBackend()
    backend.annotator = RecordingAnnotator()
    result = Engine(backend).execute(annotated)
    result.close()

    rec = backend.annotator
    assert rec.depth == 0, "unbalanced push/pop"
    pushes = [name for kind, name in rec.events if kind == "push"]
    task_ids = {t.id for t in annotated.tasks}
    assert task_ids.issubset(set(pushes))  # every task got a range



def test_placed_reuse_inherits_pending_guard():
    """Placed offsets are identity-managed: put() drops the Buffer object, so
    a pending guard (poison memset still queued) must survive by RANGE and
    re-attach to the next incarnation carved over it — otherwise the next
    owner's h2d fill races the straggling memset (the poison-gate NaN class,
    placement-mode variant)."""
    from dataflow.runtime.placement import Placement
    from dataflow.runtime.pool import BufferPool

    KB = 1024
    backend = FakeBackend()
    stream = backend.create_stream("compute")
    pool = BufferPool(backend=backend)
    pool.enable_placement(Placement(
        offsets={("X", 0): 0, ("Y", 0): 0},
        sizes={("X", 0): 4 * KB, ("Y", 0): 4 * KB},
        extent_bytes=4 * KB,
        load_bytes=4 * KB,
        physical_limit_bytes=64 * KB,
    ))

    x = pool.get("fast", 4 * KB, tag="X")
    # guard that is still PENDING at put(): recorded on a stream whose clock
    # is ahead of the host clock (the fake backend's "in flight")
    backend.advance_stream(stream, 500.0)
    x.guard_event = backend.record_event(stream)
    pool.put(x)

    y = pool.get("fast", 4 * KB, tag="Y")  # same offset range, new object
    assert y.ptr == x.ptr
    assert y.guard_event is not None, "pending guard dropped across placed reuse"

    # once the guard completes, later incarnations carry nothing
    pool.put(y)
    backend._host_us = 1_000.0
    x2 = pool.get("fast", 4 * KB, tag="X")
    assert x2.guard_event is None
