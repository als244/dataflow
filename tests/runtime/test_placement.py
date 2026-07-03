"""Static placement: offline packing succeeds where online placement fails,
runs at exact sim parity, and resets cleanly across multi-step epochs."""
import pytest

from dataflow.core.convert import to_sim_chain
from dataflow.runtime import Engine, compare_to_sim_eventlog
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.placement import PlacementRecorder, compute_placement
from dataflow.runtime.pool import BufferPool
from dataflow.runtime.slab import SlabAllocator
from dataflow.training.planning import plan_program
from dataflow.training.shaped_llama3 import ShapedLlamaConfig, build_shaped_llama3

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
    # physical limit = device VRAM class, NOT the logical budget: contiguous
    # packing carries a geometry tax over the peak load (reported first-class)
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
