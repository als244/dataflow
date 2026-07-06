"""VMM arena gates: extent allocator, stable VAs, guard-deferred reclaim,
value integrity through map cycles, and E2E equivalence with the slab path.

Design: docs/notes/vmm-slab-design-v1.md.
"""
import pytest

cudart = pytest.importorskip("cuda.bindings.runtime")
if cudart.cudaGetDeviceCount()[0] != cudart.cudaError_t.cudaSuccess or cudart.cudaGetDeviceCount()[1] < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

import torch  # noqa: E402

from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.vmm import GRANULE, VmmArena, VmmError  # noqa: E402
from dataflow.tasks.interop import torch_view  # noqa: E402

MB = 1024**2


@pytest.fixture()
def backend():
    b = CudaBackend()
    yield b


def make_arena(backend, cap=64 * MB, headroom=8 * MB, **kw):
    return VmmArena(
        device_index=getattr(backend, "device", 0),
        capacity_bytes=cap,
        event_complete=backend.event_complete,
        headroom_bytes=headroom,
        **kw,
    )


def test_stable_va_and_value_integrity(backend):
    arena = make_arena(backend)
    try:
        b1 = arena.get("W_0", 3 * MB)
        v = torch_view(b1, (3 * MB // 4,), torch.float32)
        v.fill_(7.5)
        torch.cuda.synchronize()
        assert float(v[123]) == 7.5
        va_first = b1.ptr
        arena.put(b1)

        # same object returns at the SAME va; fresh physical (no stale reuse
        # guarantee is claimed — only that writes land and read back)
        b2 = arena.get("W_0", 3 * MB)
        assert b2.ptr == va_first
        v2 = torch_view(b2, (3 * MB // 4,), torch.float32)
        v2.fill_(-2.0)
        torch.cuda.synchronize()
        assert float(v2[0]) == -2.0
        arena.put(b2)

        # different object, different va
        b3 = arena.get("A_0_0_0", 3 * MB)
        assert b3.ptr != va_first
        arena.put(b3)
    finally:
        arena.close()


def test_shape_stability_enforced(backend):
    arena = make_arena(backend)
    try:
        b = arena.get("W_0", 2 * MB)
        arena.put(b)
        with pytest.raises(VmmError, match="shape-stable"):
            arena.get("W_0", 4 * MB)
    finally:
        arena.close()


def test_budget_reflow_across_size_classes(backend):
    # fill the pool with 2 MiB handles, cache them, then demand a size the
    # budget can only satisfy by RELEASING cached handles (reflow)
    arena = make_arena(backend, cap=16 * MB, headroom=0)
    try:
        pool = arena.pool_bytes
        n = pool // (2 * MB)
        bufs = [arena.get(f"o{i}", 2 * MB) for i in range(n)]
        assert arena.used_bytes == pool and arena.created_bytes == pool
        for b in bufs:
            arena.put(b)  # all cached: created stays at pool, live at 0
        assert arena.used_bytes == 0 and arena.created_bytes == pool

        big = arena.get("big", 6 * MB)  # must reflow >= 3 cached 2 MiB handles
        assert arena.handle_reflows >= 3
        assert arena.created_bytes <= pool
        vw = torch_view(big, (6 * MB // 4,), torch.float32)
        vw.fill_(3.25)
        torch.cuda.synchronize()
        assert float(vw[-1]) == 3.25
        arena.put(big)

        # steady state: the same sizes round-trip with ZERO new creates
        creates = arena.handle_creates
        again = [arena.get(f"o{i}", 2 * MB) for i in range(3)]
        b2 = None
        for b in again:
            arena.put(b)
        b2 = arena.get("big", 6 * MB)
        arena.put(b2)
        assert arena.handle_creates == creates
    finally:
        arena.close()


def test_guard_deferred_reclaim_and_fresh_va(backend):
    arena = make_arena(backend, cap=8 * MB, headroom=0)
    try:
        compute, _h2d, _d2h = (backend.create_stream(k) for k in ("compute", "h2d", "d2h"))
        b = arena.get("A", 2 * MB)
        va0 = b.ptr
        # a straggling write guarded by an event behind a busy stream
        spin_done = backend.record_event(compute)
        backend.memset_async(b, 0xFF, compute)
        b.guard_event = backend.record_event(compute)
        arena.put(b)  # guard may be pending -> deferred (either way is legal)

        # immediate re-get of the same tag: never blocks; fresh VA if deferred
        b2 = arena.get("A", 2 * MB)
        if arena._deferred:
            assert b2.ptr != va0 and arena.va_reassigned == 1
        torch.cuda.synchronize()
        arena.drain_reclaim()
        assert not arena._deferred
        arena.put(b2)
        assert arena.used_bytes == 0
        del spin_done
    finally:
        arena.close()


def test_pool_exhaustion_is_loud(backend):
    arena = make_arena(backend, cap=8 * MB, headroom=0)
    try:
        held = [arena.get(f"x{i}", 2 * MB) for i in range(arena.pool_bytes // (2 * MB))]
        with pytest.raises(VmmError, match="cannot back"):
            arena.get("one-more", 2 * MB)
        for b in held:
            arena.put(b)
    finally:
        arena.close()


def test_parked_reuse_and_eviction_accounting(backend):
    arena = make_arena(backend, cap=8 * MB, headroom=0)
    try:
        # park then re-get the SAME tag: zero driver calls, same VA
        b = arena.get("A", 2 * MB)
        va = b.ptr
        arena.put(b)
        maps_before = arena.maps
        b2 = arena.get("A", 2 * MB)
        assert b2.ptr == va and arena.maps == maps_before and arena.park_hits == 1
        arena.put(b2)

        # fill the pool with OTHER tags; the parked "A" must be reclaimed
        # (steal or eviction) without breaking the byte accounting
        held = []
        for i in range(arena.pool_bytes // (2 * MB)):
            held.append(arena.get(f"z{i}", 2 * MB))
        assert arena.used_bytes == arena.pool_bytes
        assert arena.created_bytes <= arena.pool_bytes
        assert arena._free_bytes == 0 and not arena._parked
        for b in held:
            arena.put(b)
        assert arena.used_bytes == 0
        assert arena._free_bytes == arena.created_bytes  # all parked/cached
    finally:
        arena.close()


def test_e2e_mini_vmm_matches_static():
    """Full train() on the mini config: vmm losses == static losses bitwise
    (same kernels, same order; only addresses differ)."""
    from dataflow.training.llama3 import ShapedLlamaConfig
    from dataflow.training.planning import plan_program
    from dataflow.training.train_loop import train
    from dataflow.training.families import resolve_family

    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
        vocab_size=512, seq_len=64, batch=2, grad_accum_rounds=2,
    )
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    planned = plan_program(program, fast_memory_capacity=64 * MB)

    losses = {}
    stats = {}
    for mode in ("static", "vmm"):
        backend = CudaBackend()
        report = train(
            planned.program, cfg, backend, steps=2, seed=11,
            placement_mode=mode,
        )
        losses[mode] = report.losses
        stats[mode] = report
    assert losses["vmm"] == losses["static"], (losses["vmm"], losses["static"])
    vs = stats["vmm"].vmm_stats
    assert vs is not None and vs["maps"] > 0
    assert stats["vmm"].step_slab_overflows == stats["static"].step_slab_overflows
