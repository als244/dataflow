"""GPU-marked tests for the CUDA backend (skipped without a device)."""
import ctypes

import pytest

cudart = pytest.importorskip("cuda.bindings.runtime")
if cudart.cudaGetDeviceCount()[0] != cudart.cudaError_t.cudaSuccess or cudart.cudaGetDeviceCount()[1] < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.cuda_spin import SpinKernel, make_spin_resolver  # noqa: E402
from dataflow.training.planning import plan_program, simulate_program  # noqa: E402
from dataflow.training.shaped_llama3 import ShapedLlamaConfig, build_shaped_llama3  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def backend():
    return CudaBackend()


def test_memcpy_roundtrip_integrity(backend):
    n = 4 * 1024 * 1024
    host_a = backend.alloc("backing", n)
    host_b = backend.alloc("backing", n)
    dev = backend.alloc("fast", n)
    ctypes.memset(host_a.ptr, 0xAB, n)
    ctypes.memset(host_b.ptr, 0x00, n)
    stream = backend.create_stream("h2d")
    backend.memcpy_async(dev, host_a, n, stream)
    backend.memcpy_async(host_b, dev, n, stream)
    ev = backend.record_event(stream)
    backend.notify_after(stream, ev, "done", priority=0)
    assert backend.next_completion() == "done"
    data = (ctypes.c_ubyte * n).from_address(host_b.ptr)
    assert data[0] == 0xAB and data[n // 2] == 0xAB and data[n - 1] == 0xAB
    for buf in (host_a, host_b, dev):
        backend.free(buf)


def test_completion_tokens_in_order(backend):
    stream = backend.create_stream("compute")
    kernel = SpinKernel()
    for i in range(3):
        kernel.launch_us(stream, 200.0)
        ev = backend.record_event(stream)
        backend.notify_after(stream, ev, f"t{i}", priority=2)
    got = [backend.next_completion() for _ in range(3)]
    assert got == ["t0", "t1", "t2"]
    assert backend.next_completion() is None


def test_spin_wall_accuracy(backend):
    """Wall-true spins must hold duration at long AND short targets, warm or
    cold clocks (the clock64 version failed short-after-idle by 2.5x+)."""
    kernel = SpinKernel()
    assert kernel.verify(backend, target_us=2_000.0) == pytest.approx(1.0, rel=0.05)
    stream = backend.create_stream("compute")
    for target in (100.0, 500.0):
        a = backend.record_event(stream)
        kernel.launch_us(stream, target)
        b = backend.record_event(stream)
        backend.notify_after(stream, b, "x", priority=2)
        backend.next_completion()
        measured = backend.event_time_us(b) - backend.event_time_us(a)
        assert measured == pytest.approx(target, rel=0.15, abs=25.0)


def test_mini_program_end_to_end():
    """Real execution of a mini shaped program: completes, respects the
    budget, and lands near the simulator's predicted makespan."""
    from dataclasses import replace

    from dataflow.runtime.device.fake import FakeBackend

    cfg = ShapedLlamaConfig(
        n_layers=4, d_model=1024, n_heads=8, n_kv_heads=2, d_ff=4096,
        vocab_size=16384, seq_len=1024, batch=1,
    )
    cap = 256 * 1024 * 1024
    backend = CudaBackend()
    # plan against measured bidirectional bandwidth (directions contend on
    # this platform; see CudaBackend.measure_pcie)
    pcie = backend.measure_pcie(nbytes=128 * 1024 * 1024)
    program = replace(
        build_shaped_llama3(cfg),
        bandwidth_from_slow=pcie.bidi_h2d,
        bandwidth_to_slow=pcie.bidi_d2h,
    )
    planned = plan_program(program, fast_memory_capacity=cap)

    # fake dry run computes the exact buffer demand -> prewarm the real pool
    dry = Engine(FakeBackend()).execute(planned.program)

    resolver = make_spin_resolver(backend)
    result = Engine(backend).execute(
        planned.program, resolver=resolver, pool_prewarm=dry.pool_demand
    )

    assert result.final_location_violations == ()
    # peak ledger accounting must match the plan exactly (same admission logic)
    assert result.peak_fast_bytes == planned.peak_fast_bytes
    # physical time lands near prediction; generous 25% for a tiny program
    # where fixed overheads weigh most (the real gate uses larger configs)
    assert result.makespan_us == pytest.approx(planned.makespan_us, rel=0.25)
    # transfers really happened
    assert any(iv.track != "compute" for iv in result.trace.intervals)
    result.close()
