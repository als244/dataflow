"""Engine stress gates (GPU): poison-on-free, interleaving stress, measured-cost
replanning — all must leave the math golden.

Tests:
- test_poison_on_free_changes_nothing: enabling 0xFF poison-on-free yields identical weights and loss under heavy offload/prefetch traffic, with no NaN.
- test_interleaving_stress_changes_nothing: random device work before each task reshuffles completion order yet leaves weights and loss identical.
- test_measured_costs_replan_still_golden: profiling, writing measured runtimes and workspace back, and re-planning changes the plan but not the loss.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("cuda.bindings.runtime")  # dataflow.runtime.device.cuda imports it at module scope

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.runtime.interop import torch_view  # noqa: E402
from dataflow_training.model_families.llama3.blocks import build_resolver  # noqa: E402
from dataflow_training.model_families.llama3 import derive_dims, initial_values, lower_llama3  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.run.profiling import apply_measured_costs, profile_program  # noqa: E402
from dataflow_training.model_families.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.sim]

CFG = ShapedLlamaConfig(
    n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
    vocab_size=512, seq_len=128, batch=1,
)
CAP = 8 * 1024 * 1024  # tight: forces offload/prefetch traffic


def _run(engine_kwargs=None, resolver_wrapper=None, program_transform=None, seed=7):
    dims = derive_dims(CFG)
    program = lower_llama3(CFG)
    planned = plan_program(program, fast_memory_capacity=CAP)
    prog = planned.program
    if program_transform is not None:
        prog = program_transform(prog)

    backend = CudaBackend()
    values = initial_values(prog, CFG, backend, seed=seed)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = build_resolver(dims)
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow_training.data.segments import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, prog)},
    )
    # readback: final weights + loss
    out = {}
    for obj_id in ["W_embed", "W_0", "W_1", "W_head"]:
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        out[obj_id] = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
    loss_rec = result.objects.get("loss_0_0")
    out["loss"] = float(torch_view((loss_rec.backing or loss_rec.fast).buffer, (1,), torch.float32)[0])
    result.close()
    for buf in values.values():
        backend.free(buf)
    return out


def _assert_same(a: dict, b: dict, tol: float = 1e-3):
    assert abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-9) < tol, (a["loss"], b["loss"])
    for k in ("W_embed", "W_0", "W_1", "W_head"):
        err = rel_l2(a[k], b[k])
        assert err < tol, f"{k}: rel_l2={err}"


def test_poison_on_free_changes_nothing():
    """If any task read a released buffer, 0xFF poison would inject NaNs —
    identical results prove no use-after-release at a budget with heavy
    offload/prefetch traffic."""
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert all(v == v for v in [poisoned["loss"]])  # not NaN


def test_interleaving_stress_changes_nothing():
    """Random extra device work before each task shifts every completion
    ordering; results must be identical (event-ordering correctness)."""
    from dataflow.runtime.device.cuda_spin import SpinKernel

    def wrapper(resolver, backend):
        kernel = SpinKernel()
        rng = torch.Generator().manual_seed(123)

        class Jitter:
            def __init__(self, inner):
                self.inner = inner

            def launch(self, ctx):
                delay = float(torch.randint(20, 400, (1,), generator=rng)[0])
                kernel.launch_us(ctx.stream, delay)
                self.inner.launch(ctx)

        return lambda task: Jitter(resolver(task))

    base = _run()
    jittered = _run(resolver_wrapper=wrapper)
    _assert_same(jittered, base)


def test_measured_costs_replan_still_golden():
    """Profile every unique task, write measured runtimes+workspace back,
    re-plan on measured costs — the plan changes, the math must not."""
    dims = derive_dims(CFG)
    program = lower_llama3(CFG)
    backend = CudaBackend()
    profiles = profile_program(program, build_resolver(dims), backend, soak_seconds=0)
    measured = apply_measured_costs(program, profiles)

    # profiling must take effect: measured runtimes differ from the roofline
    # guesses (how many differ is a property of the machine, not the contract)
    changed = sum(
        1 for a, b in zip(program.tasks, measured.tasks)
        if abs(a.runtime_us - b.runtime_us) / a.runtime_us > 0.05
    )
    assert changed > 0
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    planned = plan_program(measured, fast_memory_capacity=CAP)
    values = initial_values(planned.program, CFG, backend, seed=7)
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    from dataflow_training.data.segments import uniform_segments
    result = Engine(backend).execute(
        planned.program, resolver=build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    rec = result.objects.get("loss_0_0")
    loss = float(torch_view((rec.backing or rec.fast).buffer, (1,), torch.float32)[0])
    result.close()
    assert abs(loss - base["loss"]) / max(abs(base["loss"]), 1e-9) < 1e-3
