"""M3 gate extras (GPU): poison-on-free, interleaving stress, measured-cost
replanning — all must leave the math golden."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.tasks.interop import torch_view  # noqa: E402
from dataflow.tasks.llama3_blocks import build_resolver  # noqa: E402
from dataflow.training.llama3_lowering import dims_of, initial_values, lower_llama3  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.profiling import apply_measured_costs, profile_program  # noqa: E402
from dataflow.training.shaped_llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

CFG = ShapedLlamaConfig(
    n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
    vocab_size=512, seq_len=128, batch=1,
)
CAP = 8 * 1024 * 1024  # tight: forces offload/prefetch traffic


def _run(engine_kwargs=None, resolver_wrapper=None, program_transform=None, seed=7):
    dims = dims_of(CFG)
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
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
    )
    # readback: final weights + loss
    out = {}
    for obj_id in ["W_embed", "W_0", "W_1", "W_head"]:
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        out[obj_id] = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
    loss_rec = result.objects.get("loss_0_0")
    out["loss"] = float(torch_view((loss_rec.backing or loss_rec.fast).buffer, (1,), torch.float32)[0])
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
    dims = dims_of(CFG)
    program = lower_llama3(CFG)
    backend = CudaBackend()
    profiles = profile_program(program, build_resolver(dims), backend)
    measured = apply_measured_costs(program, profiles)

    # measured runtimes should differ from roofline guesses for most tasks
    changed = sum(
        1 for a, b in zip(program.tasks, measured.tasks)
        if abs(a.runtime_us - b.runtime_us) / a.runtime_us > 0.05
    )
    assert changed > len(program.tasks) // 2
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    planned = plan_program(measured, fast_memory_capacity=CAP)
    values = initial_values(planned.program, CFG, backend, seed=7)
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
    )
    rec = result.objects.get("loss_0_0")
    loss = float(torch_view((rec.backing or rec.fast).buffer, (1,), torch.float32)[0])
    assert abs(loss - base["loss"]) / max(abs(base["loss"]), 1e-9) < 1e-3
