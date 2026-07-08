"""Qwen3 family correctness ladder (GPU) + CPU structural checks.

Ladder 1: qk-norm = the rmsnorm kernel family at head_dim-wide rows (no new
kernels) — fwd + bwd vs the autograd reference at per-head shapes.
Ladder 2: block fwd/bwd/recompute/accumulation vs autograd (family bundle).
Ladder 3: full annotated program through the real engine vs GoldenQwen3,
plus plan-invariance across budgets/recompute and a multi-step golden run.
"""
import pytest

torch = pytest.importorskip("torch")

from dataflow.tasks.layouts import Qwen3Dims, qwen3_context_layout  # noqa: E402
from dataflow.tasks.qwen3_blocks import Qwen3BlockFwd  # noqa: E402
from dataflow.training.qwen3 import dims_of_qwen3, lower_qwen3  # noqa: E402
from dataflow.training.qwen3 import ShapedQwen3Config  # noqa: E402

CFG = ShapedQwen3Config(
    n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
    d_ff=512, vocab_size=512, seq_len=128, batch=1,
)


# --- CPU: staged authoring + lowering structure ------------------------------

def test_qwen3_stage_context_completeness():
    dims = dims_of_qwen3(CFG)
    declared = {f.name for f in qwen3_context_layout(dims).fields}
    emitted = Qwen3BlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted


def test_qwen3_derived_recompute_excludes_boundary_work():
    n = Qwen3BlockFwd.recompute_stage_count()
    assert n < len(Qwen3BlockFwd.STAGES)
    last_emit = max(i for i, (_, _, e) in enumerate(Qwen3BlockFwd.STAGES) if e)
    assert n == last_emit + 1
    # rope emits nothing but sits BEFORE emitters — it must be inside the
    # recompute boundary (attn needs it), unlike the y-only tail stages
    names = [name for name, _, _ in Qwen3BlockFwd.STAGES]
    assert names.index("rope") < n
    assert names.index("swiglu") >= n


def test_qwen3_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.planning import plan_program, simulate_program

    program = lower_qwen3(CFG)
    validate_program(program)
    assert program.metadata["family"] == "qwen3-shaped"
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


# --- GPU ladders --------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@gpu
@pytest.mark.gpu
def test_qknorm_kernel_reuse_matches_reference():
    """Ladder 1 for the one new op shape: rmsnorm at (tokens*heads, head_dim)."""
    from dataflow.tasks import ops
    from dataflow.tasks.kernels import KernelCtx, resolve_kernels
    from dataflow.training.testing.gradcheck import rel_l2

    t, h, hd = 128, 4, 64
    gen = torch.Generator(device="cuda").manual_seed(3)
    qm = (torch.randn(t, h, hd, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(hd, generator=gen, device="cuda") * 0.1 + 1.0).to(torch.bfloat16)
    K, kctx = resolve_kernels(), KernelCtx()

    out = torch.empty(t * h, hd, dtype=torch.bfloat16, device="cuda")
    rstd = torch.empty(t * h, dtype=torch.float32, device="cuda")
    K.rmsnorm_fwd(kctx, qm.view(t * h, hd), w, out, rstd)
    ref = ops.rmsnorm_reference(qm, w)
    assert rel_l2(out.view(t, h, hd), ref) < 2e-2

    dy = (torch.randn(t, h, hd, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dx = torch.empty(t * h, hd, dtype=torch.bfloat16, device="cuda")
    dw = torch.empty(hd, dtype=torch.float32, device="cuda")
    K.rmsnorm_bwd(kctx, dy.view(t * h, hd), qm.view(t * h, hd), rstd, w, dx, dw)

    qm_ref = qm.clone().requires_grad_()
    w_ref = w.clone().requires_grad_()
    ops.rmsnorm_reference(qm_ref, w_ref).backward(dy)
    assert rel_l2(dx.view(t, h, hd), qm_ref.grad) < 3e-2
    assert rel_l2(dw, w_ref.grad.float()) < 3e-2


@gpu
@pytest.mark.gpu
def test_qwen3_block_backward():
    """Ladder 2: dx + every packed dW field (incl. q/k norm weights),
    recompute-equivalence, 2x accumulation."""
    from dataflow.training.families import family
    from dataflow.training.testing.gradcheck import check_block_backward

    check_block_backward(dims_of_qwen3(CFG), family=family("qwen3")).assert_ok()


@gpu
@pytest.mark.gpu
def test_qwen3_model_step_vs_golden():
    from dataflow.training.testing.gradcheck import check_model_step

    check_model_step(CFG, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


@gpu
@pytest.mark.gpu
def test_qwen3_plan_invariance():
    """Different budgets + recompute plans must produce identical math."""
    from dataflow.training.testing.gradcheck import check_model_step

    r1 = check_model_step(CFG, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2)
    r2 = check_model_step(CFG, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2)
    levels = {f"A_0_0_{i}": 1 for i in range(CFG.n_layers)}
    r3 = check_model_step(
        CFG, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
    )
    for r in (r1, r2, r3):
        r.assert_ok()


@gpu
@pytest.mark.gpu
def test_qwen3_multistep_matches_golden_and_loss_decreases():
    from dataflow.models.qwen3_reference import GoldenQwen3
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.planning import plan_program
    from dataflow.training.qwen3 import initial_values_qwen3
    from dataflow.training.train_loop import train

    STEPS = 3
    dims = dims_of_qwen3(CFG)
    planned = plan_program(lower_qwen3(CFG), fast_memory_capacity=8 * 1024 * 1024)
    backend = CudaBackend()

    gen = torch.Generator().manual_seed(99)
    one_batch = (
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
    )
    batches = [one_batch] * STEPS

    snapshot = initial_values_qwen3(planned.program, CFG, backend, seed=5)

    def pinned(name):
        buf = snapshot[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenQwen3.from_packed_bytes(
        dims, CFG.n_layers,
        pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(CFG.n_layers)],
        pinned("W_head"),
    )
    golden_losses = [
        golden.train_step(t.long().cuda(), g.long().cuda()) for t, g in batches
    ]

    report = train(
        planned.program, CFG, backend,
        steps=STEPS, seed=5, token_stream=lambda s: batches[s],
    )
    for ours, ref in zip(report.losses, golden_losses):
        assert abs(ours - ref) / max(abs(ref), 1e-9) < 3e-2, (report.losses, golden_losses)
    assert report.losses[-1] < report.losses[0]
    assert all(n == 0 for n in report.step_slab_overflows[1:]), report.step_slab_overflows
