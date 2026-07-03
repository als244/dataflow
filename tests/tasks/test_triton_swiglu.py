"""Fused Triton swiglu vs the eager ops.py forms and autograd.

The fused kernels must match the row-chunked eager implementations at
bf16 output tolerance (same fp32 math, potentially different transcendental
ulps) and the autograd gradient of the pure reference.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("triton")

from dataflow.tasks import ops  # noqa: E402
from dataflow.tasks import triton_kernels as tk  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

SHAPES = [(128, 512), (2048, 1024), (4096 + 3, 14336)]  # incl. non-multiple of block


@pytest.mark.parametrize("rows,dff", SHAPES)
def test_fused_fwd_matches_eager(rows, dff):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x1 = torch.randn(rows, dff, device="cuda", generator=gen, dtype=torch.bfloat16)
    x3 = torch.randn(rows, dff, device="cuda", generator=gen, dtype=torch.bfloat16)

    eager = torch.empty_like(x1)
    ops.swiglu_fwd_out(x1, x3, eager)
    fused = torch.empty_like(x1)
    tk.swiglu_fwd_out(x1, x3, fused)
    assert rel_l2(fused.float(), eager.float()) < 2e-3


@pytest.mark.parametrize("rows,dff", SHAPES)
def test_fused_bwd_matches_eager_and_autograd(rows, dff):
    gen = torch.Generator(device="cuda").manual_seed(1)
    x1 = torch.randn(rows, dff, device="cuda", generator=gen, dtype=torch.bfloat16)
    x3 = torch.randn(rows, dff, device="cuda", generator=gen, dtype=torch.bfloat16)
    ds = torch.randn(rows, dff, device="cuda", generator=gen, dtype=torch.bfloat16)

    e_dx1, e_dx3 = ops.swiglu_bwd(ds, x1, x3)
    f_dx1, f_dx3 = tk.swiglu_bwd(ds, x1, x3)
    assert rel_l2(f_dx1.float(), e_dx1.float()) < 2e-3
    assert rel_l2(f_dx3.float(), e_dx3.float()) < 2e-3

    # against autograd through the pure reference
    a1 = x1.detach().clone().requires_grad_(True)
    a3 = x3.detach().clone().requires_grad_(True)
    ops.swiglu_fwd(a1, a3).backward(ds)
    assert rel_l2(f_dx1.float(), a1.grad.float()) < 3e-2
    assert rel_l2(f_dx3.float(), a3.grad.float()) < 3e-2


def test_fused_bwd_no_torch_allocation():
    """Steady-state launch must not touch the torch allocator (post-JIT)."""
    x1 = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    x3 = torch.randn_like(x1)
    ds = torch.randn_like(x1)
    dx1, dx3 = torch.empty_like(x1), torch.empty_like(x1)
    tk.swiglu_bwd(ds, x1, x3, dx1, dx3)  # JIT warm-up
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    tk.swiglu_bwd(ds, x1, x3, dx1, dx3)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
