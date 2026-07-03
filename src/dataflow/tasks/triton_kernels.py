"""Fused elementwise Triton kernels: fp32 math in registers, zero temporaries.

The eager forms in ops.py materialize every fp32 intermediate of an
elementwise chain (~6 tensors of [rows x d_ff] for swiglu_bwd); row-chunking
bounds that scratch but keeps the extra memory passes. These kernels cast
bf16 -> fp32 *inside the kernel*, so the only global-memory traffic is one
read per input and one write per output, and torch allocates nothing.

Numerics: identical operation order to the eager fp32 forms; per-element
(no reductions), so results are independent of launch geometry. Validated
against the ops.py references by tests/tasks/test_triton_swiglu.py.

Launches follow torch's current stream (Executables run under
ExternalStream), and outputs are caller-provided — the no-alloc/no-sync
contract of the tasks layer holds. First call per shape JIT-compiles;
profiling warm-up absorbs that off the steady-state path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _swiglu_fwd_kernel(x1_ptr, x3_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0).to(tl.float32)
    x3 = tl.load(x3_ptr + offs, mask=mask, other=0).to(tl.float32)
    silu = x1 * tl.sigmoid(x1)
    # match ops.swiglu_fwd: silu rounds to the storage dtype BEFORE the product
    silu = silu.to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + offs, (silu * x3).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(ds_ptr, x1_ptr, x3_ptr, dx1_ptr, dx3_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0).to(tl.float32)
    x3 = tl.load(x3_ptr + offs, mask=mask, other=0).to(tl.float32)
    ds = tl.load(ds_ptr + offs, mask=mask, other=0).to(tl.float32)
    sig = tl.sigmoid(x1)
    dx1 = ds * x3 * (sig * (1 + x1 * (1 - sig)))
    dx3 = ds * (x1 * sig)
    tl.store(dx1_ptr + offs, dx1.to(dx1_ptr.dtype.element_ty), mask=mask)
    tl.store(dx3_ptr + offs, dx3.to(dx3_ptr.dtype.element_ty), mask=mask)


def _check(*tensors: torch.Tensor) -> int:
    n = tensors[0].numel()
    for t in tensors:
        assert t.is_contiguous() and t.is_cuda and t.numel() == n
    return n


def swiglu_fwd_out(x1: torch.Tensor, x3: torch.Tensor, out: torch.Tensor) -> None:
    """Drop-in for ops.swiglu_fwd_out: out = silu(x1) * x3, fused."""
    n = _check(x1, x3, out)
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_fwd_kernel[grid](x1, x3, out, n, BLOCK=_BLOCK)


def swiglu_bwd(
    ds: torch.Tensor, x1: torch.Tensor, x3: torch.Tensor,
    dx1_out: torch.Tensor | None = None, dx3_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ops.swiglu_bwd: dx1, dx3 in one fused pass."""
    dx1 = dx1_out if dx1_out is not None else torch.empty_like(x1)
    dx3 = dx3_out if dx3_out is not None else torch.empty_like(x3)
    n = _check(ds, x1, x3, dx1, dx3)
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_bwd_kernel[grid](ds, x1, x3, dx1, dx3, n, BLOCK=_BLOCK)
    return dx1, dx3
