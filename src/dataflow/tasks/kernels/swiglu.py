"""swiglu forward/backward: fused Triton (default) + eager fallback.

Op signatures (launch forms; out params are caller-provided):
- ``swiglu_fwd_out(kctx, x1, x3, out)``: out = silu(x1) * x3
- ``swiglu_bwd(kctx, ds, x1, x3, dx1_out, dx3_out)``

Eager materializes ~6 fp32 intermediates per row-chunk (bounded, but extra
memory passes); the fused kernels cast bf16 -> fp32 in registers only —
one read per input, one write per output, zero temporaries. Purely
elementwise, so results are launch-geometry independent.
"""
from __future__ import annotations

import torch

from .. import ops
from .registry import internal, none, register

_BLOCK = 1024


def _rowchunk_hint(*tensors: torch.Tensor) -> int:
    # eager fp32 temporaries: ~6 chunk-sized tensors (see ops.swiglu_bwd)
    d = tensors[0].shape[-1]
    return 6 * min(tensors[0].shape[0], ops.ROWWISE_CHUNK) * d * 4


register(
    "swiglu_fwd_out", "eager", deterministic=True, allocates="torch",
    workspace=internal(_rowchunk_hint), priority=0,
    fn=lambda kctx, x1, x3, out: ops.swiglu_fwd_out(x1, x3, out),
)
register(
    "swiglu_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_rowchunk_hint), priority=0,
    fn=lambda kctx, ds, x1, x3, dx1, dx3: ops.swiglu_bwd(ds, x1, x3, dx1, dx3),
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments
    triton = None

if triton is not None:

    @triton.jit
    def _fwd_kernel(x1_ptr, x3_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x1 = tl.load(x1_ptr + offs, mask=mask, other=0).to(tl.float32)
        x3 = tl.load(x3_ptr + offs, mask=mask, other=0).to(tl.float32)
        silu = x1 * tl.sigmoid(x1)
        # match ops.swiglu_fwd: silu rounds to storage dtype BEFORE the product
        silu = silu.to(out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_ptr + offs, (silu * x3).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _bwd_kernel(ds_ptr, x1_ptr, x3_ptr, dx1_ptr, dx3_ptr, n, BLOCK: tl.constexpr):
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

    @register("swiglu_fwd_out", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fwd(kctx, x1, x3, out):
        n = _check(x1, x3, out)
        _fwd_kernel[(triton.cdiv(n, _BLOCK),)](x1, x3, out, n, BLOCK=_BLOCK)

    @register("swiglu_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _bwd(kctx, ds, x1, x3, dx1_out, dx3_out):
        n = _check(ds, x1, x3, dx1_out, dx3_out)
        _bwd_kernel[(triton.cdiv(n, _BLOCK),)](
            ds, x1, x3, dx1_out, dx3_out, n, BLOCK=_BLOCK
        )
