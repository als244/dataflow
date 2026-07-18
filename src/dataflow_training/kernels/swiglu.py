"""swiglu forward/backward: fused Triton (default) + eager fallback.

Op signatures (launch forms; out params are caller-provided):
- ``swiglu_fwd_out(kctx, x1, x3, out)``: out = silu(x1) * x3
- ``swiglu_bwd(kctx, ds, x1, x3, dx1_out, dx3_out)``
- ``swiglu_packed_fwd(kctx, h13, out)``: h13 = [x1 | x3] packed rows
- ``swiglu_packed_bwd(kctx, ds, h13, dh13_out)``: dh13 packed like h13

The PACKED variants read/write one (rows, 2F) buffer whose first F columns
are x1 (the silu input) and second F columns are x3 (the value) — the
repo-wide packed-matrix convention (introduced for MoE's stacked
``w13_experts``; dense MLP/QKV convert later). Numerics are pinned
BIT-IDENTICAL to the unpacked forms on the same values (silu rounds to the
storage dtype BEFORE the product — the ops.swiglu_fwd convention).

Eager materializes ~6 fp32 intermediates per row-chunk (bounded, but extra
memory passes); the fused kernels cast bf16 -> fp32 in registers only —
one read per input, one write per output, zero temporaries. Purely
elementwise, so results are launch-geometry independent.
"""
from __future__ import annotations

import torch

from ..blocks import ops
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


def _packed_halves(h13: torch.Tensor):
    f = h13.shape[-1] // 2
    return h13[:, :f], h13[:, f:]  # strided views; eager torch handles them


def _eager_packed_fwd(kctx, h13, out):
    x1, x3 = _packed_halves(h13)
    ops.swiglu_fwd_out(x1, x3, out)


def _eager_packed_bwd(kctx, ds, h13, dh13):
    x1, x3 = _packed_halves(h13)
    d1, d3 = _packed_halves(dh13)
    ops.swiglu_bwd(ds, x1, x3, d1, d3)


register(
    "swiglu_packed_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_rowchunk_hint), priority=0, fn=_eager_packed_fwd,
)
register(
    "swiglu_packed_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_rowchunk_hint), priority=0, fn=_eager_packed_bwd,
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

    # --- packed [x1 | x3] variants ---------------------------------------------
    # Same math bit-for-bit as the flat kernels; only the addressing changes
    # (row stride 2F, halves at column offsets 0 and F).

    @triton.jit
    def _packed_fwd_kernel(h_ptr, out_ptr, rows, f, BLOCK_F: tl.constexpr):
        pid_r = tl.program_id(0).to(tl.int64)
        pid_f = tl.program_id(1).to(tl.int64)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F).to(tl.int64)
        mask = offs_f < f
        row = h_ptr + pid_r * (2 * f)
        x1 = tl.load(row + offs_f, mask=mask, other=0).to(tl.float32)
        x3 = tl.load(row + f + offs_f, mask=mask, other=0).to(tl.float32)
        silu = x1 * tl.sigmoid(x1)
        # match ops.swiglu_fwd: silu rounds to storage dtype BEFORE the product
        silu = silu.to(out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_ptr + pid_r * f + offs_f,
                 (silu * x3).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _packed_bwd_kernel(ds_ptr, h_ptr, dh_ptr, rows, f, BLOCK_F: tl.constexpr):
        pid_r = tl.program_id(0).to(tl.int64)
        pid_f = tl.program_id(1).to(tl.int64)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F).to(tl.int64)
        mask = offs_f < f
        row = h_ptr + pid_r * (2 * f)
        x1 = tl.load(row + offs_f, mask=mask, other=0).to(tl.float32)
        x3 = tl.load(row + f + offs_f, mask=mask, other=0).to(tl.float32)
        ds = tl.load(ds_ptr + pid_r * f + offs_f, mask=mask, other=0).to(tl.float32)
        sig = tl.sigmoid(x1)
        dx1 = ds * x3 * (sig * (1 + x1 * (1 - sig)))
        dx3 = ds * (x1 * sig)
        drow = dh_ptr + pid_r * (2 * f)
        tl.store(drow + offs_f, dx1.to(dh_ptr.dtype.element_ty), mask=mask)
        tl.store(drow + f + offs_f, dx3.to(dh_ptr.dtype.element_ty), mask=mask)

    def _check_packed(h13, narrow, *wide):
        rows, two_f = h13.shape
        f = two_f // 2
        assert two_f == 2 * f and h13.is_contiguous() and h13.is_cuda
        assert narrow.shape == (rows, f) and narrow.is_contiguous()
        for t in wide:
            assert t.shape == (rows, two_f) and t.is_contiguous()
        return rows, f

    @register("swiglu_packed_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _packed_fwd(kctx, h13, out):
        rows, f = _check_packed(h13, out)
        _packed_fwd_kernel[(rows, triton.cdiv(f, _BLOCK))](
            h13, out, rows, f, BLOCK_F=_BLOCK
        )

    @register("swiglu_packed_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _packed_bwd(kctx, ds, h13, dh13_out):
        rows, f = _check_packed(h13, ds, dh13_out)
        _packed_bwd_kernel[(rows, triton.cdiv(f, _BLOCK))](
            ds, h13, dh13_out, rows, f, BLOCK_F=_BLOCK
        )
