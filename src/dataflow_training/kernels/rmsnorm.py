"""rmsnorm family: fused Triton (default) + eager fallback.

Op signatures (d = row width; all tensors contiguous):
- ``rmsnorm_fwd(kctx, x, w, out, rstd_out)``: out = bf16(x*rstd) * w, rstd fp32
- ``rmsnorm_apply(kctx, x, rstd, w, out)``: recompute from saved rstd
- ``rmsnorm_noweight(kctx, x, out, rstd_out)``: final model norm (w == 1)
- ``rmsnorm_bwd(kctx, dy, x, rstd, w, dx_out, dw_out)``: dw fp32-accumulated

Rounding matches the eager forms exactly in structure: the normalized value
rounds to the storage dtype BEFORE the weight multiply (torch bf16 semantics).

bwd needs a cross-row reduction for dw. Deterministic two-stage shape (the
flextrain structure): each program owns a contiguous row range and writes a
per-program fp32 partial dw row to a small internal buffer (n_programs x d);
a fixed-shape torch ``sum(0)`` reduces the partials. No atomics anywhere.
"""
from __future__ import annotations

import torch

from ..blocks import ops
from .registry import internal, none, register


def _eager_fwd_hint(x: torch.Tensor, *a) -> int:
    return 3 * x.numel() * 4


def _eager_bwd_hint(dy: torch.Tensor, *a) -> int:
    d = dy.shape[-1]
    return 6 * min(dy.shape[0], ops.ROWWISE_CHUNK) * d * 4


register(
    "rmsnorm_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_fwd_hint), priority=0,
    fn=lambda kctx, x, w, out, rstd_out: ops.rmsnorm_fwd(x, w, out, rstd_out),
)
register(
    "rmsnorm_apply", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_fwd_hint), priority=0,
    fn=lambda kctx, x, rstd, w, out: out.copy_(ops.rmsnorm_apply(x, rstd, w)),
)


def _eager_noweight(kctx, x, out, rstd_out):
    y, rstd = ops.rmsnorm_noweight(x)
    out.copy_(y)
    rstd_out.copy_(rstd)


register(
    "rmsnorm_noweight", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_fwd_hint), priority=0, fn=_eager_noweight,
)


def _eager_bwd(kctx, dy, x, rstd, w, dx_out, dw_out):
    dx, dw = ops.rmsnorm_bwd(dy, x, rstd, w)
    dx_out.copy_(dx)
    dw_out.copy_(dw)


register(
    "rmsnorm_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_bwd_hint), priority=0, fn=_eager_bwd,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _fwd_kernel(
        x_ptr, w_ptr, out_ptr, rstd_ptr, d, eps,
        HAS_W: tl.constexpr, SAVED_RSTD: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)

        if SAVED_RSTD:
            rstd = tl.load(rstd_ptr + row)
        else:
            acc = tl.zeros((BLOCK,), tl.float32)
            for off in range(0, d, BLOCK):
                xv = tl.load(x_ptr + row * d + off + cols,
                             mask=off + cols < d, other=0).to(tl.float32)
                acc += xv * xv
            rstd = 1.0 / tl.sqrt(tl.sum(acc) / d + eps)
            tl.store(rstd_ptr + row, rstd)

        for off in range(0, d, BLOCK):
            mask = off + cols < d
            xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
            # round to storage dtype BEFORE the weight multiply (eager parity)
            xn = (xv * rstd).to(out_ptr.dtype.element_ty).to(tl.float32)
            if HAS_W:
                wv = tl.load(w_ptr + off + cols, mask=mask, other=0).to(tl.float32)
                xn = xn * wv
            tl.store(out_ptr + row * d + off + cols,
                     xn.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _bwd_kernel(
        dy_ptr, x_ptr, rstd_ptr, w_ptr, dx_ptr, dw_partial_ptr,
        n_rows, d, ROWS_PER_PROG: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        row_lo = pid * ROWS_PER_PROG
        for r in range(ROWS_PER_PROG):
            row = row_lo + r
            live = row < n_rows
            rstd = tl.load(rstd_ptr + row, mask=live, other=0.0)
            # pass 1: c = mean(dxhat * xhat) over the row
            acc = tl.zeros((BLOCK,), tl.float32)
            for off in range(0, d, BLOCK):
                mask = live & (off + cols < d)
                xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                dyv = tl.load(dy_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                wv = tl.load(w_ptr + off + cols, mask=off + cols < d, other=0).to(tl.float32)
                acc += (dyv * wv) * (xv * rstd)
            c = tl.sum(acc) / d
            # pass 2: dx and the per-program dw partial
            for off in range(0, d, BLOCK):
                mask = live & (off + cols < d)
                xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                dyv = tl.load(dy_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                wv = tl.load(w_ptr + off + cols, mask=off + cols < d, other=0).to(tl.float32)
                xhat = xv * rstd
                dx = rstd * (dyv * wv - xhat * c)
                tl.store(dx_ptr + row * d + off + cols,
                         dx.to(dx_ptr.dtype.element_ty), mask=mask)
                part_ptr = dw_partial_ptr + pid * d + off + cols
                prev = tl.load(part_ptr, mask=off + cols < d, other=0.0)
                contrib = tl.where(mask, dyv * xhat, 0.0)
                tl.store(part_ptr, prev + contrib, mask=off + cols < d)

    _BLOCK = 1024
    _ROWS_PER_PROG = 64  # partials buffer stays ceil(rows/64) x d fp32

    def _fused_fwd(x, w, out, rstd_out, has_w, saved_rstd):
        d = x.shape[-1]
        _fwd_kernel[(x.shape[0],)](
            x, w if has_w else x, out, rstd_out, d, ops.RMS_EPS,
            HAS_W=has_w, SAVED_RSTD=saved_rstd, BLOCK=_BLOCK,
        )

    @register("rmsnorm_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fwd(kctx, x, w, out, rstd_out):
        _fused_fwd(x, w, out, rstd_out, True, False)

    @register("rmsnorm_apply", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _apply(kctx, x, rstd, w, out):
        _fused_fwd(x, w, out, rstd, True, True)

    @register("rmsnorm_noweight", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _noweight(kctx, x, out, rstd_out):
        _fused_fwd(x, x, out, rstd_out, False, False)

    def _bwd_hint(dy: torch.Tensor, *a) -> int:
        d = dy.shape[-1]
        n_progs = -(-dy.shape[0] // _ROWS_PER_PROG)
        return n_progs * d * 4  # fp32 dw partials

    @register("rmsnorm_bwd", "triton", deterministic=True, allocates="torch",
              workspace=internal(_bwd_hint),
              requires=lambda c: c.get("triton"), priority=10)
    def _bwd(kctx, dy, x, rstd, w, dx_out, dw_out):
        n_rows, d = dy.shape
        n_progs = -(-n_rows // _ROWS_PER_PROG)
        partials = torch.zeros((n_progs, d), device=dy.device, dtype=torch.float32)
        _bwd_kernel[(n_progs,)](
            dy, x, rstd, w, dx_out, partials, n_rows, d,
            ROWS_PER_PROG=_ROWS_PER_PROG, BLOCK=_BLOCK,
        )
        # fixed-shape torch reduction: deterministic, ~n_progs x d fp32 traffic
        dw_out.copy_(partials.sum(0).to(dw_out.dtype))
