"""layernorm family (gpt2): fused Triton (default) + eager fallback.

Op signatures (d = row width; all tensors contiguous; ``b`` may be None
— the bias-free variant):
- ``layernorm_fwd(kctx, x, w, b, out, mean_out, rstd_out)``: fp32 mean/rstd
  saved; normalized value rounds to storage dtype BEFORE the affine (the
  rmsnorm-family convention), then the multiply and bias-add each round
  like the eager torch ops they mirror.
- ``layernorm_apply(kctx, x, mean, rstd, w, b, out)``: recompute from saved
  statistics.
- ``layernorm_bwd(kctx, dy, x, mean, rstd, w, dx_out, dw_out, db_out)``:
  dw/db fp32-accumulated. Triton bwd uses the rmsnorm family's
  deterministic two-stage shape: per-program fp32 partial dw/db rows, a
  fixed-shape torch ``sum(0)`` reduce. No atomics anywhere.
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
    "layernorm_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_fwd_hint), priority=0,
    fn=lambda kctx, x, w, b, out, mean_out, rstd_out:
        ops.layernorm_fwd(x, w, b, out, mean_out, rstd_out),
)
register(
    "layernorm_apply", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_fwd_hint), priority=0,
    fn=lambda kctx, x, mean, rstd, w, b, out:
        out.copy_(ops.layernorm_apply(x, mean, rstd, w, b)),
)


def _eager_bwd(kctx, dy, x, mean, rstd, w, dx_out, dw_out, db_out):
    dx, dw, db = ops.layernorm_bwd(dy, x, mean, rstd, w)
    dx_out.copy_(dx)
    dw_out.copy_(dw)
    db_out.copy_(db)


register(
    "layernorm_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_bwd_hint), priority=0, fn=_eager_bwd,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _ln_fwd_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr, d, eps,
        HAS_B: tl.constexpr, SAVED_STATS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)

        if SAVED_STATS:
            mean = tl.load(mean_ptr + row)
            rstd = tl.load(rstd_ptr + row)
        else:
            acc = tl.zeros((BLOCK,), tl.float32)
            for off in range(0, d, BLOCK):
                xv = tl.load(x_ptr + row * d + off + cols,
                             mask=off + cols < d, other=0).to(tl.float32)
                acc += xv
            mean = tl.sum(acc) / d
            vacc = tl.zeros((BLOCK,), tl.float32)
            for off in range(0, d, BLOCK):
                mask = off + cols < d
                xv = tl.load(x_ptr + row * d + off + cols,
                             mask=mask, other=0).to(tl.float32)
                dv = tl.where(mask, xv - mean, 0.0)
                vacc += dv * dv
            rstd = 1.0 / tl.sqrt(tl.sum(vacc) / d + eps)
            tl.store(mean_ptr + row, mean)
            tl.store(rstd_ptr + row, rstd)

        for off in range(0, d, BLOCK):
            mask = off + cols < d
            xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
            # eager parity: normalized value rounds to storage dtype BEFORE
            # the affine; multiply and add then round like the torch ops
            xn = ((xv - mean) * rstd).to(out_ptr.dtype.element_ty).to(tl.float32)
            wv = tl.load(w_ptr + off + cols, mask=mask, other=0).to(tl.float32)
            y = (xn * wv).to(out_ptr.dtype.element_ty).to(tl.float32)
            if HAS_B:
                bv = tl.load(b_ptr + off + cols, mask=mask, other=0).to(tl.float32)
                y = y + bv
            tl.store(out_ptr + row * d + off + cols,
                     y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _ln_bwd_kernel(
        dy_ptr, x_ptr, mean_ptr, rstd_ptr, w_ptr, dx_ptr,
        dw_partial_ptr, db_partial_ptr,
        n_rows, d, ROWS_PER_PROG: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        row_lo = pid * ROWS_PER_PROG
        for r in range(ROWS_PER_PROG):
            row = row_lo + r
            live = row < n_rows
            mean = tl.load(mean_ptr + row, mask=live, other=0.0)
            rstd = tl.load(rstd_ptr + row, mask=live, other=0.0)
            # pass 1: c1 = mean(dxhat), c2 = mean(dxhat * xhat) over the row
            acc1 = tl.zeros((BLOCK,), tl.float32)
            acc2 = tl.zeros((BLOCK,), tl.float32)
            for off in range(0, d, BLOCK):
                mask = live & (off + cols < d)
                xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                dyv = tl.load(dy_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                wv = tl.load(w_ptr + off + cols, mask=off + cols < d, other=0).to(tl.float32)
                xhat = (xv - mean) * rstd
                dxhat = dyv * wv
                acc1 += tl.where(mask, dxhat, 0.0)
                acc2 += tl.where(mask, dxhat * xhat, 0.0)
            c1 = tl.sum(acc1) / d
            c2 = tl.sum(acc2) / d
            # pass 2: dx + the per-program dw/db partials
            for off in range(0, d, BLOCK):
                mask = live & (off + cols < d)
                xv = tl.load(x_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                dyv = tl.load(dy_ptr + row * d + off + cols, mask=mask, other=0).to(tl.float32)
                wv = tl.load(w_ptr + off + cols, mask=off + cols < d, other=0).to(tl.float32)
                xhat = (xv - mean) * rstd
                dx = rstd * (dyv * wv - c1 - xhat * c2)
                tl.store(dx_ptr + row * d + off + cols,
                         dx.to(dx_ptr.dtype.element_ty), mask=mask)
                dwp = dw_partial_ptr + pid * d + off + cols
                dbp = db_partial_ptr + pid * d + off + cols
                prev_w = tl.load(dwp, mask=off + cols < d, other=0.0)
                prev_b = tl.load(dbp, mask=off + cols < d, other=0.0)
                tl.store(dwp, prev_w + tl.where(mask, dyv * xhat, 0.0),
                         mask=off + cols < d)
                tl.store(dbp, prev_b + tl.where(mask, dyv, 0.0),
                         mask=off + cols < d)

    _BLOCK = 1024
    _ROWS_PER_PROG = 64

    def _fused_ln_fwd(x, w, b, out, mean_out, rstd_out, saved):
        d = x.shape[-1]
        _ln_fwd_kernel[(x.shape[0],)](
            x, w, b if b is not None else x, out, mean_out, rstd_out,
            d, ops.LN_EPS,
            HAS_B=b is not None, SAVED_STATS=saved, BLOCK=_BLOCK,
        )

    @register("layernorm_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fwd(kctx, x, w, b, out, mean_out, rstd_out):
        _fused_ln_fwd(x, w, b, out, mean_out, rstd_out, False)

    @register("layernorm_apply", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _apply(kctx, x, mean, rstd, w, b, out):
        _fused_ln_fwd(x, w, b, out, mean, rstd, True)

    def _bwd_hint(dy: torch.Tensor, *a) -> int:
        d = dy.shape[-1]
        n_progs = -(-dy.shape[0] // _ROWS_PER_PROG)
        return 2 * n_progs * d * 4  # fp32 dw + db partials

    @register("layernorm_bwd", "triton", deterministic=True, allocates="torch",
              workspace=internal(_bwd_hint),
              requires=lambda c: c.get("triton"), priority=10)
    def _bwd(kctx, dy, x, mean, rstd, w, dx_out, dw_out, db_out):
        n_rows, d = dy.shape
        n_progs = -(-n_rows // _ROWS_PER_PROG)
        partials = torch.zeros((2, n_progs, d), device=dy.device,
                               dtype=torch.float32)
        _ln_bwd_kernel[(n_progs,)](
            dy, x, mean, rstd, w, dx_out, partials[0], partials[1],
            n_rows, d, ROWS_PER_PROG=_ROWS_PER_PROG, BLOCK=_BLOCK,
        )
        dw_out.copy_(partials[0].sum(0).to(dw_out.dtype))
        db_out.copy_(partials[1].sum(0).to(db_out.dtype))
