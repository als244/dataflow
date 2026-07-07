"""Deterministic embedding-gradient accumulation.

``index_add_`` is CUDA float atomicAdd — with duplicate tokens the add
order varies run to run (the ~1-in-5 one-ulp W_embed lottery, 8ff0075).
The deterministic sort+cumsum-diff eager form is sync-free but slow at
scale (fp32 cumsum over (t, d) = 536 MB scan ≈ 30 ms at 65k x 2048).

Triton design (default): stable-sort the tokens (device radix, same as
moe_sort), then ONE kernel with a program per sorted position. Only the
program at each segment START does work: it finds its run's end by
fixed-iteration binary search over the sorted tokens (predicate
monotone; no data-dependent while), accumulates the run's dy rows in
fp32, and read-modify-writes its vocab row. Exactly one writer per
vocab row -> deterministic WITHOUT atomics, exact fp32 segment sums
(no cumsum-diff cancellation), one rounding into the grad dtype.
Duplicate-heavy worst case (one token repeated t times) degrades to one
program scanning t rows — bounded, and unreachable for training data.
"""
from __future__ import annotations

import torch

from .registry import internal, register

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None


def _ws_hint(*tensors) -> int:
    t = tensors[0].shape[0]
    return 32 * t + 1024


def _eager_embed_bwd(kctx, tokens, dy, dw_embed, *, zero_first):
    from .. import ops

    ops.embed_bwd_accum(tokens, dy, dw_embed, zero_first=zero_first)


register("embed_bwd_accum", "eager", deterministic=True,
         workspace=internal(_ws_hint), priority=0, allocates="torch",
         fn=_eager_embed_bwd)


if triton is not None:

    @triton.jit
    def _embed_seg_kernel(st_ptr, order_ptr, dy_ptr, dw_ptr,
                          T, D, dy_stride, dw_stride,
                          ITERS: tl.constexpr, BD: tl.constexpr):
        i = tl.program_id(0)
        tok = tl.load(st_ptr + i)
        if i > 0:
            if tl.load(st_ptr + i - 1) == tok:
                return  # not a segment start — another program owns this run
        # first index in (i, T] whose token differs: binary search on the
        # monotone predicate P(j) = (j >= T) | (st[j] != tok); fixed
        # iteration count keeps control flow data-independent
        lo = i
        hi = T
        for _ in range(ITERS):
            mid = (lo + hi) // 2
            v = tl.load(st_ptr + tl.minimum(mid, T - 1))
            p = (mid >= T) | (v != tok)
            hi = tl.where(p, mid, hi)
            lo = tl.where(p, lo, mid)
        seg_len = hi - i
        for d0 in range(0, D, BD):
            rd = d0 + tl.arange(0, BD)
            dmask = rd < D
            acc = tl.zeros((BD,), tl.float32)
            for jj in range(0, seg_len):
                src = tl.load(order_ptr + i + jj)
                acc += tl.load(dy_ptr + src * dy_stride + rd, mask=dmask,
                               other=0.0).to(tl.float32)
            cur = tl.load(dw_ptr + tok * dw_stride + rd, mask=dmask,
                          other=0.0).to(tl.float32)
            tl.store(dw_ptr + tok * dw_stride + rd,
                     (cur + acc).to(dw_ptr.dtype.element_ty), mask=dmask)

    @register("embed_bwd_accum", "triton", deterministic=True,
              workspace=internal(_ws_hint),
              requires=lambda c: c.get("triton"), priority=10,
              allocates="torch")
    def _triton_embed_bwd(kctx, tokens, dy, dw_embed, *, zero_first):
        if zero_first:
            dw_embed.zero_()
        t, d = dy.shape
        order = torch.argsort(tokens.long(), stable=True)
        st = tokens.long()[order].contiguous()
        iters = max(1, (t - 1).bit_length() + 1)
        _embed_seg_kernel[(t,)](
            st, order, dy, dw_embed,
            t, d, dy.stride(0), dw_embed.stride(0),
            ITERS=iters, BD=1024, num_warps=4, num_stages=2,
        )
