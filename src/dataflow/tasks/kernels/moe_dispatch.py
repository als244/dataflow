"""MoE dispatch/combine ops — the expert-parallelism seam.

Dispatch and combine are adjoint pairs with clean tensor boundaries; under
future expert parallelism ONLY these ops swap implementations (all-to-all
exchange) while route/experts stages and family code stay unchanged.

Permutation vocabulary (pinned):
- ``order (rows,) i32``  — GATHER form: order[j] = flat assignment index
  (t*K + k) occupying slot j of the expert-sorted buffer. This is the
  saved-context field (single source of truth).
- ``slot_of (t,K) i32``  — SCATTER form (the inverse): slot_of[t,k] =
  slot j. Derived from ``order`` by one unique-index scatter where needed
  (fwd combine, bwd unpermutes) — never saved.

Ops:
- ``moe_sort(kctx, route_ids (t,K) i32, order_out (rows,) i32,
             offsets_out (E_local+1,) i32, *, n_experts)``
      stable sort of flat assignments by expert -> expert-contiguous
      segments; offsets[0]=0, offsets[e+1]-offsets[e] = count_e.
- ``moe_dispatch_fwd(kctx, x (t,d), order, out (rows,d), *, top_k)``
      out[j] = x[order[j] // K]  (token gather; measured bytes-bound).
- ``moe_dispatch_bwd(kctx, dxp (rows,d), slot_of (t,K) i32, out (t,d) fp32)``
      out[t] = sum_k dxp[slot_of[t,k]]  — gradient of dispatch.
- ``moe_combine_fwd(kctx, yp (rows,d), slot_of, route_w (t,K) bf16,
                    resid (t,d), out (t,d))``
      out[t] = resid[t] + sum_k route_w[t,k] * yp[slot_of[t,k]], fp32
      accumulator (single rounding to out dtype).

Combine's backward needs no op of its own: dyp = moe_dispatch_fwd(dy,
order) (the same gather) and dprob comes from a plain dot in the tail.

Determinism: no atomics anywhere — sort is torch's stable radix path,
gathers are pure, and both gather-sum kernels reduce over K inside one
program per (token, d-tile). The fused gather-sum kernel is a port of
flextrain's ``moe_gather_kernel`` (refs/flextrain, _kernels/moe.py; eager
unpermute+sum measured 4.97 ms vs ~0.4 ms fused at qwen35moe shapes — the
12x penalty that made this a v1 port, not a follow-up).
"""
from __future__ import annotations

import torch

from .registry import internal, none, register


# --- sort / gather: aten is already the right tool -----------------------------


def _sort_aten(kctx, route_ids, order_out, offsets_out, *, n_experts):
    flat = route_ids.reshape(-1).long()
    order_out.copy_(torch.argsort(flat, stable=True).to(order_out.dtype))
    counts = torch.bincount(flat, minlength=n_experts)
    offsets_out[:1].zero_()
    offsets_out[1:].copy_(counts.cumsum(0).to(offsets_out.dtype))


def _dispatch_fwd_aten(kctx, x, order, out, *, top_k):
    src = torch.div(order.long(), top_k, rounding_mode="floor")
    torch.index_select(x, 0, src, out=out)


register("moe_sort", "aten", deterministic=True, allocates="torch",
         workspace=internal(), priority=10, fn=_sort_aten)
register("moe_dispatch_fwd", "aten", deterministic=True, allocates="torch",
         workspace=internal(), priority=10, fn=_dispatch_fwd_aten)


# --- gather-sum pair: eager fallbacks ------------------------------------------


def _gathered(src, slot_of):
    t, k = slot_of.shape
    picked = torch.index_select(src, 0, slot_of.reshape(-1).long())
    return picked.view(t, k, src.shape[1]).float()


def _eager_dispatch_bwd(kctx, dxp, slot_of, out):
    out.copy_(_gathered(dxp, slot_of).sum(1))


def _eager_combine_fwd(kctx, yp, slot_of, route_w, resid, out):
    acc = (_gathered(yp, slot_of) * route_w.float().unsqueeze(-1)).sum(1)
    out.copy_((acc + resid.float()).to(out.dtype))


def _gather_hint(*tensors) -> int:
    src, slot_of = tensors[0], tensors[1]
    t, k = slot_of.shape
    return 2 * t * k * src.shape[1] * 4  # gathered fp32 (t,K,d) + product


register("moe_dispatch_bwd", "eager", deterministic=True, allocates="torch",
         workspace=internal(_gather_hint), priority=0, fn=_eager_dispatch_bwd)
register("moe_combine_fwd", "eager", deterministic=True, allocates="torch",
         workspace=internal(_gather_hint), priority=0, fn=_eager_combine_fwd)


# --- tail elementwise utilities (workspace discipline) --------------------------
# The v1 tail wrote `(x.float() * srw).to(bf16)` and
# `(a.float() * b.float()).sum(-1)` — correctness-first eager that
# MATERIALIZES multi-GiB fp32 tensors (4.3 GiB at bs64). These ops do the
# same math in registers: fp32 only inside the kernel, nothing but the
# (small) outputs allocated.
#
# - ``moe_scale_rows(kctx, x (rows,n), srw (rows,) fp32)``: x *= srw[row],
#   IN PLACE (fp32 product, bf16 store — bit-identical to the old
#   round-trip on the same values).
# - ``moe_rowdot(kctx, a (rows,n), b (rows,n), out (rows,) fp32)``:
#   out[r] = sum_j a[r,j]*b[r,j] in fp32.

_SCALE_CHUNK = 65536  # eager fallback: bounds the fp32 temp to chunk x n


def _eager_scale_rows(kctx, x, srw):
    for lo in range(0, x.shape[0], _SCALE_CHUNK):
        hi = min(lo + _SCALE_CHUNK, x.shape[0])
        x[lo:hi] = (x[lo:hi].float() * srw[lo:hi].unsqueeze(1)).to(x.dtype)


def _eager_rowdot(kctx, a, b, out):
    for lo in range(0, a.shape[0], _SCALE_CHUNK):
        hi = min(lo + _SCALE_CHUNK, a.shape[0])
        out[lo:hi] = (a[lo:hi].float() * b[lo:hi].float()).sum(-1)


def _chunk_hint(*tensors) -> int:
    n = tensors[0].shape[-1]
    return 2 * _SCALE_CHUNK * n * 4


register("moe_scale_rows", "eager", deterministic=True, allocates="torch",
         workspace=internal(_chunk_hint), priority=0, fn=_eager_scale_rows)
register("moe_rowdot", "eager", deterministic=True, allocates="torch",
         workspace=internal(_chunk_hint), priority=0, fn=_eager_rowdot)


# --- gather-sum pair: fused Triton (flextrain moe_gather_kernel port) ----------

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments
    triton = None

if triton is not None:

    @triton.jit
    def _gather_sum_kernel(
        src_ptr, slot_ptr, w_ptr, res_ptr, out_ptr,
        t, d, rows,
        K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
        USE_W: tl.constexpr, HAS_RES: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        mask_m = offs_m < t
        mask_d = offs_d < d
        tile = mask_m[:, None] & mask_d[None, :]

        if HAS_RES:
            acc = tl.load(
                res_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=tile, other=0.0,
            ).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for k in range(K):
            slot = tl.load(slot_ptr + offs_m * K + k, mask=mask_m, other=0).to(tl.int64)
            # clamp = memory safety under arbitrary bytes; no-op on real data
            slot = tl.minimum(tl.maximum(slot, 0), rows - 1)
            val = tl.load(
                src_ptr + slot[:, None] * d + offs_d[None, :],
                mask=tile, other=0.0,
            ).to(tl.float32)
            if USE_W:
                w = tl.load(w_ptr + offs_m * K + k, mask=mask_m, other=0.0).to(tl.float32)
                acc += val * w[:, None]
            else:
                acc += val

        tl.store(
            out_ptr + offs_m[:, None] * d + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty), mask=tile,
        )

    def _check_gather(src, slot_of, out):
        assert src.is_cuda and src.is_contiguous() and src.dim() == 2
        assert slot_of.is_cuda and slot_of.is_contiguous() and slot_of.dim() == 2
        assert out.is_cuda and out.is_contiguous()
        t, k = slot_of.shape
        assert out.shape == (t, src.shape[1])
        return t, k, src.shape[0], src.shape[1]

    _BM, _BD = 32, 128

    @register("moe_dispatch_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _dispatch_bwd(kctx, dxp, slot_of, out):
        t, k, rows, d = _check_gather(dxp, slot_of, out)
        _gather_sum_kernel[(triton.cdiv(t, _BM), triton.cdiv(d, _BD))](
            dxp, slot_of, dxp, dxp, out, t, d, rows,
            K=k, BLOCK_M=_BM, BLOCK_D=_BD, USE_W=False, HAS_RES=False,
        )

    @register("moe_combine_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _combine_fwd(kctx, yp, slot_of, route_w, resid, out):
        t, k, rows, d = _check_gather(yp, slot_of, out)
        assert route_w.shape == (t, k) and route_w.is_contiguous()
        assert resid.shape == out.shape and resid.is_contiguous()
        _gather_sum_kernel[(triton.cdiv(t, _BM), triton.cdiv(d, _BD))](
            yp, slot_of, route_w, resid, out, t, d, rows,
            K=k, BLOCK_M=_BM, BLOCK_D=_BD, USE_W=True, HAS_RES=True,
        )

    @triton.jit
    def _scale_rows_kernel(x_ptr, s_ptr, total, n, BLOCK: tl.constexpr):
        offs = (tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64))
        mask = offs < total
        row = offs // n
        x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
        s = tl.load(s_ptr + row, mask=mask, other=0.0)
        tl.store(x_ptr + offs, (x * s).to(x_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rowdot_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for off in range(0, n, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            m = cols < n
            a = tl.load(a_ptr + row * n + cols, mask=m, other=0).to(tl.float32)
            b = tl.load(b_ptr + row * n + cols, mask=m, other=0).to(tl.float32)
            acc += a * b
        tl.store(out_ptr + row, tl.sum(acc, axis=0))

    @register("moe_scale_rows", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _scale_rows(kctx, x, srw):
        assert x.is_cuda and x.is_contiguous() and x.dim() == 2
        assert srw.is_contiguous() and srw.numel() == x.shape[0]
        total = x.numel()
        _scale_rows_kernel[(triton.cdiv(total, 1024),)](
            x, srw, total, x.shape[1], BLOCK=1024
        )

    @register("moe_rowdot", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _rowdot(kctx, a, b, out):
        assert a.shape == b.shape and a.is_contiguous() and b.is_contiguous()
        assert out.numel() == a.shape[0] and out.dtype == torch.float32
        _rowdot_kernel[(a.shape[0],)](a, b, out, a.shape[1], BLOCK_N=1024)
