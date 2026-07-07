"""Grouped GEMM over expert-contiguous row segments (the MoE experts stage).

One op family, three directions, all driven by DEVICE-side segment offsets
(the strict-paced engine forbids host reads of expert counts — flextrain's
default per-expert host loop is disqualified by its per-layer sync):

- ``moe_grouped_mm_fwd(kctx, x (M,Kd), w (E,Kd,N), offsets (E+1,) i32, out (M,N)|None)``
      out[r] = x[r] @ w[e(r)]   for offsets[e] <= r < offsets[e+1]
- ``moe_grouped_mm_dgrad(kctx, dy (M,N), w (E,Kd,N), offsets, dx_out (M,Kd)|None)``
      dx[r] = dy[r] @ w[e(r)].T

fwd/dgrad are DUAL-MODE: with out=None they RETURN the result tensor
(aten's own allocation) instead of copying into a caller buffer — scratch
destinations skip a full copy pass + a duplicate buffer (workspace
discipline); pass out= only when the destination is a ctx VIEW
(write-through).
- ``moe_grouped_mm_wgrad(kctx, x (M,Kd), dy (M,N), offsets, dw (E,Kd,N), *, accumulate)``
      dw[e] (+)= x_e.T @ dy_e   (out-of-place result, bf16-rounded, then
      copy_/add_ — the same rounding convention as the dense ``acc()``)

Rows at/after offsets[-1] are zero-filled in fwd/dgrad outputs and ignored
by wgrad (in our use offsets[-1] == M always). Empty segments are legal:
their dw slice is ZERO-filled in create mode (pinned by ladder test).

Default implementation is ``triton`` (ours): device-side offsets end to
end — a tiny async prep computes per-expert TILE prefix sums on device,
the main kernel binary-searches its tile, and a STATIC worst-case grid
(ceil(M/BM) + E partial tiles) early-exits past the real tile count, so
NO host code ever reads a count. Deterministic: every output tile has
exactly one owning program (wgrad loops its segment's K-dim in-program —
no atomics, no split-k); the epilogue writes/accumulates IN PLACE.

``aten-grouped`` (torch F.grouped_mm) is KEPT FOR A/B ONLY and demoted:
it HARD-SYNCS on every call (reads its offs tensor back to host to build
cutlass group descriptors — a full compute-stream drain; spin-kernel
audited). In the strict-paced engine that starved the pipeline inside
every MoE task (rc windows 48% GPU-idle) and, because profiling's
back-to-back reps hide drain costs, made in-run costs inflate
plan-dependently (the bs64 envelope-curve inversion). It violates the
registry ABI's no-sync contract.

The eager fallback is a masked dense E-loop — E-times the flops, exists
for DATAFLOW_KERNELS=eager numerics bisection at ladder scale only.
Weight dim 0 is the LOCAL expert count (kernels never assume it equals
global E — the expert-parallelism sharding seam).
"""
from __future__ import annotations

import torch

from .registry import internal, none, register

_ATEN_OK: bool | None = None


def _aten_grouped_available(caps: dict) -> bool:
    """One-shot capability probe (resolve-time only — syncs are fine here)."""
    global _ATEN_OK
    if not caps.get("cuda"):
        return False
    if _ATEN_OK is None:
        try:
            import torch.nn.functional as F

            x = torch.zeros(4, 16, dtype=torch.bfloat16, device="cuda")
            w = torch.zeros(2, 16, 16, dtype=torch.bfloat16, device="cuda")
            offs = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
            F.grouped_mm(x, w, offs=offs)
            F.grouped_mm(x.t(), x, offs=offs)  # wgrad form (strided mat_a)
            torch.cuda.synchronize()
            _ATEN_OK = True
        except Exception:
            _ATEN_OK = False
    return _ATEN_OK


@register("moe_grouped_mm_fwd", "aten-grouped", deterministic=True,
          workspace=internal(), requires=_aten_grouped_available,
          priority=5, allocates="vendor")
def _fwd_aten(kctx, x, w, offsets, out=None):
    import torch.nn.functional as F

    res = F.grouped_mm(x, w, offs=offsets[1:])
    if out is None:
        return res  # dual mode: scratch destinations skip the copy pass
    out.copy_(res)
    return out


@register("moe_grouped_mm_dgrad", "aten-grouped", deterministic=True,
          workspace=internal(), requires=_aten_grouped_available,
          priority=5, allocates="vendor")
def _dgrad_aten(kctx, dy, w, offsets, dx_out=None):
    import torch.nn.functional as F

    res = F.grouped_mm(dy, w.transpose(-2, -1), offs=offsets[1:])
    if dx_out is None:
        return res
    dx_out.copy_(res)
    return dx_out


@register("moe_grouped_mm_wgrad", "aten-grouped", deterministic=True,
          workspace=internal(), requires=_aten_grouped_available,
          priority=5, allocates="vendor")
def _wgrad_aten(kctx, x, dy, offsets, dw, *, accumulate: bool):
    import torch.nn.functional as F

    res = F.grouped_mm(x.t(), dy, offs=offsets[1:])  # (E, Kd, N)
    if accumulate:
        dw.add_(res.to(dw.dtype))
    else:
        dw.copy_(res.to(dw.dtype))


# --- eager fallback: masked dense E-loop (bisection scale only) ---------------


def _row_experts(offsets: torch.Tensor, m: int) -> torch.Tensor:
    """row -> segment id, device-side (no host counts): e(r) = #(ends <= r)."""
    rows = torch.arange(m, device=offsets.device, dtype=offsets.dtype)
    return torch.searchsorted(offsets[1:].contiguous(), rows, right=True)


def _eager_fwd(kctx, x, w, offsets, out=None):
    if out is None:
        out = torch.zeros(x.shape[0], w.shape[2], dtype=x.dtype, device=x.device)
    else:
        out.zero_()
    eids = _row_experts(offsets, x.shape[0])
    for e in range(w.shape[0]):
        sel = (eids == e).unsqueeze(1)
        out.copy_(torch.where(sel, x @ w[e], out))
    return out


def _eager_dgrad(kctx, dy, w, offsets, dx_out=None):
    if dx_out is None:
        dx_out = torch.zeros(dy.shape[0], w.shape[1], dtype=dy.dtype, device=dy.device)
    else:
        dx_out.zero_()
    eids = _row_experts(offsets, dy.shape[0])
    for e in range(w.shape[0]):
        sel = (eids == e).unsqueeze(1)
        dx_out.copy_(torch.where(sel, dy @ w[e].t(), dx_out))
    return dx_out


def _eager_wgrad(kctx, x, dy, offsets, dw, *, accumulate: bool):
    eids = _row_experts(offsets, x.shape[0])
    for e in range(dw.shape[0]):
        sel = (eids == e).unsqueeze(1)
        res = (x * sel.to(x.dtype)).t() @ dy  # rows outside e contribute zero
        if accumulate:
            dw[e].add_(res.to(dw.dtype))
        else:
            dw[e].copy_(res.to(dw.dtype))


for _op, _fn in (
    ("moe_grouped_mm_fwd", _eager_fwd),
    ("moe_grouped_mm_dgrad", _eager_dgrad),
    ("moe_grouped_mm_wgrad", _eager_wgrad),
):
    register(_op, "eager", deterministic=True, workspace=internal(),
             priority=0, allocates="torch", fn=_fn)


# --- triton grouped GEMM (default): device offsets, zero host syncs -----------

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments
    triton = None

if triton is not None:

    # BM is FIXED (the tile-prefix prep depends on it). 128x256x64 won a
    # static config sweep at BOTH perf shapes (bs16 131k rows / bs64 524k
    # rows, ragged multinomial segments): fwd 6% faster than
    # F.grouped_mm, wgrad within 9% (aten pays a hard sync on top
    # in-run). Static choice, NOT triton autotune — autotune's
    # timing-based selection is nondeterministic across runs.
    _BM, _BN, _BK = 128, 256, 64

    def _tile_prefix(offsets: torch.Tensor) -> torch.Tensor:
        """(E+1,) int32 device: cumsum of per-expert M-tile counts. Pure
        async torch — nothing reads device values on host."""
        counts = offsets[1:] - offsets[:-1]
        tiles = (counts + (_BM - 1)) // _BM
        tp = torch.zeros_like(offsets)
        tp[1:].copy_(torch.cumsum(tiles, 0).to(offsets.dtype))
        return tp

    @triton.jit
    def _grouped_mm_kernel(
        a_ptr, b_ptr, c_ptr, offs_ptr, tp_ptr,
        n_dim, k_dim, n_experts,
        sam, sak, sbe, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """C[r, n] = A[r, :] @ B[e(r), :, n] over expert-contiguous row
        segments. Grid dim0 is the STATIC worst case (ceil(M/BM) + E);
        programs past the true tile count (read from device) exit early.
        dgrad reuses this kernel with B's k/n strides swapped."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        total = tl.load(tp_ptr + n_experts)
        if pid_m >= total:
            return
        # binary search: largest e with tile_prefix[e] <= pid_m
        lo = 0
        hi = n_experts
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tl.load(tp_ptr + mid) <= pid_m:
                lo = mid
            else:
                hi = mid - 1
        e = lo
        seg_end = tl.load(offs_ptr + e + 1).to(tl.int64)
        row0 = tl.load(offs_ptr + e).to(tl.int64) \
            + (pid_m - tl.load(tp_ptr + e)).to(tl.int64) * BM
        rm = row0 + tl.arange(0, BM).to(tl.int64)
        rmask = rm < seg_end
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        nmask = rn < n_dim
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        b_base = b_ptr + e.to(tl.int64) * sbe
        for kk in range(0, k_dim, BK):
            rk = (kk + tl.arange(0, BK)).to(tl.int64)
            kmask = rk < k_dim
            a = tl.load(
                a_ptr + rm[:, None] * sam + rk[None, :] * sak,
                mask=rmask[:, None] & kmask[None, :], other=0.0,
            )
            b = tl.load(
                b_base + rk[:, None] * sbk + rn[None, :] * sbn,
                mask=kmask[:, None] & nmask[None, :], other=0.0,
            )
            acc = tl.dot(a, b, acc)
        tl.store(
            c_ptr + rm[:, None] * scm + rn[None, :] * scn,
            acc.to(c_ptr.dtype.element_ty),
            mask=rmask[:, None] & nmask[None, :],
        )

    @triton.jit
    def _grouped_wgrad_kernel(
        a_ptr, g_ptr, d_ptr, offs_ptr,
        k_dim, n_dim,
        sam, sak, sgm, sgn, sde, sdk, sdn,
        ACCUM: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr,
    ):
        """D[e] (+)= A_e^T @ G_e. Grid (E, ceil(Kd/BI), ceil(N/BJ)): every
        output tile has exactly ONE owning program (segment K-loop runs
        in-program) — deterministic, no atomics, no split-k. Empty
        segments store zeros in create mode and leave D unchanged in
        accumulate mode. The epilogue is IN PLACE (single rounding)."""
        e = tl.program_id(0).to(tl.int64)
        pid_i = tl.program_id(1)
        pid_j = tl.program_id(2)
        seg_s = tl.load(offs_ptr + e).to(tl.int64)
        seg_e = tl.load(offs_ptr + e + 1).to(tl.int64)
        ri = (pid_i * BI + tl.arange(0, BI)).to(tl.int64)
        imask = ri < k_dim
        rj = (pid_j * BJ + tl.arange(0, BJ)).to(tl.int64)
        jmask = rj < n_dim
        acc = tl.zeros((BI, BJ), dtype=tl.float32)
        for rr in range(seg_s, seg_e, BK):
            rk = rr + tl.arange(0, BK).to(tl.int64)
            kmask = rk < seg_e
            at = tl.load(  # A^T tile loaded directly: (BI, BK)
                a_ptr + rk[None, :] * sam + ri[:, None] * sak,
                mask=kmask[None, :] & imask[:, None], other=0.0,
            )
            g = tl.load(
                g_ptr + rk[:, None] * sgm + rj[None, :] * sgn,
                mask=kmask[:, None] & jmask[None, :], other=0.0,
            )
            acc = tl.dot(at, g, acc)
        ptr = d_ptr + e * sde + ri[:, None] * sdk + rj[None, :] * sdn
        m = imask[:, None] & jmask[None, :]
        if ACCUM:
            acc += tl.load(ptr, mask=m, other=0.0).to(tl.float32)
        tl.store(ptr, acc.to(d_ptr.dtype.element_ty), mask=m)

    def _check_grouped(x, w, offsets):
        assert x.is_cuda and x.is_contiguous() and x.dim() == 2
        assert w.is_cuda and w.dim() == 3
        assert offsets.is_cuda and offsets.dtype == torch.int32
        assert offsets.numel() == w.shape[0] + 1

    def _grid0(m: int, e: int) -> int:
        return (m + _BM - 1) // _BM + e  # static worst case; kernel early-exits

    @register("moe_grouped_mm_fwd", "triton", deterministic=True,
              workspace=none() if triton is None else internal(),
              requires=lambda c: c.get("triton"), priority=20, allocates="torch")
    def _fwd_triton(kctx, x, w, offsets, out=None):
        _check_grouped(x, w, offsets)
        m, n = x.shape[0], w.shape[2]
        if out is None:
            out = torch.empty(m, n, dtype=x.dtype, device=x.device)
        tp = _tile_prefix(offsets)
        _grouped_mm_kernel[(_grid0(m, w.shape[0]), (n + _BN - 1) // _BN)](
            x, w, out, offsets, tp, n, w.shape[1], w.shape[0],
            x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
            out.stride(0), out.stride(1),
            BM=_BM, BN=_BN, BK=_BK, num_warps=8, num_stages=3,
        )
        return out

    @register("moe_grouped_mm_dgrad", "triton", deterministic=True,
              workspace=none() if triton is None else internal(),
              requires=lambda c: c.get("triton"), priority=20, allocates="torch")
    def _dgrad_triton(kctx, dy, w, offsets, dx_out=None):
        # dx = dy @ w[e].T == the fwd kernel with w's k/n strides swapped
        _check_grouped(dy, w, offsets)
        m, kd = dy.shape[0], w.shape[1]
        if dx_out is None:
            dx_out = torch.empty(m, kd, dtype=dy.dtype, device=dy.device)
        tp = _tile_prefix(offsets)
        _grouped_mm_kernel[(_grid0(m, w.shape[0]), (kd + _BN - 1) // _BN)](
            dy, w, dx_out, offsets, tp, kd, w.shape[2], w.shape[0],
            dy.stride(0), dy.stride(1), w.stride(0), w.stride(2), w.stride(1),
            dx_out.stride(0), dx_out.stride(1),
            BM=_BM, BN=_BN, BK=_BK, num_warps=8, num_stages=3,
        )
        return dx_out

    @register("moe_grouped_mm_wgrad", "triton", deterministic=True,
              workspace=none() if triton is None else internal(),
              requires=lambda c: c.get("triton"), priority=20, allocates="torch")
    def _wgrad_triton(kctx, x, dy, offsets, dw, *, accumulate: bool):
        assert x.is_cuda and dy.is_cuda and dw.is_cuda and dw.dim() == 3
        kd, n = dw.shape[1], dw.shape[2]
        assert x.shape[1] == kd and dy.shape[1] == n
        _grouped_wgrad_kernel[(
            dw.shape[0], (kd + _BM - 1) // _BM, (n + _BN - 1) // _BN,
        )](
            x, dy, dw, offsets, kd, n,
            x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
            dw.stride(0), dw.stride(1), dw.stride(2),
            ACCUM=accumulate, BI=_BM, BJ=_BN, BK=_BK,
            num_warps=8, num_stages=3,
        )
