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

Default implementation is ``aten-grouped`` (torch F.grouped_mm — probed on
sm_120: bitwise-repeatable, 131-162 TF/s at target shapes = parity with a
per-expert cuBLAS loop, empty segments handled). Offsets ABI note:
F.grouped_mm takes cumulative segment ENDS (E,), i.e. ``offsets[1:]``.

The eager fallback is a masked dense E-loop — E-times the flops, exists for
DATAFLOW_KERNELS=eager numerics bisection at ladder scale only. Weight dim
0 is the LOCAL expert count (kernels never assume it equals global E — the
expert-parallelism sharding seam).
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
          priority=10, allocates="torch")
def _fwd_aten(kctx, x, w, offsets, out=None):
    import torch.nn.functional as F

    res = F.grouped_mm(x, w, offs=offsets[1:])
    if out is None:
        return res  # dual mode: scratch destinations skip the copy pass
    out.copy_(res)
    return out


@register("moe_grouped_mm_dgrad", "aten-grouped", deterministic=True,
          workspace=internal(), requires=_aten_grouped_available,
          priority=10, allocates="torch")
def _dgrad_aten(kctx, dy, w, offsets, dx_out=None):
    import torch.nn.functional as F

    res = F.grouped_mm(dy, w.transpose(-2, -1), offs=offsets[1:])
    if dx_out is None:
        return res
    dx_out.copy_(res)
    return dx_out


@register("moe_grouped_mm_wgrad", "aten-grouped", deterministic=True,
          workspace=internal(), requires=_aten_grouped_available,
          priority=10, allocates="torch")
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
