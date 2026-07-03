"""AdamW step: fused Triton (default) + eager fallback.

Op signature:
- ``adamw_step(kctx, w, g, m, v, *, lr, beta1, beta2, eps, weight_decay,
  step)``: in-place on flat bf16 views.

Numerics contract (matches ops.adamw_step and the golden model EXACTLY in
structure): moments update in fp32, ROUND-TRIP through bf16 storage, and the
bias-corrected estimates are computed from the ROUNDED values — what the
next step will actually see. Bias-correction divisors are computed host-side
in python floats (same as eager) and passed as scalars.

The eager form makes ~12 passes with ~6 fp32 chunk temporaries; sitting on
the serial optimizer tail, that traffic is pure makespan. The fused kernel
is one pass: 4 reads + 3 writes, fp32 in registers only.
"""
from __future__ import annotations

from .. import ops
from .registry import internal, none, register


def _eager_hint(w, *a) -> int:
    return 6 * min(w.numel(), ops.ADAMW_CHUNK_ELEMS) * 4


def _eager(kctx, w, g, m, v, *, lr, beta1, beta2, eps, weight_decay, step):
    ops.adamw_step(w, g, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                   weight_decay=weight_decay, step=step)


register(
    "adamw_step", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0, fn=_eager,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _adamw_kernel(
        w_ptr, g_ptr, m_ptr, v_ptr, n,
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        gf = tl.load(g_ptr + offs, mask=mask, other=0).to(tl.float32)
        mf = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.float32)
        vf = tl.load(v_ptr + offs, mask=mask, other=0).to(tl.float32)
        wf = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)

        mf = mf * beta1 + gf * (1 - beta1)
        vf = vf * beta2 + gf * gf * (1 - beta2)
        # round-trip through storage dtype, THEN bias-correct (eager parity)
        m_r = mf.to(m_ptr.dtype.element_ty)
        v_r = vf.to(v_ptr.dtype.element_ty)
        tl.store(m_ptr + offs, m_r, mask=mask)
        tl.store(v_ptr + offs, v_r, mask=mask)
        mhat = m_r.to(tl.float32) / bc1
        vhat = v_r.to(tl.float32) / bc2
        wf = wf - lr * (mhat / (tl.sqrt(vhat) + eps) + weight_decay * wf)
        tl.store(w_ptr + offs, wf.to(w_ptr.dtype.element_ty), mask=mask)

    _BLOCK = 1024

    @register("adamw_step", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fused(kctx, w, g, m, v, *, lr, beta1, beta2, eps, weight_decay, step):
        n = w.numel()
        assert g.numel() == n and m.numel() == n and v.numel() == n
        _adamw_kernel[(triton.cdiv(n, _BLOCK),)](
            w, g, m, v, n,
            lr, beta1, beta2, eps, weight_decay,
            1 - beta1 ** step, 1 - beta2 ** step,  # host-side python floats
            BLOCK=_BLOCK,
        )
