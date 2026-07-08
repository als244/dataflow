"""gated_rmsnorm family: silu(z) * rmsnorm(o) * w over lin_v_head_dim rows
(the Gated-DeltaNet output norm).

Op signatures (rows = tokens*lin_v_heads, d = lin_v_head_dim; all contiguous):
- ``gated_rmsnorm_fwd(kctx, o, z, w, out, rstd_out)``: o/z/out (rows, d),
  w (d,), rstd_out (rows,) fp32.
- ``gated_rmsnorm_bwd(kctx, dy, o, z, w, rstd, do_out, dz_out, dw_out, y_out)``:
  gradients for all three inputs from the saved rstd; ``y_out`` receives the
  recomputed forward output (the caller's out-projection grad needs it, and
  the fused kernel produces it for free).

Default implementation wraps fla's fused Triton kernels
(``fla.modules.fused_norm_gate``, activation="swish", is_rms_norm=True) —
pinned bit-identical fwd / bf16-tolerance bwd against
``ops.gated_rmsnorm_reference`` by tests/tasks/test_qwen35_math.py. The
eager fallback runs the reference forms (autograd for bwd).
"""
from __future__ import annotations

import torch

from .. import ops
from .registry import internal, register


def _fla_available(caps: dict) -> bool:
    if not (caps.get("cuda") and caps.get("triton")):
        return False
    try:
        import fla.modules.fused_norm_gate  # noqa: F401

        return True
    except Exception:
        return False


def _fwd_hint(o: torch.Tensor, *a) -> int:
    # fla allocates y (+ rstd) internally before we copy into the out views
    return o.numel() * o.element_size() + o.shape[0] * 4


def _bwd_hint(dy: torch.Tensor, *a) -> int:
    # dx, dg, y allocated internally + fp32 dw partials
    return 3 * dy.numel() * dy.element_size() + 64 * dy.shape[-1] * 4


def _fla_fwd(kctx, o, z, w, out, rstd_out):
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    y, _mean, rstd, _res = layer_norm_gated_fwd(
        o, z, w, None, activation="swish", eps=ops.RMS_EPS, is_rms_norm=True,
    )
    out.copy_(y)
    rstd_out.copy_(rstd)


def _fla_bwd(kctx, dy, o, z, w, rstd, do_out, dz_out, dw_out, y_out):
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd

    dx, dg, dw, _db, _dres, y_norm = layer_norm_gated_bwd(
        dy, o, z, w, None, activation="swish", eps=ops.RMS_EPS,
        rstd=rstd, is_rms_norm=True, recompute_output=True,
    )
    do_out.copy_(dx)
    dz_out.copy_(dg)
    dw_out.copy_(dw.float())
    # fla's recomputed output is the PRE-GATE norm rms(o)*w; compose the gate
    y_out.copy_((y_norm.float() * torch.nn.functional.silu(z.float())).to(y_out.dtype))


register(
    "gated_rmsnorm_fwd", "fla-fused", deterministic=True, allocates="torch",
    workspace=internal(_fwd_hint), requires=_fla_available, priority=10,
    fn=_fla_fwd,
)
register(
    "gated_rmsnorm_bwd", "fla-fused", deterministic=True, allocates="torch",
    workspace=internal(_bwd_hint), requires=_fla_available, priority=10,
    fn=_fla_bwd,
)


def _eager_fwd(kctx, o, z, w, out, rstd_out):
    of = o.float()
    rstd_out.copy_(torch.rsqrt(of.pow(2).mean(-1) + ops.RMS_EPS))
    out.copy_(ops.gated_rmsnorm_reference(o, z, w))


def _eager_bwd(kctx, dy, o, z, w, rstd, do_out, dz_out, dw_out, y_out):
    with torch.enable_grad():
        o_l = o.detach().requires_grad_()
        z_l = z.detach().requires_grad_()
        w_l = w.detach().requires_grad_()
        y = ops.gated_rmsnorm_reference(o_l, z_l, w_l)
    do, dz, dw = torch.autograd.grad(y, (o_l, z_l, w_l), grad_outputs=dy)
    do_out.copy_(do)
    dz_out.copy_(dz)
    dw_out.copy_(dw.float())
    y_out.copy_(y.detach())


register(
    "gated_rmsnorm_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_fwd_hint), priority=0, fn=_eager_fwd,
)
register(
    "gated_rmsnorm_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_bwd_hint), priority=0, fn=_eager_bwd,
)
