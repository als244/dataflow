"""layernorm family (gpt2): eager forms over the shared op library.

Op signatures (d = row width; all tensors contiguous):
- ``layernorm_fwd(kctx, x, w, b, out, mean_out, rstd_out)``: fp32 mean/rstd
  saved; normalized value rounds to storage dtype BEFORE the affine (the
  rmsnorm-family convention).
- ``layernorm_apply(kctx, x, mean, rstd, w, b, out)``: recompute from saved
  statistics.
- ``layernorm_bwd(kctx, dy, x, mean, rstd, w, dx_out, dw_out, db_out)``:
  dw/db fp32-accumulated, deterministic torch reductions (row-chunked).
"""
from __future__ import annotations

import torch

from ..blocks import ops
from .registry import internal, register


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
