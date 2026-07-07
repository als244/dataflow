"""muon_step: Nesterov momentum + quintic Newton-Schulz orthogonalized
update, ported from flextrain (refs/flextrain/flextrain/ops/_kernels/
muon.py — the reference Shein pointed at) and adapted to the registry
ABI + batched rank-3 expert stacks.

Faithful semantics kept from the port:
- momentum ARITHMETIC in the momentum buffer's dtype (bf16 under the
  default opt-dtype policy — grad is cast first, exactly as flextrain);
- nesterov: orthogonalize (g + beta * m_new);
- per-matrix Frobenius normalization with eps INSIDE the add;
- addmm/baddbmm-fused quintic NS (a=3.4445, b=-4.775, c=2.0315, 5 its);
- Moonshot scale factor 0.2 * sqrt(max(rows, cols));
- decoupled weight decay on the param before the update.

Extension over the port: a leading batch dim (E, r, c) runs NS per
expert slice in one batched pass (norms/matmuls batch; scale is shape-
uniform across slices). The GEMMs are the work — cuBLAS IS the fused
kernel — so there is one implementation, aten.
"""
from __future__ import annotations

import torch

from .registry import none, register

NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_ITERS = 5


def ns_orthogonalize_batched(g: torch.Tensor, *, eps: float = 1e-8,
                             iters: int = NS_ITERS) -> torch.Tensor:
    """(B, r, c) -> per-slice approximate UV^T, flextrain NS loop."""
    x = g
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    norm = x.norm(dim=(-2, -1), keepdim=True)
    x = x / norm.add_(eps)
    for _ in range(iters):
        a = x @ x.mT
        b = torch.baddbmm(a, a, a, beta=NS_B, alpha=NS_C)
        x = torch.baddbmm(x, b, x, beta=NS_A, alpha=1.0)
    return x.mT if transposed else x


@register("muon_step", "aten", deterministic=True, workspace=none(),
          priority=10)
def _muon_step_aten(kctx, w, g, m, *, shape, lr, beta, eps,
                    weight_decay):
    gm = g.to(m.dtype)                      # flextrain: bf16 momentum math
    m.mul_(beta).add_(gm)
    eff = gm.add(m, alpha=beta)             # nesterov
    eff3 = eff.view(shape if len(shape) == 3 else (1, *shape))
    o = ns_orthogonalize_batched(eff3.float(), eps=eps)
    if weight_decay:
        w32 = w.float().mul_(1.0 - lr * weight_decay)
        w.copy_(w32.to(w.dtype))
    scale = 0.2 * max(shape[-2], shape[-1]) ** 0.5
    w.add_(o.reshape(w.shape).to(w.dtype), alpha=-lr * scale)
