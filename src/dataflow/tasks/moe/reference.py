"""Autograd-able reference forms for the MoE MLP — the correctness anchor.

Three consumers, one source of truth:
  1. ladder-1 op pins (every kernel fwd AND bwd vs these + autograd),
  2. the standalone module harness (full tail fwd/bwd vs autograd,
     family-independent),
  3. the family goldens (GoldenOlmoe / GoldenQwen35Moe compose
     ``moe_mlp_reference`` and autograd CE + aux).

Numerics conventions pinned here (the runtime kernels mirror them):
  - router logits are a bf16 GEMM; ALL routing math (softmax, top-k
    weights, aux loss) is fp32 from those bf16 logits;
  - tie-break = smallest expert index, realized by
    ``torch.sort(descending=True, stable=True)`` (stable descending keeps
    ascending index order among equals — probed; torch.topk violates it);
  - swiglu rounds silu to the storage dtype BEFORE the product
    (``ops.swiglu_fwd`` — reused, not reimplemented);
  - combine: routed contributions accumulate in fp32; the shared-expert
    output rounds to bf16 once, joins the residual in storage dtype, and
    the (base + routed) sum rounds once at the end:

        routed = sum_k w_k * (swiglu(h13_e) @ w2_e)          # fp32 acc
        base   = resid (+ shared_bf16 when n_shared_experts)  # bf16 add
        y      = (base.float() + routed).to(dtype)
"""
from __future__ import annotations

import torch

from .. import ops
from .spec import MoESpec


def moe_topk_reference(
    logits: torch.Tensor, top_k: int, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing weights + expert ids. Returns (weights fp32 (t,K), ids int64).

    Differentiable through ``weights`` (via the selected scores); ``ids``
    are discrete. Both modes share the smallest-index tie-break.
    """
    lf = logits.float()
    scores = torch.softmax(lf, dim=-1) if mode == "softmax_then_topk" else lf
    vals, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    ids = idx[:, :top_k]
    sel = vals[:, :top_k]
    if mode == "topk_then_softmax":
        return torch.softmax(sel, dim=-1), ids
    return sel, ids


def moe_aux_loss_reference(
    logits: torch.Tensor, ids: torch.Tensor, *, n_experts: int, aux_coef: float
) -> torch.Tensor:
    """Load-balance loss L = alpha * E * sum_e f_e * pbar_e (fp32 scalar).

    f_e = counts_e / (T*K) comes from the DISCRETE assignments (detached
    by construction — the int path); pbar is the mean full-E fp32 softmax,
    so autograd of this expression reproduces the runtime's analytic
    injected gradient exactly: dL/dz = (alpha*E/T) * p * (f - <f, p>).
    """
    t, k = ids.shape
    counts = torch.bincount(ids.reshape(-1).long(), minlength=n_experts)
    f = counts.float() / (t * k)
    p = torch.softmax(logits.float(), dim=-1)
    return aux_coef * n_experts * (f * p.mean(0)).sum()


def moe_mlp_reference(
    h2: torch.Tensor,
    w: dict,
    moe: MoESpec,
    resid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full MoE tail: router -> experts -> combine (+ shared expert).

    ``h2`` is the post-ffn-norm activation, ``resid`` the post-attention
    residual stream. Returns (y, aux) with y = block output (residual
    included, per the pinned combine convention) and aux the fp32
    load-balance loss scalar (zeros(()) when aux_coef == 0).

    Masked E-loop (autograd-able, tiny-scale only). Honors partial
    ownership: only locally-held experts contribute — the EP accounting
    tests compare a sharded experts stage against this restricted form.
    """
    f = moe.d_ff_expert
    logits = h2 @ w["w_router"]                      # bf16 (t, E) — runtime path
    route_w, ids = moe_topk_reference(logits, moe.top_k, moe.routing_mode)
    if moe.aux_coef > 0:
        aux = moe_aux_loss_reference(
            logits, ids, n_experts=moe.n_experts, aux_coef=moe.aux_coef
        )
    else:
        aux = torch.zeros((), dtype=torch.float32, device=h2.device)

    local_ids = moe.expert_ids or tuple(range(moe.n_experts))
    routed = torch.zeros(h2.shape, dtype=torch.float32, device=h2.device)
    for slot, e in enumerate(local_ids):
        coef = (route_w * (ids == e)).sum(-1)        # (t,) fp32; <=1 hit per row
        h13_e = h2 @ w["w13_experts"][slot]          # bf16 (t, 2F)
        act = ops.swiglu_fwd(h13_e[:, :f], h13_e[:, f:])
        routed = routed + coef[:, None] * (act @ w["w2_experts"][slot]).float()

    base = resid
    if moe.n_shared_experts:
        fs = moe.d_ff_shared
        s13 = h2 @ w["w_s13"]
        s_act = ops.swiglu_fwd(s13[:, :fs], s13[:, fs:])
        sh_each = s_act @ w["w_s2"]                  # (t, d) bf16
        gate = torch.sigmoid((h2 @ w["w_shared_gate"]).float())  # (t, S=1) fp32
        base = resid + (gate * sh_each.float()).to(resid.dtype)

    return (base.float() + routed).to(h2.dtype), aux
