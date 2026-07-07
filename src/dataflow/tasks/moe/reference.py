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
    logits: torch.Tensor, top_k: int, mode: str, *,
    bias: torch.Tensor | None = None,
    n_group: int = 1, topk_group: int = 1, routed_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing weights + expert ids. Returns (weights fp32 (t,K), ids int64).

    Differentiable through ``weights`` (via the selected scores); ``ids``
    are discrete. ALL modes share the smallest-index tie-break (expert and
    group level).

    sigmoid_noaux_tc (DeepSeek-V3): scores = sigmoid(logits); SELECTION on
    (scores + bias) under the group limit — groups ranked by the sum of
    their top-2 selection scores, only the best topk_group groups stay
    eligible — then greedy top-K; WEIGHTS = the selected RAW sigmoid
    scores renormalized to sum 1, x routed_scaling. bias enters selection
    ONLY (detached by construction: it feeds sort indices, never values).
    """
    lf = logits.float()
    if mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(lf)                               # (t, E) fp32
        t, e = scores.shape
        with torch.no_grad():
            sel = scores + (bias if bias is not None else 0.0)   # selection scores
            g = sel.view(t, n_group, e // n_group)
            g_sorted, _ = torch.sort(g, dim=-1, descending=True, stable=True)
            group_score = g_sorted[..., : min(2, g.shape[-1])].sum(-1)  # (t, n_group)
            _, g_idx = torch.sort(group_score, dim=-1, descending=True, stable=True)
            keep_groups = g_idx[:, :topk_group]                  # (t, topk_group)
            group_mask = torch.zeros(t, n_group, dtype=torch.bool, device=lf.device)
            group_mask.scatter_(1, keep_groups, True)
            expert_mask = group_mask.repeat_interleave(e // n_group, dim=1)
            masked = sel.masked_fill(~expert_mask, float("-inf"))
            _, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
            ids = idx[:, :top_k]
        picked = scores.gather(1, ids)                           # raw sigmoid scores
        w = picked / picked.sum(-1, keepdim=True) * routed_scaling
        return w, ids
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


def moe_seq_aux_loss_reference(
    logits: torch.Tensor, ids: torch.Tensor, *,
    n_experts: int, top_k: int, aux_coef: float,
    seq_lens: tuple[int, ...],
) -> torch.Tensor:
    """DeepSeek-V3 complementary SEQUENCE-WISE balance loss (fp32 scalar).

    Per sequence s (T_s tokens): L_s = alpha * sum_e f_e * P_e with
    f_e = (E/(K*T_s)) * count_e^s from the DISCRETE assignments (detached
    int path) and P_e = mean over the sequence's tokens of the per-token
    NORMALIZED sigmoid scores s'_te = sigmoid(z)_te / sum_e sigmoid(z)_te.
    Gradient flows through P only — autograd of this expression IS the
    runtime's injected form. Total = sum over sequences.
    """
    lf = logits.float()
    s = torch.sigmoid(lf)
    sn = s / s.sum(-1, keepdim=True)
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    lo = 0
    for t_s in seq_lens:
        hi = lo + t_s
        counts = torch.bincount(ids[lo:hi].reshape(-1).long(), minlength=n_experts)
        f = counts.float() * n_experts / (top_k * t_s)
        total = total + aux_coef * (f * sn[lo:hi].mean(0)).sum()
        lo = hi
    return total


def moe_mlp_reference(
    h2: torch.Tensor,
    w: dict,
    moe: MoESpec,
    resid: torch.Tensor,
    *,
    route_ids: torch.Tensor | None = None,
    seq_lens: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full MoE tail: router -> experts -> combine (+ shared expert).

    ``h2`` is the post-ffn-norm activation, ``resid`` the post-attention
    residual stream. Returns (y, aux) with y = block output (residual
    included, per the pinned combine convention) and aux the fp32
    load-balance loss scalar (zeros(()) when aux_coef == 0).

    ``route_ids`` pins the DISCRETE expert selection while the routing
    weights stay differentiable functions of this model's own logits.
    Block-level gradcheck ladders use it: near-tie top-k selections flip
    between two numerically-different-but-correct forwards (kernel vs
    reference attention paths), and since selection is non-differentiable,
    the correct gradient comparison conditions both sides on the SAME
    selection. End-to-end gates run selection-free.

    Masked E-loop (autograd-able, tiny-scale only). Honors partial
    ownership: only locally-held experts contribute — the EP accounting
    tests compare a sharded experts stage against this restricted form.
    """
    f = moe.d_ff_expert
    logits = h2 @ w["w_router"]                      # bf16 (t, E) — runtime path
    noaux = moe.routing_mode == "sigmoid_noaux_tc"
    if route_ids is None:
        route_w, ids = moe_topk_reference(
            logits, moe.top_k, moe.routing_mode,
            bias=(w["w_router_bias"].float() if noaux else None),
            n_group=moe.n_group, topk_group=moe.topk_group,
            routed_scaling=moe.routed_scaling,
        )
    else:
        ids = route_ids.long()
        lf = logits.float()
        if noaux:
            picked = torch.sigmoid(lf).gather(1, ids)
            route_w = picked / picked.sum(-1, keepdim=True) * moe.routed_scaling
        elif moe.routing_mode == "softmax_then_topk":
            route_w = torch.softmax(lf, dim=-1).gather(1, ids)
        else:
            route_w = torch.softmax(lf.gather(1, ids), dim=-1)
    if moe.aux_coef > 0 and noaux:
        aux = moe_seq_aux_loss_reference(
            logits, ids, n_experts=moe.n_experts, top_k=moe.top_k,
            aux_coef=moe.aux_coef,
            seq_lens=seq_lens if seq_lens is not None else (h2.shape[0],),
        )
    elif moe.aux_coef > 0:
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
        if moe.shared_gate:
            gate = torch.sigmoid((h2 @ w["w_shared_gate"]).float())  # (t, S=1) fp32
            base = resid + (gate * sh_each.float()).to(resid.dtype)
        else:
            base = resid + sh_each                   # V3: plain additive shared

    return (base.float() + routed).to(h2.dtype), aux
