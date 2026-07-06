"""Reusable MoE SwiGLU MLP tail: forward stages + backward, family-pluggable.

Module grammar (the EP-facing shape): route -> dispatch -> experts ->
combine. Families splice the exported stage tuples after their ffn-norm
stage and point ``_mlp_bwd`` at ``moe_mlp_tail_bwd`` — the 5-point plug-in
contract lives in docs/extending.md. Stage fns use the BlockFwd stage ABI
``(kctx, K, d, st)`` and read the family-invariant state keys ``st["h2"]``
(post-ffn-norm) and ``st["h_mid"]`` (post-attention residual); ``d.moe`` is
the layer's MoESpec.

Saved-context fields (see spec.moe_context_specs): the routing decision
(logits/route_w/route_ids/route_order/route_offsets) and the
pre-activations h13 (+ shared s13/gate_pre). Everything else (xp, yp,
sact, slot_of, dprob) is re-derived in backward from the saved order —
deterministically, so save-mode and recompute-mode backwards are
bit-identical (all ops single-owner, no atomics).

The combine convention (single source of truth, mirrored by
``reference.moe_mlp_reference``):

    routed = sum_k route_w[t,k] * yp[slot(t,k)]           # fp32 accumulate
    base   = h_mid (+ shared_bf16 when n_shared_experts)  # bf16 add
    y      = (base.float() + routed).to(bf16)             # one rounding

Contiguity contract: every tensor handed to a registry kernel here is
either a fresh contiguous allocation or a full packed-layout field view
(contiguous by construction). Packed [x1|x3] halves are only ever split
INSIDE swiglu_packed_* — never hand-sliced into other kernels.
"""
from __future__ import annotations

import hashlib

import torch

from .spec import MoESpec, moe_local_rows


def _spec(d) -> MoESpec:
    moe = getattr(d, "moe", None)
    if moe is None:
        raise TypeError("MoE stages require dims.moe: MoESpec")
    return moe


def _slot_of_from_order(order: torch.Tensor, t: int, top_k: int) -> torch.Tensor:
    """Inverse permutation (scatter form), derived — unique indices, so the
    scatter is deterministic."""
    rows = order.shape[0]
    slot_of = torch.empty((t, top_k), dtype=torch.int32, device=order.device)
    slot_of.view(-1).scatter_(
        0, order.long(), torch.arange(rows, dtype=torch.int32, device=order.device)
    )
    return slot_of


# --- forward stages -------------------------------------------------------------


def stage_moe_route(kctx, K, d, st):
    moe, a, w = _spec(d), st["a"], st["w"]
    h2 = st["h2"]
    if a is not None:
        logits, route_w, route_ids = a["router_logits"], a["route_w"], a["route_ids"]
        torch.matmul(h2, w["w_router"], out=logits)
    else:
        logits = h2 @ w["w_router"]
        route_w = torch.empty(
            (d.tokens, moe.top_k), dtype=torch.bfloat16, device=h2.device
        )
        route_ids = torch.empty(
            (d.tokens, moe.top_k), dtype=torch.int32, device=h2.device
        )
    K.moe_topk_softmax(
        kctx, logits, route_w, route_ids, top_k=moe.top_k, mode=moe.routing_mode
    )
    st.update(logits=logits, route_w=route_w, route_ids=route_ids)


def stage_moe_dispatch(kctx, K, d, st):
    moe, a = _spec(d), st["a"]
    h2 = st["h2"]
    rows = moe_local_rows(moe, d.tokens)
    if a is not None:
        order, offsets = a["route_order"], a["route_offsets"]
    else:
        order = torch.empty(rows, dtype=torch.int32, device=h2.device)
        offsets = torch.empty(
            moe.n_local_experts + 1, dtype=torch.int32, device=h2.device
        )
    K.moe_sort(kctx, st["route_ids"], order, offsets, n_experts=moe.n_experts)
    xp = torch.empty((rows, d.d_model), dtype=torch.bfloat16, device=h2.device)
    K.moe_dispatch_fwd(kctx, h2, order, xp, top_k=moe.top_k)
    st.update(
        order=order, offsets=offsets,
        slot_of=_slot_of_from_order(order, d.tokens, moe.top_k), xp=xp,
    )


def stage_moe_experts13(kctx, K, d, st):
    moe, a = _spec(d), st["a"]
    xp = st.pop("xp")
    if a is not None:
        h13 = a["h13"]
    else:
        h13 = torch.empty(
            (xp.shape[0], 2 * moe.d_ff_expert), dtype=torch.bfloat16, device=xp.device
        )
    K.moe_grouped_mm_fwd(kctx, xp, st["w"]["w13_experts"], st["offsets"], h13)
    st["h13"] = h13


def stage_moe_shared(kctx, K, d, st):
    moe, a, w = _spec(d), st["a"], st["w"]
    h2 = st["h2"]
    if a is not None:
        s13, gate_pre = a["s13"], a["gate_pre"]
        torch.matmul(h2, w["w_s13"], out=s13)
        torch.matmul(h2, w["w_shared_gate"], out=gate_pre)
    else:
        s13 = h2 @ w["w_s13"]
        gate_pre = h2 @ w["w_shared_gate"]
    st.update(s13=s13, gate_pre=gate_pre)


def stage_moe_experts2_combine(kctx, K, d, st):
    moe, w = _spec(d), st["w"]
    h13 = st.pop("h13")
    rows, f = h13.shape[0], moe.d_ff_expert
    sact = torch.empty((rows, f), dtype=torch.bfloat16, device=h13.device)
    K.swiglu_packed_fwd(kctx, h13, sact)
    yp = torch.empty((rows, d.d_model), dtype=torch.bfloat16, device=h13.device)
    K.moe_grouped_mm_fwd(kctx, sact, w["w2_experts"], st["offsets"], yp)
    del sact

    base = st.pop("h_mid")
    if moe.n_shared_experts:
        fs = moe.d_ff_shared
        s13 = st.pop("s13")
        s_act = torch.empty((d.tokens, fs), dtype=torch.bfloat16, device=h13.device)
        K.swiglu_packed_fwd(kctx, s13, s_act)
        sh_each = s_act @ w["w_s2"]
        del s_act
        sig = torch.sigmoid(st.pop("gate_pre").float())
        # base is (or aliases) a ctx view — never mutated in place
        base = base + (sig * sh_each.float()).to(base.dtype)
        del sh_each, sig

    K.moe_combine_fwd(kctx, yp, st["slot_of"], st["route_w"], base, st["y"])
    for key in ("h2", "logits", "route_w", "route_ids", "order", "offsets", "slot_of"):
        st.pop(key)


MOE_STAGES = (
    ("moe_route", stage_moe_route, ("router_logits", "route_w", "route_ids")),
    ("moe_dispatch", stage_moe_dispatch, ("route_order", "route_offsets")),
    ("moe_experts13", stage_moe_experts13, ("h13",)),
    ("moe_experts2_combine", stage_moe_experts2_combine, ()),
)

MOE_SHARED_STAGES = (
    MOE_STAGES[:3]
    + (("moe_shared", stage_moe_shared, ("s13", "gate_pre")),)
    + MOE_STAGES[3:]
)


# --- backward tail ---------------------------------------------------------------


def moe_mlp_tail_bwd(kctx, K, d, dy, a, w, dw, accum, acc, norm_bwd, *, resid_field):
    """MoE analog of dense_mlp_tail_bwd. Returns the residual-stream
    gradient WITH ``dy`` added. ``dw``/``accum`` are needed beyond the
    ``acc`` closure because the grouped wgrads write their stacked expert
    fields directly (create-vs-accumulate handled inside the op, same
    bf16-round-then-add convention).

    Scratch discipline: del each (rows, .) / (t, .) temporary at last use.
    """
    moe = _spec(d)
    t, topk, f = d.tokens, moe.top_k, moe.d_ff_expert

    resid = a[resid_field]
    h2 = torch.empty_like(resid)
    K.rmsnorm_apply(kctx, resid, a["rstd_ffn"], w["ffn_norm_w"], h2)

    order, offsets = a["route_order"], a["route_offsets"]
    rows = order.shape[0]
    slot_of = _slot_of_from_order(order, t, topk)

    # re-gather permuted inputs (never saved; deterministic pure gathers)
    xp = torch.empty((rows, d.d_model), dtype=torch.bfloat16, device=dy.device)
    K.moe_dispatch_fwd(kctx, h2, order, xp, top_k=topk)
    dyp_raw = torch.empty_like(xp)
    K.moe_dispatch_fwd(kctx, dy, order, dyp_raw, top_k=topk)

    sact = torch.empty((rows, f), dtype=torch.bfloat16, device=dy.device)
    K.swiglu_packed_fwd(kctx, a["h13"], sact)
    dsact_raw = torch.empty_like(sact)
    K.moe_grouped_mm_dgrad(kctx, dyp_raw, w["w2_experts"], offsets, dsact_raw)

    # dL/d route_w at slot j: <dy[token(j)], yp_j> == <dsact_raw_j, sact_j>
    # (the F-dim dot — no yp recompute needed)
    dprob_slot = (dsact_raw.float() * sact.float()).sum(-1)
    dprob = dprob_slot[slot_of.view(-1).long()].view(t, topk).contiguous()
    del dprob_slot

    srw = a["route_w"].reshape(-1)[order.long()].float().unsqueeze(1)  # (rows, 1)
    dyp = (dyp_raw.float() * srw).to(torch.bfloat16)
    del dyp_raw
    K.moe_grouped_mm_wgrad(kctx, sact, dyp, offsets, dw["w2_experts"], accumulate=accum)
    del sact, dyp
    dsact = (dsact_raw.float() * srw).to(torch.bfloat16)
    del dsact_raw, srw

    dh13 = torch.empty((rows, 2 * f), dtype=torch.bfloat16, device=dy.device)
    K.swiglu_packed_bwd(kctx, dsact, a["h13"], dh13)
    del dsact
    K.moe_grouped_mm_wgrad(kctx, xp, dh13, offsets, dw["w13_experts"], accumulate=accum)
    del xp
    dxp = torch.empty((rows, d.d_model), dtype=torch.bfloat16, device=dy.device)
    K.moe_grouped_mm_dgrad(kctx, dh13, w["w13_experts"], offsets, dxp)
    del dh13

    dh2 = torch.empty((t, d.d_model), dtype=torch.float32, device=dy.device)
    K.moe_dispatch_bwd(kctx, dxp, slot_of, dh2)
    del dxp, slot_of

    # router backward + per-round aux load-balance injection
    dlogits = torch.empty((t, moe.n_experts), dtype=torch.float32, device=dy.device)
    K.moe_router_bwd(
        kctx, dprob, a["route_w"], a["route_ids"], a["router_logits"], dlogits,
        mode=moe.routing_mode,
    )
    del dprob
    if moe.aux_coef > 0:
        counts = (offsets[1:] - offsets[:-1]).contiguous()
        K.moe_aux_lb_grad(
            kctx, a["router_logits"], counts, dlogits,
            alpha=moe.aux_coef, top_k=topk,
        )
        del counts
    dlogits_bf = dlogits.to(torch.bfloat16)
    del dlogits
    acc("w_router", h2.T @ dlogits_bf)
    dh2.add_(dlogits_bf @ w["w_router"].T)
    del dlogits_bf

    if moe.n_shared_experts:
        fs = moe.d_ff_shared
        s_act = torch.empty((t, fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_fwd(kctx, a["s13"], s_act)
        sh_each = s_act @ w["w_s2"]
        sig = torch.sigmoid(a["gate_pre"].float())
        d_sh = (dy.float() * sig).to(torch.bfloat16)
        acc("w_s2", s_act.T @ d_sh)
        del s_act
        d_gate = (
            (dy.float() * sh_each.float()).sum(-1, keepdim=True) * sig * (1 - sig)
        ).to(torch.bfloat16)
        del sh_each, sig
        acc("w_shared_gate", h2.T @ d_gate)
        ds_act = d_sh @ w["w_s2"].T
        del d_sh
        ds13 = torch.empty((t, 2 * fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_bwd(kctx, ds_act, a["s13"], ds13)
        del ds_act
        acc("w_s13", h2.T @ ds13)
        dh2.add_(ds13 @ w["w_s13"].T)
        dh2.add_(d_gate @ w["w_shared_gate"].T)
        del ds13, d_gate
    del h2

    dh2_bf = dh2.to(torch.bfloat16)
    del dh2
    dh_mid, dffn = norm_bwd(dh2_bf, resid, a["rstd_ffn"], w["ffn_norm_w"])
    del dh2_bf
    acc("ffn_norm_w", dffn)
    dh_mid.add_(dy)
    return dh_mid


# --- profiling support ------------------------------------------------------------


class MoEProfileFill:
    """Mixin for MoE block executables: deterministic buffer seeding for the
    profiling harness (training/profiling.py calls ``profile_fill(ctx)``
    once per signature, before the workspace/timing launches).

    Two jobs:
      1. float inputs get small seeded pseudo-random values — routing
         becomes near-balanced (multinomial, ~±1σ) and REPRODUCIBLE across
         cache refreshes (garbage/zero logits route everything to K experts,
         which is 4-30% faster per grouped op — an anti-conservative,
         allocator-history-dependent cost bias);
      2. saved-context int32 routing fields get VALID balanced routing
         (identity-consistent ids/order/offsets) — garbage there is an
         illegal memory access in the gathers, not a bias.
    """

    def profile_fill(self, ctx) -> None:
        from ..interop import torch_view

        key = ctx.task.compute_block_key
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        for oid in ctx.task.inputs:
            b = ctx.inputs[oid]
            n = b.size_bytes // 2
            v = torch_view(b, (n,), torch.bfloat16)
            v.copy_(
                torch.rand(n, generator=gen, dtype=torch.bfloat16, device="cuda")
                .sub_(0.5).mul_(0.05)
            )

        if not ctx.task.compute_block_key.endswith("_bwd"):
            return  # fwd/recompute derive routing live from the seeded floats

        moe = _spec(self.dims)
        t, topk = self.dims.tokens, moe.top_k
        a_buf = ctx.inputs[ctx.task.inputs[1]]  # (dy, A, x, W[, dW]) contract
        cl = self.cl
        ids = cl.view(a_buf, "route_ids")
        order = cl.view(a_buf, "route_order")
        offsets = cl.view(a_buf, "route_offsets")
        rows = order.shape[0]

        flat_ids = (
            torch.arange(rows, dtype=torch.int64, device="cuda") % moe.n_experts
        )
        ids.copy_(flat_ids.view(t, topk).to(torch.int32))
        order.copy_(torch.argsort(flat_ids, stable=True).to(torch.int32))
        counts = torch.bincount(flat_ids, minlength=moe.n_experts)
        offsets[:1].zero_()
        offsets[1:].copy_(counts.cumsum(0).to(torch.int32))
        cl.view(a_buf, "route_w").fill_(1.0 / topk)
