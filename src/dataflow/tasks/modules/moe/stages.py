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
    # metadata families write the discrete decision into the M views
    # (st["aux_temp"]); ctx families keep the historical a-destination.
    # router_logits is NOT a decision — it stays in the ctx either way.
    aux_temp = st.get("aux_temp")
    dst = aux_temp if aux_temp is not None else a
    if a is not None and "router_logits" in a:
        logits = a["router_logits"]
        torch.matmul(h2, w["w_router"], out=logits)
    else:
        logits = h2 @ w["w_router"]
    if dst is not None:
        route_w, route_ids = dst["route_w"], dst["route_ids"]
    else:
        route_w = torch.empty(
            (d.tokens, moe.top_k), dtype=torch.bfloat16, device=h2.device
        )
        route_ids = torch.empty(
            (d.tokens, moe.top_k), dtype=torch.int32, device=h2.device
        )
    if st.get("aux_temp_ready"):
        # recompute with the decision supplied: NEVER re-select
        st.update(logits=logits, route_w=route_w, route_ids=route_ids)
        return
    if moe.routing_mode == "sigmoid_noaux_tc":
        K.moe_topk_sigmoid_noaux(
            kctx, logits, w["w_router_bias"], route_w, route_ids,
            top_k=moe.top_k, n_group=moe.n_group, topk_group=moe.topk_group,
            routed_scaling=moe.routed_scaling,
        )
    else:
        K.moe_topk_softmax(
            kctx, logits, route_w, route_ids, top_k=moe.top_k, mode=moe.routing_mode
        )
    st.update(logits=logits, route_w=route_w, route_ids=route_ids)


def stage_moe_dispatch(kctx, K, d, st):
    moe, a = _spec(d), st["a"]
    h2 = st["h2"]
    rows = moe_local_rows(moe, d.tokens)
    aux_temp = st.get("aux_temp")
    dst = aux_temp if aux_temp is not None else a
    if dst is not None:
        order, offsets = dst["route_order"], dst["route_offsets"]
    else:
        order = torch.empty(rows, dtype=torch.int32, device=h2.device)
        offsets = torch.empty(
            moe.n_local_experts + 1, dtype=torch.int32, device=h2.device
        )
    if not st.get("aux_temp_ready"):
        K.moe_sort(kctx, st["route_ids"], order, offsets, n_experts=moe.n_experts)
    xp = torch.empty((rows, d.d_model), dtype=torch.bfloat16, device=h2.device)
    K.moe_dispatch_fwd(kctx, h2, order, xp, top_k=moe.top_k)
    st.update(
        order=order, offsets=offsets,
        slot_of=_slot_of_from_order(order, d.tokens, moe.top_k), xp=xp,
    )


def stage_moe_experts13(kctx, K, d, st):
    a = st["a"]
    xp = st.pop("xp")
    if a is not None and "h13" in a:
        # ctx write-through: the aten path pays one copy into the view
        h13 = a["h13"]
        K.moe_grouped_mm_fwd(kctx, xp, st["w"]["w13_experts"], st["offsets"], h13)
    else:
        # scratch destination: dual-mode return skips the copy + duplicate
        h13 = K.moe_grouped_mm_fwd(kctx, xp, st["w"]["w13_experts"], st["offsets"])
    st["h13"] = h13


def stage_moe_shared(kctx, K, d, st):
    moe, a, w = _spec(d), st["a"], st["w"]
    h2 = st["h2"]
    if a is not None and "s13" in a:
        s13, gate_pre = a["s13"], a["gate_pre"]
        torch.matmul(h2, w["w_s13"], out=s13)
        torch.matmul(h2, w["w_shared_gate"], out=gate_pre)
    else:
        s13 = h2 @ w["w_s13"]
        gate_pre = h2 @ w["w_shared_gate"]
    st.update(s13=s13, gate_pre=gate_pre)


def stage_moe_shared_nogate(kctx, K, d, st):
    """DeepSeek-V3 flavor: plain additive shared expert — no gate
    projection, no gate_pre ctx field (separate stage fn because the
    emitted-fields tuples are STATIC declarations)."""
    a, w = st["a"], st["w"]
    h2 = st["h2"]
    if a is not None and "s13" in a:
        s13 = a["s13"]
        torch.matmul(h2, w["w_s13"], out=s13)
    else:
        s13 = h2 @ w["w_s13"]
    st["s13"] = s13


def stage_moe_experts2_combine(kctx, K, d, st):
    moe, w = _spec(d), st["w"]
    h13 = st.pop("h13")
    rows, f = h13.shape[0], moe.d_ff_expert
    sact = torch.empty((rows, f), dtype=torch.bfloat16, device=h13.device)
    K.swiglu_packed_fwd(kctx, h13, sact)
    yp = K.moe_grouped_mm_fwd(kctx, sact, w["w2_experts"], st["offsets"])
    del sact

    base = st.pop("h_mid")
    if moe.n_shared_experts:
        fs = moe.d_ff_shared
        s13 = st.pop("s13")
        s_act = torch.empty((d.tokens, fs), dtype=torch.bfloat16, device=h13.device)
        K.swiglu_packed_fwd(kctx, s13, s_act)
        sh = s_act @ w["w_s2"]
        del s_act
        if moe.shared_gate:
            sig = torch.sigmoid(st.pop("gate_pre").float()).reshape(-1).contiguous()
            # sigma-gate as an in-place row scale (no (t,d) fp32 materialization)
            K.moe_scale_rows(kctx, sh, sig)
            del sig
        # base is (or aliases) a ctx view — never mutated in place
        base = base + sh
        del sh

    K.moe_combine_fwd(kctx, yp, st["slot_of"], st["route_w"], base, st["y"])
    for key in ("h2", "logits", "route_w", "route_ids", "order", "offsets", "slot_of"):
        st.pop(key)


MOE_STAGES = (
    # the routing DECISION (route_w/ids/order/offsets) is M-object
    # metadata, not ctx — only the recomputable logits are declared here
    ("moe_route", stage_moe_route, ("router_logits",)),
    ("moe_dispatch", stage_moe_dispatch, ()),
    ("moe_experts13", stage_moe_experts13, ("h13",)),
    ("moe_experts2_combine", stage_moe_experts2_combine, ()),
)

MOE_SHARED_STAGES = (
    MOE_STAGES[:3]
    + (("moe_shared", stage_moe_shared, ("s13", "gate_pre")),)
    + MOE_STAGES[3:]
)

# DeepSeek-V3 flavor: ungated additive shared expert (spec.shared_gate=False)
MOE_SHARED_NOGATE_STAGES = (
    MOE_STAGES[:3]
    + (("moe_shared", stage_moe_shared_nogate, ("s13",)),)
    + MOE_STAGES[3:]
)


# --- backward tail ---------------------------------------------------------------


def moe_mlp_tail_bwd(kctx, K, d, dy, a, w, dw, accum, acc, norm_bwd, *, resid_field):
    """MoE analog of dense_mlp_tail_bwd. Returns the residual-stream
    gradient WITH ``dy`` added. ``dw``/``accum`` are needed beyond the
    ``acc`` closure because the grouped wgrads write their stacked expert
    fields directly (create-vs-accumulate handled inside the op, same
    bf16-round-then-add convention).

    Workspace discipline (beyond del-at-last-use): NO multi-GiB fp32
    materializations — the route-weight scalings run IN PLACE via
    ``moe_scale_rows`` (dyp/dsact hold the RAW values until their raw
    consumers ran, then become the scaled values in the same bytes), the
    dprob dot is a fused rowdot, grouped dgrads use the dual-mode
    return form (no duplicate buffer + copy pass), and dh2 accumulates in
    bf16 via addmm_ — the dense tail's convention.
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
    dyp = torch.empty_like(xp)                      # RAW until scaled below
    K.moe_dispatch_fwd(kctx, dy, order, dyp, top_k=topk)

    sact = torch.empty((rows, f), dtype=torch.bfloat16, device=dy.device)
    K.swiglu_packed_fwd(kctx, a["h13"], sact)
    dsact = K.moe_grouped_mm_dgrad(kctx, dyp, w["w2_experts"], offsets)  # RAW

    # dL/d route_w at slot j: <dy[token(j)], yp_j> == <dsact_raw_j, sact_j>
    # (the F-dim dot — no yp recompute, no (rows,F) product tensor)
    dprob_slot = torch.empty(rows, dtype=torch.float32, device=dy.device)
    K.moe_rowdot(kctx, dsact, sact, dprob_slot)
    dprob = dprob_slot[slot_of.view(-1).long()].view(t, topk).contiguous()
    del dprob_slot

    srw = a["route_w"].reshape(-1)[order.long()].float().contiguous()  # (rows,)
    K.moe_scale_rows(kctx, dyp, srw)                # raw -> scaled, in place
    if dw is not None and "w2_experts" in dw:       # frozen: no storage, skip
        K.moe_grouped_mm_wgrad(kctx, sact, dyp, offsets, dw["w2_experts"], accumulate=accum)
    del sact, dyp
    K.moe_scale_rows(kctx, dsact, srw)              # raw -> scaled, in place
    del srw

    dh13 = torch.empty((rows, 2 * f), dtype=torch.bfloat16, device=dy.device)
    K.swiglu_packed_bwd(kctx, dsact, a["h13"], dh13)
    del dsact
    if dw is not None and "w13_experts" in dw:
        K.moe_grouped_mm_wgrad(kctx, xp, dh13, offsets, dw["w13_experts"], accumulate=accum)
    del xp
    dxp = K.moe_grouped_mm_dgrad(kctx, dh13, w["w13_experts"], offsets)
    del dh13

    # bf16 residual-stream accumulator + addmm_ joins: the dense tail's
    # convention (dh2 = dx1@w1.T; dh2.addmm_(...)), no fp32 copy at the end
    dh2 = torch.empty((t, d.d_model), dtype=torch.bfloat16, device=dy.device)
    K.moe_dispatch_bwd(kctx, dxp, slot_of, dh2)
    del dxp, slot_of

    # router backward + per-round aux load-balance injection
    noaux = moe.routing_mode == "sigmoid_noaux_tc"
    dlogits = torch.empty((t, moe.n_experts), dtype=torch.float32, device=dy.device)
    if noaux:
        K.moe_router_bwd_sigmoid(
            kctx, dprob, a["route_w"], a["route_ids"], a["router_logits"], dlogits,
        )
        # per-round expert counts ride the bias's dW slot (fp32 via the
        # family's dtype-policy override): grad-accum aggregation for
        # free; the optimizer's per-field special applies the V3 sign
        # rule to the STEP total — AdamW math never touches the bias
        if dw is not None and "w_router_bias" in dw:
            cnt = (offsets[1:] - offsets[:-1]).to(torch.float32)
            if accum:
                dw["w_router_bias"].add_(cnt)
            else:
                dw["w_router_bias"].copy_(cnt)
        if moe.aux_coef > 0:
            # per-round aux load-balance needs per-sequence bounds: the round's
            # Segments (merged into `a` by the bwd launch), never d.seq_spec
            K.moe_seq_aux_grad(
                kctx, a["router_logits"], a["route_ids"], dlogits,
                alpha=moe.aux_coef, top_k=topk,
                seq_bounds=tuple(a["_seg"].bounds),
            )
        del cnt
    else:
        K.moe_router_bwd(
            kctx, dprob, a["route_w"], a["route_ids"], a["router_logits"], dlogits,
            mode=moe.routing_mode,
        )
        if moe.aux_coef > 0:
            counts = (offsets[1:] - offsets[:-1]).contiguous()
            K.moe_aux_lb_grad(
                kctx, a["router_logits"], counts, dlogits,
                alpha=moe.aux_coef, top_k=topk,
            )
            del counts
    del dprob
    dlogits_bf = dlogits.to(torch.bfloat16)
    del dlogits
    if acc.wanted("w_router"):
        acc("w_router", h2.T @ dlogits_bf)
    dh2.addmm_(dlogits_bf, w["w_router"].T)
    del dlogits_bf

    if moe.n_shared_experts and moe.shared_gate:
        fs = moe.d_ff_shared
        s_act = torch.empty((t, fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_fwd(kctx, a["s13"], s_act)
        sh_each = s_act @ w["w_s2"]
        sig = torch.sigmoid(a["gate_pre"].float()).reshape(-1).contiguous()  # (t,)
        d_gate_row = torch.empty(t, dtype=torch.float32, device=dy.device)
        K.moe_rowdot(kctx, dy, sh_each, d_gate_row)     # <dy, sh_each> per token
        del sh_each
        d_sh = dy.clone()
        K.moe_scale_rows(kctx, d_sh, sig)               # dy * sigma, in place
        if acc.wanted("w_s2"):
            acc("w_s2", s_act.T @ d_sh)
        del s_act
        d_gate = (d_gate_row * sig * (1.0 - sig)).to(torch.bfloat16).unsqueeze(1)
        del d_gate_row, sig
        if acc.wanted("w_shared_gate"):
            acc("w_shared_gate", h2.T @ d_gate)
        ds_act = d_sh @ w["w_s2"].T
        del d_sh
        ds13 = torch.empty((t, 2 * fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_bwd(kctx, ds_act, a["s13"], ds13)
        del ds_act
        if acc.wanted("w_s13"):
            acc("w_s13", h2.T @ ds13)
        dh2.addmm_(ds13, w["w_s13"].T)
        dh2.addmm_(d_gate, w["w_shared_gate"].T)
        del ds13, d_gate
    elif moe.n_shared_experts:
        # ungated (DeepSeek-V3): shared output feeds the residual directly,
        # so d_shared = dy verbatim — no gate, no rowdot, no scaling
        fs = moe.d_ff_shared
        s_act = torch.empty((t, fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_fwd(kctx, a["s13"], s_act)
        if acc.wanted("w_s2"):
            acc("w_s2", s_act.T @ dy)
        del s_act
        ds_act = dy @ w["w_s2"].T
        ds13 = torch.empty((t, 2 * fs), dtype=torch.bfloat16, device=dy.device)
        K.swiglu_packed_bwd(kctx, ds_act, a["s13"], ds13)
        del ds_act
        if acc.wanted("w_s13"):
            acc("w_s13", h2.T @ ds13)
        dh2.addmm_(ds13, w["w_s13"].T)
        del ds13
    del h2

    dh_mid, dffn = norm_bwd(dh2, resid, a["rstd_ffn"], w["ffn_norm_w"])
    del dh2
    acc("ffn_norm_w", dffn)
    dh_mid.add_(dy)
    return dh_mid


def moe_bias_update(kctx, kernels, w_view, g_view, *, speed: float) -> None:
    """DeepSeek-V3 balance-bias rule, applied by the family's AdamW
    per-field special (update_specials={"w_router_bias": ...}) in place of
    AdamW math: b_e += speed * sign(mean_load - load_e), on the STEP's
    aggregate counts (the bwd tail accumulated per-round counts into the
    bias's dW slot). fp32 in and out (dtype-policy override); plain torch
    (2 elementwise ops on (E,)) — async, deterministic, not worth a
    registry op."""
    counts = g_view.float()
    w_view.add_(torch.sign(counts.mean() - counts).to(w_view.dtype), alpha=speed)


# --- profiling support ------------------------------------------------------------


class MoEMetaState:
    """Metadata-object plumbing for pure-MoE families: the layer's M
    holds the discrete routing pack (moe_aux_temp_specs). Exposed to stages
    as st["aux_temp"]; recompute sets aux_temp_ready (the moe stages then skip
    topk + sort and consume the decision verbatim — METADATA IS NEVER
    RECOMPUTED). Backward merges the M views into `a` so every
    downstream read (a["route_*"]) is unchanged."""

    def _aux_temp_state(self, ctx):
        from .spec import moe_aux_temp_layout

        layout = moe_aux_temp_layout(self.dims, _spec(self.dims))
        key = ctx.task.compute_block_key
        if key.endswith(("_recompute", "_bwd")):
            for j, oid in enumerate(ctx.task.inputs):
                if oid.startswith("AuxTemp_"):
                    st = {"aux_temp": layout.views(self._in(ctx, j))}
                    if key.endswith("_recompute"):
                        st["aux_temp_ready"] = True
                    return st
            raise RuntimeError(f"no M_ input on {ctx.task.id}")
        for j, o in enumerate(ctx.task.outputs):
            if o.id.startswith("AuxTemp_"):
                return {"aux_temp": layout.views(self._out(ctx, j))}
        raise RuntimeError(f"no M_ output on {ctx.task.id}")

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum, aux_temp=None):
        if aux_temp:
            a = {**a, **aux_temp["aux_temp"]}
        super()._backward(kctx, dy, a, x, w, dx_out, dw, accum)


class MoEProfileFill:
    """Mixin for MoE block executables: deterministic buffer seeding for
    the profiling harness (training/profiling.py calls
    ``profile_fill(ctx)`` once per signature, before workspace/timing
    launches). Float inputs get small seeded pseudo-random values
    (routing near-balanced and REPRODUCIBLE); the M_ metadata INPUT
    (bwd/recompute signatures) gets VALID balanced routing — garbage
    there is an illegal memory access in the gathers, not a bias.
    M-era: the routing pack lives in the M object, never the ctx."""

    def profile_fill(self, ctx) -> None:
        from ...interop import torch_view
        from .spec import moe_aux_temp_layout

        key = ctx.task.compute_block_key
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        for oid in ctx.task.inputs:
            if oid.startswith("AuxTemp_"):
                continue
            b = ctx.inputs[oid]
            n = b.size_bytes // 2
            v = torch_view(b, (n,), torch.bfloat16)
            v.copy_(
                torch.rand(n, generator=gen, dtype=torch.bfloat16, device="cuda")
                .sub_(0.5).mul_(0.05)
            )

        moe = _spec(self.dims)
        layout = moe_aux_temp_layout(self.dims, moe)
        t, topk = self.dims.tokens, moe.top_k
        for oid in ctx.task.inputs:
            if not oid.startswith("AuxTemp_"):
                continue
            m = layout.views(ctx.inputs[oid])
            rows = m["route_order"].shape[0]
            flat_ids = (
                torch.arange(rows, dtype=torch.int64, device="cuda") % moe.n_experts
            )
            m["route_ids"].copy_(flat_ids.view(t, topk).to(torch.int32))
            m["route_order"].copy_(torch.argsort(flat_ids, stable=True).to(torch.int32))
            counts = torch.bincount(flat_ids, minlength=moe.n_experts)
            m["route_offsets"][:1].zero_()
            m["route_offsets"][1:].copy_(counts.cumsum(0).to(torch.int32))
            m["route_w"].fill_(1.0 / topk)
