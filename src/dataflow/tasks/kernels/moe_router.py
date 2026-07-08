"""MoE router ops: fused top-k+softmax, router backward, aux-loss gradient.

Op signatures (out params caller-provided; ``mode`` is a MoESpec
routing_mode string):

- ``moe_topk_softmax(kctx, logits (t,E) bf16, route_w_out (t,K) bf16,
                     route_ids_out (t,K) i32, *, top_k, mode)``
- ``moe_router_bwd(kctx, dprob (t,K) fp32, route_w (t,K) bf16,
                   route_ids (t,K) i32, logits (t,E) bf16,
                   dlogits_out (t,E) fp32, *, mode)``  — WRITES the full row
- ``moe_aux_lb_grad(kctx, logits (t,E) bf16, counts (E,) i32,
                    dlogits (t,E) fp32, *, alpha, top_k)`` — ACCUMULATES
      the load-balance gradient (alpha*E/T)*p*(f - <f,p>) in place; T is
      the per-round token count = logits rows, f = counts/(T*K) detached.

Semantics pinned by tests/modules/test_moe.py against
``dataflow.tasks.modules.moe.reference``: fp32 routing math from bf16 logits;
tie-break = SMALLEST expert index in both modes (torch.topk's CUDA
tie-break picks the larger index — that is why the eager path is
sort-based, not topk-based).

The fused Triton kernels are ports of flextrain's
``fused_topk_softmax_kernel`` / ``moe_router_gate_bwd_kernel`` /
``load_balance_bwd_kernel`` (refs/flextrain flextrain/ops/_kernels/moe.py),
adapted to this ABI: dprob arrives in original (t, K) order (no scatter
indirection), the router backward writes full rows instead of
accumulating, and MODE=0's softmax keeps the selected logits in registers
(fp32 end-to-end) instead of round-tripping them through the bf16 output
buffer. E is a compile-time constant (one specialization per model).
"""
from __future__ import annotations

import torch

from .registry import internal, none, register

_MODE_ID = {"topk_then_softmax": 0, "softmax_then_topk": 1}


# --- eager (reference composition; the numerics anchor) -----------------------


def _eager_topk_softmax(kctx, logits, route_w_out, route_ids_out, *, top_k, mode):
    from ..modules.moe.reference import moe_topk_reference

    w, ids = moe_topk_reference(logits, top_k, mode)
    route_w_out.copy_(w.to(route_w_out.dtype))
    route_ids_out.copy_(ids.to(route_ids_out.dtype))


def _eager_router_bwd(kctx, dprob, route_w, route_ids, logits, dlogits_out, *, mode):
    t, k = route_ids.shape
    e = dlogits_out.shape[1]
    ids = route_ids.long()
    dp = dprob.float()
    if mode == "topk_then_softmax":
        # softmax was over the K selected logits: dz_k = p_k (dp_k - <p, dp>)
        p = route_w.float()
        dz = p * (dp - (p * dp).sum(-1, keepdim=True))
        dlogits_out.zero_()
        dlogits_out.scatter_(1, ids, dz)  # ids unique per row -> deterministic
    else:
        # softmax over all E, top-K selected unnormalized:
        # dz[j] = -P[j] * sum_k dp_k P[c_k]  +  dp_k P[c_k] at j == c_k
        p_full = torch.softmax(logits.float(), dim=-1)
        topk_p = p_full.gather(1, ids)
        row_dot = (dp * topk_p).sum(-1, keepdim=True)
        dlogits_out.copy_(-p_full * row_dot)
        dlogits_out.scatter_add_(1, ids, dp * topk_p)


def _eager_aux_lb_grad(kctx, logits, counts, dlogits, *, alpha, top_k):
    t, e = logits.shape
    p = torch.softmax(logits.float(), dim=-1)
    f = counts.float() / (t * top_k)
    scale = alpha * e / t
    dlogits.add_(scale * p * (f.unsqueeze(0) - (p @ f).unsqueeze(1)))


def _row_hint(*tensors) -> int:
    lg = tensors[0]
    return 4 * lg.shape[0] * lg.shape[1] * 4  # a few fp32 (t, E) temporaries


register("moe_topk_softmax", "eager", deterministic=True, allocates="torch",
         workspace=internal(_row_hint), priority=0, fn=_eager_topk_softmax)
register("moe_router_bwd", "eager", deterministic=True, allocates="torch",
         workspace=internal(_row_hint), priority=0, fn=_eager_router_bwd)
register("moe_aux_lb_grad", "eager", deterministic=True, allocates="torch",
         workspace=internal(_row_hint), priority=0, fn=_eager_aux_lb_grad)


# --- sigmoid_noaux_tc (DeepSeek-V3) — SEPARATE op names ------------------------
# The triton moe_topk_softmax/moe_router_bwd kernels dispatch on a runtime
# ``mode`` argument they do not implement for this family; distinct op
# names keep resolution honest (eager-only v1 — the sort-based eager
# router measured 0.29 ms at E=256/t=16k, off the critical path; triton
# ports are a filed follow-up if profiles ever disagree).


def _eager_topk_sigmoid_noaux(kctx, logits, bias, route_w_out, route_ids_out, *,
                              top_k, n_group, topk_group, routed_scaling):
    from ..modules.moe.reference import moe_topk_reference

    w, ids = moe_topk_reference(
        logits, top_k, "sigmoid_noaux_tc", bias=bias.float(),
        n_group=n_group, topk_group=topk_group, routed_scaling=routed_scaling,
    )
    route_w_out.copy_(w.to(route_w_out.dtype))
    route_ids_out.copy_(ids.to(route_ids_out.dtype))


def _eager_router_bwd_sigmoid(kctx, dprob, route_w, route_ids, logits,
                              dlogits_out):
    """w_j = c*s_j/S (c = routed_scaling = sum_j w_j, S = sum_j s_j over
    the K selected): dL/ds_i = (c*dp_i - D)/S with D = <dp, w>, then the
    sigmoid chain dz = dL/ds * s*(1-s). Selection and bias are detached
    (they fed sort indices only). Writes full rows (zeros off-selection)."""
    ids = route_ids.long()
    dp = dprob.float()
    w = route_w.float()
    s_sel = torch.sigmoid(logits.float()).gather(1, ids)         # (t, K)
    s_sum = s_sel.sum(-1, keepdim=True)
    c = w.sum(-1, keepdim=True)
    d_dot = (dp * w).sum(-1, keepdim=True)
    dz = ((c * dp - d_dot) / s_sum) * s_sel * (1.0 - s_sel)
    dlogits_out.zero_()
    dlogits_out.scatter_(1, ids, dz)  # ids unique per row -> deterministic


def _eager_seq_aux_grad(kctx, logits, route_ids, dlogits, *, alpha, top_k,
                        seq_bounds):
    """DeepSeek-V3 sequence-wise complementary aux, injected analytically:
    dL/ds'_te = alpha*f_e/T_s; through the per-token normalization
    s' = s/sum(s): dL/ds_i = (alpha/(T_s*sum_t)) * (f_i - <f, s'_t>);
    then the sigmoid chain. ``seq_bounds`` = ((lo, hi), ...) host ints
    (plan-time constants — never read from device). f from per-sequence
    counts of the DISCRETE assignments (detached int path)."""
    lf = logits.float()
    s = torch.sigmoid(lf)
    row_sum = s.sum(-1, keepdim=True)
    sn = s / row_sum
    e = logits.shape[1]
    ids_flat = route_ids.long()
    for lo, hi in seq_bounds:
        t_s = hi - lo
        counts = torch.zeros(e, dtype=torch.float32, device=logits.device)
        counts.scatter_add_(
            0, ids_flat[lo:hi].reshape(-1),
            torch.ones((hi - lo) * route_ids.shape[1], dtype=torch.float32,
                       device=logits.device),
        )
        f = counts * (e / (top_k * t_s))
        seg_sn = sn[lo:hi]
        coef = alpha / (t_s * row_sum[lo:hi])
        dl_ds = coef * (f.unsqueeze(0) - (seg_sn @ f).unsqueeze(1))
        dlogits[lo:hi].add_(dl_ds * s[lo:hi] * (1.0 - s[lo:hi]))


register("moe_topk_sigmoid_noaux", "eager", deterministic=True,
         allocates="torch", workspace=internal(_row_hint), priority=0,
         fn=_eager_topk_sigmoid_noaux)
register("moe_router_bwd_sigmoid", "eager", deterministic=True,
         allocates="torch", workspace=internal(_row_hint), priority=0,
         fn=_eager_router_bwd_sigmoid)
register("moe_seq_aux_grad", "eager", deterministic=True, allocates="torch",
         workspace=internal(_row_hint), priority=0, fn=_eager_seq_aux_grad)


# --- fused Triton --------------------------------------------------------------

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments
    triton = None

if triton is not None:

    @triton.jit
    def _topk_softmax_kernel(
        logits_ptr, w_ptr, ids_ptr,
        t,
        E: tl.constexpr, K: tl.constexpr,
        BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr,
        MODE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        tok = (pid * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        tmask = tok < t
        eoff = tl.arange(0, BLOCK_E).to(tl.int64)
        emask = eoff < E
        lmask = tmask[:, None] & emask[None, :]

        logits = tl.load(
            logits_ptr + tok[:, None] * E + eoff[None, :],
            mask=lmask, other=-float("inf"),
        ).to(tl.float32)

        if MODE == 1:
            row_max = tl.max(logits, axis=1)
            ex = tl.exp(logits - row_max[:, None])
            ex = tl.where(emask[None, :], ex, 0.0)
            remaining = ex / tl.sum(ex, axis=1)[:, None]
        else:
            remaining = logits

        # iterative top-K: max value, SMALLEST index among ties (pinned)
        chosen = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.int1)
        for k in range(K):
            mx = tl.max(remaining, axis=1)
            is_max = (remaining == mx[:, None]) & emask[None, :]
            cand = tl.where(is_max, eoff[None, :], BLOCK_E)
            # all-NaN rows (profiling garbage) have no max-match: clamp the
            # sentinel so stored ids are ALWAYS in [0, E) — never poisons
            # downstream sort/offsets
            eid = tl.minimum(tl.min(cand, axis=1), E - 1)
            tl.store(ids_ptr + tok * K + k, eid.to(tl.int32), mask=tmask)
            if MODE == 1:
                tl.store(w_ptr + tok * K + k,
                         mx.to(w_ptr.dtype.element_ty), mask=tmask)
            sel = eoff[None, :] == eid[:, None]
            chosen = chosen | sel
            remaining = tl.where(sel, -float("inf"), remaining)

        if MODE == 0:
            # fp32 softmax over the K selected logits, kept in registers
            # (never round-tripped through the bf16 output buffer)
            sel_logits = tl.where(chosen, logits, -float("inf"))
            smax = tl.max(sel_logits, axis=1)
            sex = tl.exp(sel_logits - smax[:, None])
            sex = tl.where(chosen, sex, 0.0)
            ssum = tl.sum(sex, axis=1)
            probs_full = sex / ssum[:, None]  # nonzero only at chosen slots
            for k in range(K):
                eid = tl.load(ids_ptr + tok * K + k, mask=tmask, other=0).to(tl.int64)
                hit = eoff[None, :] == eid[:, None]
                pk = tl.sum(tl.where(hit, probs_full, 0.0), axis=1)
                tl.store(w_ptr + tok * K + k,
                         pk.to(w_ptr.dtype.element_ty), mask=tmask)

    @triton.jit
    def _router_bwd_kernel(
        dprob_ptr, route_w_ptr, ids_ptr, logits_ptr, dlogits_ptr,
        t,
        E: tl.constexpr, K: tl.constexpr,
        BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
        MODE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        koff = tl.arange(0, BLOCK_K)
        kmask = koff < K
        eoff = tl.arange(0, BLOCK_E).to(tl.int64)
        emask = eoff < E

        dp = tl.load(dprob_ptr + pid * K + koff, mask=kmask, other=0.0).to(tl.float32)
        ids = tl.load(ids_ptr + pid * K + koff, mask=kmask, other=0).to(tl.int64)
        is_chosen = kmask[:, None] & (eoff[None, :] == ids[:, None])  # (K, E)

        if MODE == 0:
            p = tl.load(route_w_ptr + pid * K + koff, mask=kmask, other=0.0).to(tl.float32)
            dz_k = p * (dp - tl.sum(dp * p))
            dz_full = tl.sum(tl.where(is_chosen, dz_k[:, None], 0.0), axis=0)
        else:
            logits = tl.load(
                logits_ptr + pid * E + eoff, mask=emask, other=-float("inf")
            ).to(tl.float32)
            mx = tl.max(logits, axis=0)
            ex = tl.exp(logits - mx)
            ex = tl.where(emask, ex, 0.0)
            p_full = ex / tl.sum(ex, axis=0)
            topk_p = tl.sum(tl.where(is_chosen, p_full[None, :], 0.0), axis=1)
            row_dot = tl.sum(dp * topk_p)
            dz_full = -p_full * row_dot + tl.sum(
                tl.where(is_chosen, (dp * topk_p)[:, None], 0.0), axis=0
            )

        tl.store(dlogits_ptr + pid * E + eoff, dz_full, mask=emask)

    @triton.jit
    def _aux_lb_kernel(
        logits_ptr, counts_ptr, dlogits_ptr,
        t, scale, denom,
        E: tl.constexpr, BLOCK_E: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        eoff = tl.arange(0, BLOCK_E).to(tl.int64)
        emask = eoff < E

        logits = tl.load(
            logits_ptr + pid * E + eoff, mask=emask, other=-float("inf")
        ).to(tl.float32)
        mx = tl.max(logits, axis=0)
        ex = tl.exp(logits - mx)
        ex = tl.where(emask, ex, 0.0)
        p = ex / tl.sum(ex, axis=0)

        f = tl.load(counts_ptr + eoff, mask=emask, other=0).to(tl.float32) / denom
        grad = scale * p * (f - tl.sum(f * p, axis=0))

        ptrs = dlogits_ptr + pid * E + eoff
        cur = tl.load(ptrs, mask=emask, other=0.0)
        tl.store(ptrs, cur + grad, mask=emask)

    def _pow2(n: int) -> int:
        return triton.next_power_of_2(max(n, 1))

    def _check_router(logits, *rest):
        assert logits.is_cuda and logits.is_contiguous() and logits.dim() == 2
        for r in rest:
            assert r.is_cuda and r.is_contiguous()
        return logits.shape

    @register("moe_topk_softmax", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _topk_softmax(kctx, logits, route_w_out, route_ids_out, *, top_k, mode):
        t, e = _check_router(logits, route_w_out, route_ids_out)
        block_t = 8
        _topk_softmax_kernel[(triton.cdiv(t, block_t),)](
            logits, route_w_out, route_ids_out, t,
            E=e, K=top_k, BLOCK_E=_pow2(e), BLOCK_T=block_t,
            MODE=_MODE_ID[mode],
        )

    @register("moe_router_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _router_bwd(kctx, dprob, route_w, route_ids, logits, dlogits_out, *, mode):
        t, e = _check_router(logits, dprob, route_w, route_ids, dlogits_out)
        k = route_ids.shape[1]
        _router_bwd_kernel[(t,)](
            dprob, route_w, route_ids, logits, dlogits_out, t,
            E=e, K=k, BLOCK_E=_pow2(e), BLOCK_K=_pow2(k),
            MODE=_MODE_ID[mode],
        )

    @register("moe_aux_lb_grad", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _aux_lb(kctx, logits, counts, dlogits, *, alpha, top_k):
        t, e = _check_router(logits, counts, dlogits)
        _aux_lb_kernel[(t,)](
            logits, counts, dlogits, t,
            float(alpha) * e / t, float(t * top_k),
            E=e, BLOCK_E=_pow2(e),
        )
