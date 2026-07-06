"""Ladder 1 for the pluggable MoE module: every op fwd AND bwd pinned
against dataflow.tasks.moe.reference + autograd, then the full tail
(stages + moe_mlp_tail_bwd) against moe_mlp_reference — family-independent.

Pinned contracts:
- tie-break = smallest expert index in BOTH routing modes (torch.topk's
  CUDA tie-break returns the larger index — the eager path is sort-based
  for exactly this reason; crafted-tie rows make the pin load-bearing);
- swiglu_packed == unpacked swiglu bit-for-bit on the same values;
- grouped GEMMs match a dense per-segment loop with uneven counts,
  ZERO-row experts, non-tile-multiple counts; create-mode wgrad
  zero-fills empty experts; accumulate-mode adds at grad dtype;
- combine/dispatch_bwd match fp32 einsum forms (no atomics; bitwise
  repeatable);
- the aux load-balance kernel matches autograd AND finite differences of
  the reference loss (f detached);
- EP accounting: partial-ownership specs size layouts locally, and the
  experts stage on a local shard matches the reference restricted to the
  held experts; program-level lowering of partial specs is rejected
  (multi-rank runtime pending).
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("MoE ladder-1 needs CUDA", allow_module_level=True)

pytestmark = pytest.mark.gpu

from dataflow.tasks import ops
from dataflow.tasks.kernels import KernelCtx, resolve_kernels
from dataflow.tasks.moe import (
    MOE_SHARED_STAGES,
    MOE_STAGES,
    MoESpec,
    moe_aux_loss_reference,
    moe_context_specs,
    moe_local_rows,
    moe_mlp_reference,
    moe_mlp_tail_bwd,
    moe_topk_reference,
    moe_weight_specs,
)
from dataflow.training.testing.gradcheck import rel_l2

MODES = ("topk_then_softmax", "softmax_then_topk")
_MOE_OPS = (
    "moe_topk_softmax", "moe_router_bwd", "moe_aux_lb_grad",
    "moe_sort", "moe_dispatch_fwd", "moe_dispatch_bwd", "moe_combine_fwd",
    "moe_grouped_mm_fwd", "moe_grouped_mm_dgrad", "moe_grouped_mm_wgrad",
    "swiglu_packed_fwd", "swiglu_packed_bwd",
)


def _kctx():
    return KernelCtx(stream_handle=0, torch_stream=None)


def _kset(eager: bool = False):
    if not eager:
        return resolve_kernels()
    return resolve_kernels(overrides={
        op: ("aten" if op in ("moe_sort", "moe_dispatch_fwd") else "eager")
        for op in _MOE_OPS
    })


def _gen(seed: int = 0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _randb(*shape, gen, scale=1.0):
    return (torch.rand(*shape, generator=gen, device="cuda") - 0.5).mul(scale).bfloat16()


# --- routing --------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_topk_softmax_vs_reference_incl_ties(mode):
    g = _gen(1)
    t, e, k = 257, 24, 4
    logits = _randb(t, e, gen=g, scale=4.0)
    # crafted exact ties: bf16 quantization makes collisions likely anyway;
    # force them so the smallest-index pin is load-bearing
    logits[0, :] = 0.5                      # full-row tie -> ids 0..k-1
    logits[1, 3] = logits[1, 17] = logits[1].max() + 1.0
    logits[2, :6] = logits[2].max() + 2.0   # 6-way tie at the top

    ref_w, ref_ids = moe_topk_reference(logits, k, mode)
    for eager in (False, True):
        K = _kset(eager)
        w_out = torch.empty(t, k, dtype=torch.bfloat16, device="cuda")
        ids_out = torch.empty(t, k, dtype=torch.int32, device="cuda")
        K.moe_topk_softmax(_kctx(), logits, w_out, ids_out, top_k=k, mode=mode)
        torch.cuda.synchronize()
        assert torch.equal(ids_out.long(), ref_ids), f"ids diverge (eager={eager})"
        assert rel_l2(w_out.float(), ref_w) < 2e-2
    assert ref_ids[0, :].tolist() == list(range(k))
    assert ref_ids[1, 0].item() == 3        # smallest index among the tie
    assert ref_ids[2, 0].item() == 0


def test_topk_eager_matches_reference_bitwise():
    g = _gen(2)
    logits = _randb(129, 16, gen=g, scale=3.0)
    K = _kset(eager=True)
    for mode in MODES:
        ref_w, ref_ids = moe_topk_reference(logits, 3, mode)
        w_out = torch.empty(129, 3, dtype=torch.bfloat16, device="cuda")
        ids_out = torch.empty(129, 3, dtype=torch.int32, device="cuda")
        K.moe_topk_softmax(_kctx(), logits, w_out, ids_out, top_k=3, mode=mode)
        torch.cuda.synchronize()
        assert torch.equal(w_out, ref_w.to(torch.bfloat16))
        assert torch.equal(ids_out.long(), ref_ids)


@pytest.mark.parametrize("mode", MODES)
def test_router_bwd_vs_autograd(mode):
    g = _gen(3)
    t, e, k = 128, 12, 3
    logits = _randb(t, e, gen=g, scale=3.0)
    dprob = (torch.rand(t, k, generator=g, device="cuda") - 0.5).float()

    leaf = logits.float().detach().requires_grad_()
    ref_w, ref_ids = moe_topk_reference(leaf, k, mode)
    (ref_w * dprob).sum().backward()

    route_w = ref_w.detach().to(torch.bfloat16)
    ids = ref_ids.to(torch.int32)
    for eager in (False, True):
        K = _kset(eager)
        dlogits = torch.full((t, e), 7.7, dtype=torch.float32, device="cuda")
        K.moe_router_bwd(_kctx(), dprob, route_w, ids, logits, dlogits, mode=mode)
        torch.cuda.synchronize()
        tol = 2e-2 if mode == "topk_then_softmax" else 1e-3
        assert rel_l2(dlogits, leaf.grad) < tol, f"eager={eager}"


def test_aux_grad_vs_autograd_and_finite_difference():
    g = _gen(4)
    t, e, k, alpha = 96, 8, 2, 0.05
    logits = _randb(t, e, gen=g, scale=2.0)
    _, ids = moe_topk_reference(logits, k, "softmax_then_topk")
    counts = torch.bincount(ids.reshape(-1), minlength=e).to(torch.int32)

    leaf = logits.float().detach().requires_grad_()
    moe_aux_loss_reference(leaf, ids, n_experts=e, aux_coef=alpha).backward()

    for eager in (False, True):
        K = _kset(eager)
        dlogits = torch.zeros(t, e, dtype=torch.float32, device="cuda")
        K.moe_aux_lb_grad(_kctx(), logits, counts, dlogits, alpha=alpha, top_k=k)
        torch.cuda.synchronize()
        assert rel_l2(dlogits, leaf.grad) < 1e-3, f"eager={eager}"

    # finite differences of the reference loss (fp64, f held fixed)
    l64 = logits.double()
    h = 1e-4
    for (ti, ei) in ((0, 0), (5, 3)):
        lp, lm = l64.clone(), l64.clone()
        lp[ti, ei] += h
        lm[ti, ei] -= h
        fd = (
            moe_aux_loss_reference(lp.float(), ids, n_experts=e, aux_coef=alpha)
            - moe_aux_loss_reference(lm.float(), ids, n_experts=e, aux_coef=alpha)
        ).item() / (2 * h)
        assert abs(fd - leaf.grad[ti, ei].item()) < 5e-3 * max(1.0, abs(fd))


# --- sort / dispatch / combine ----------------------------------------------------


def test_moe_sort_permutation_offsets_stability():
    g = _gen(5)
    t, k, e = 300, 4, 16
    ids = torch.randint(0, e, (t, k), generator=g, device="cuda", dtype=torch.int32)
    K = _kset()
    order = torch.empty(t * k, dtype=torch.int32, device="cuda")
    offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    K.moe_sort(_kctx(), ids, order, offsets, n_experts=e)
    torch.cuda.synchronize()

    flat = ids.reshape(-1).long()
    assert torch.equal(order.long().sort().values, torch.arange(t * k, device="cuda"))
    sorted_e = flat[order.long()]
    assert bool((sorted_e[1:] >= sorted_e[:-1]).all()), "expert-monotone"
    for exp in range(e):  # stability: original flat order within each expert
        seg = order.long()[sorted_e == exp]
        assert bool((seg[1:] > seg[:-1]).all())
    counts = torch.bincount(flat, minlength=e)
    assert offsets[0].item() == 0
    assert torch.equal(offsets[1:].long(), counts.cumsum(0))


def test_dispatch_and_combine_vs_einsum():
    g = _gen(6)
    t, k, e, d = 128, 3, 8, 64
    ids = torch.randint(0, e, (t, k), generator=g, device="cuda", dtype=torch.int32)
    x = _randb(t, d, gen=g)
    K = _kset()
    order = torch.empty(t * k, dtype=torch.int32, device="cuda")
    offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    K.moe_sort(_kctx(), ids, order, offsets, n_experts=e)
    xp = torch.empty(t * k, d, dtype=torch.bfloat16, device="cuda")
    K.moe_dispatch_fwd(_kctx(), x, order, xp, top_k=k)
    torch.cuda.synchronize()
    assert torch.equal(xp, x[torch.div(order.long(), k, rounding_mode="floor")])

    slot_of = torch.empty(t * k, dtype=torch.int32, device="cuda")
    slot_of.scatter_(0, order.long(), torch.arange(t * k, dtype=torch.int32, device="cuda"))
    slot_of = slot_of.view(t, k)

    yp = _randb(t * k, d, gen=g)
    route_w = torch.rand(t, k, generator=g, device="cuda").bfloat16()
    resid = _randb(t, d, gen=g)
    ref_gather = yp[slot_of.reshape(-1).long()].view(t, k, d).float()
    ref_combine = (
        (ref_gather * route_w.float().unsqueeze(-1)).sum(1) + resid.float()
    ).to(torch.bfloat16)
    ref_dbwd = ref_gather.sum(1)

    for eager in (False, True):
        K2 = _kset(eager)
        out = torch.empty(t, d, dtype=torch.bfloat16, device="cuda")
        K2.moe_combine_fwd(_kctx(), yp, slot_of, route_w, resid, out)
        acc = torch.empty(t, d, dtype=torch.float32, device="cuda")
        K2.moe_dispatch_bwd(_kctx(), yp, slot_of, acc)
        torch.cuda.synchronize()
        assert rel_l2(out.float(), ref_combine.float()) < 1e-2, f"eager={eager}"
        assert rel_l2(acc, ref_dbwd) < 1e-3, f"eager={eager}"
        out2 = torch.empty_like(out)
        K2.moe_combine_fwd(_kctx(), yp, slot_of, route_w, resid, out2)
        torch.cuda.synchronize()
        assert torch.equal(out, out2), "combine must be bitwise repeatable"


# --- grouped GEMM -----------------------------------------------------------------


def _uneven_offsets(counts):
    c = torch.tensor(counts, dtype=torch.int32, device="cuda")
    off = torch.zeros(len(counts) + 1, dtype=torch.int32, device="cuda")
    off[1:] = c.cumsum(0)
    return off


@pytest.mark.parametrize("eager", (False, True))
def test_grouped_mm_vs_dense_loop(eager):
    g = _gen(7)
    counts = [0, 37, 128, 0, 191, 1, 300, 63]  # zero rows, non-tile counts
    e = len(counts)
    m = sum(counts)
    kd, n = 96, 80
    x = _randb(m, kd, gen=g)
    w = _randb(e, kd, n, gen=g)
    dy = _randb(m, n, gen=g)
    offsets = _uneven_offsets(counts)
    K = _kset(eager)

    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    K.moe_grouped_mm_fwd(_kctx(), x, w, offsets, out)
    dx = torch.empty(m, kd, dtype=torch.bfloat16, device="cuda")
    K.moe_grouped_mm_dgrad(_kctx(), dy, w, offsets, dx)
    dw = torch.full((e, kd, n), 3.3, dtype=torch.bfloat16, device="cuda")
    K.moe_grouped_mm_wgrad(_kctx(), x, dy, offsets, dw, accumulate=False)
    torch.cuda.synchronize()

    lo = 0
    for exp, c in enumerate(counts):
        hi = lo + c
        if c:
            assert rel_l2(out[lo:hi].float(), (x[lo:hi] @ w[exp]).float()) < 1e-2
            assert rel_l2(dx[lo:hi].float(), (dy[lo:hi] @ w[exp].t()).float()) < 1e-2
            assert rel_l2(dw[exp].float(), (x[lo:hi].t() @ dy[lo:hi]).float()) < 1e-2
        else:
            assert bool((dw[exp] == 0).all()), "create-mode empty expert must ZERO"
        lo = hi

    pre = _randb(e, kd, n, gen=g)
    dw2 = pre.clone()
    K.moe_grouped_mm_wgrad(_kctx(), x, dy, offsets, dw2, accumulate=True)
    torch.cuda.synchronize()
    lo = 0
    for exp, c in enumerate(counts):
        hi = lo + c
        expect = pre[exp].float() + (
            (x[lo:hi].t() @ dy[lo:hi]).to(torch.bfloat16).float() if c else 0.0
        )
        assert rel_l2(dw2[exp].float(), expect) < 2e-2
        lo = hi


def test_grouped_mm_bitwise_repeatable():
    g = _gen(8)
    counts = [64, 0, 200, 56]
    m = sum(counts)
    x, w = _randb(m, 64, gen=g), _randb(4, 64, 48, gen=g)
    offsets = _uneven_offsets(counts)
    K = _kset()
    a = torch.empty(m, 48, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    K.moe_grouped_mm_fwd(_kctx(), x, w, offsets, a)
    K.moe_grouped_mm_fwd(_kctx(), x, w, offsets, b)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


# --- swiglu packed ------------------------------------------------------------------


def test_swiglu_packed_matches_unpacked_bitwise():
    g = _gen(9)
    rows, f = 513, 96
    h13 = _randb(rows, 2 * f, gen=g, scale=4.0)
    ds = _randb(rows, f, gen=g)
    x1 = h13[:, :f].contiguous()
    x3 = h13[:, f:].contiguous()
    for impl in ("triton", "eager"):  # packed and unpacked pinned to the SAME impl
        K = resolve_kernels(overrides={
            "swiglu_fwd_out": impl, "swiglu_bwd": impl,
            "swiglu_packed_fwd": impl, "swiglu_packed_bwd": impl,
        })
        ref_out = torch.empty(rows, f, dtype=torch.bfloat16, device="cuda")
        K.swiglu_fwd_out(_kctx(), x1, x3, ref_out)
        out = torch.empty_like(ref_out)
        K.swiglu_packed_fwd(_kctx(), h13, out)
        rd1, rd3 = torch.empty_like(x1), torch.empty_like(x3)
        K.swiglu_bwd(_kctx(), ds, x1, x3, rd1, rd3)
        dh13 = torch.empty_like(h13)
        K.swiglu_packed_bwd(_kctx(), ds, h13, dh13)
        torch.cuda.synchronize()
        assert torch.equal(out, ref_out), f"packed fwd != unpacked ({impl})"
        assert torch.equal(dh13[:, :f], rd1) and torch.equal(dh13[:, f:], rd3)


# --- standalone module harness (full tail fwd + bwd vs reference autograd) ---------


def _harness_weights(d_model, moe, gen):
    w = {}
    for name, shape in moe_weight_specs(
        type("D", (), {"d_model": d_model})(), moe
    ):
        w[name] = _randb(*shape, gen=gen, scale=0.15)
    w["ffn_norm_w"] = torch.ones(d_model, dtype=torch.bfloat16, device="cuda")
    return w


class _Dims:
    def __init__(self, d_model, tokens, moe):
        self.d_model, self.tokens, self.moe = d_model, tokens, moe


def _run_tail_fwd(K, dims, w, resid, a):
    """Mirror a family block's ffn_norm + spliced MoE stages."""
    kctx = _kctx()
    h2 = torch.empty_like(resid)
    rstd = torch.empty(dims.tokens, dtype=torch.float32, device="cuda")
    K.rmsnorm_fwd(kctx, resid, w["ffn_norm_w"], h2, rstd)
    if a is not None:
        a["rstd_ffn"].copy_(rstd)
    y = torch.empty_like(resid)
    st = {"w": w, "a": a, "y": y, "h2": h2, "h_mid": resid}
    stages = MOE_SHARED_STAGES if dims.moe.n_shared_experts else MOE_STAGES
    for _name, fn, _emits in stages:
        fn(kctx, K, dims, st)
    return y


def _ctx_dict(dims, moe):
    a = {"rstd_ffn": torch.empty(dims.tokens, dtype=torch.float32, device="cuda")}
    tmap = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}
    for name, shape, dt in moe_context_specs(dims, moe):
        a[name] = torch.empty(shape, dtype=tmap[dt], device="cuda")
    return a


@pytest.mark.parametrize("shared,aux,mode", [
    (False, 0.0, "softmax_then_topk"),
    (False, 0.01, "softmax_then_topk"),
    (True, 0.001, "topk_then_softmax"),
])
def test_moe_tail_fwd_bwd_vs_reference(shared, aux, mode):
    g = _gen(11)
    d_model, t, e, k, f = 64, 192, 8, 2, 48
    moe = MoESpec(
        n_experts=e, top_k=k, d_ff_expert=f, routing_mode=mode, aux_coef=aux,
        n_shared_experts=int(shared), d_ff_shared=32 if shared else 0,
    )
    dims = _Dims(d_model, t, moe)
    K = _kset()
    w = _harness_weights(d_model, moe, g)
    resid = _randb(t, d_model, gen=g)
    dy = _randb(t, d_model, gen=g)

    a = _ctx_dict(dims, moe)
    y = _run_tail_fwd(K, dims, w, resid, a)

    # reference: same math via autograd leaves
    leaves = {n: v.detach().clone().requires_grad_() for n, v in w.items()}
    rl = resid.detach().clone().requires_grad_()
    h2_ref = ops.rmsnorm_reference(rl, leaves["ffn_norm_w"])
    y_ref, aux_ref = moe_mlp_reference(h2_ref, leaves, moe, rl)
    assert rel_l2(y.float(), y_ref.float()) < 2e-2

    ((y_ref.float() * dy.float()).sum() + aux_ref).backward()

    dw = {
        n: torch.full_like(v, 9.9) for n, v in w.items()
        if n not in ("ffn_norm_w",)
    }
    dw["ffn_norm_w"] = torch.full_like(w["ffn_norm_w"], 9.9)
    acc_written = set()

    def acc(name, value):
        acc_written.add(name)
        dw[name].copy_(value.to(dw[name].dtype))

    def norm_bwd(dyv, xv, rstd, wv):
        dxv = torch.empty_like(xv)
        dwv = torch.empty(wv.numel(), dtype=torch.float32, device="cuda")
        K.rmsnorm_bwd(_kctx(), dyv, xv, rstd, wv, dxv, dwv)
        return dxv, dwv

    a["xo"] = resid  # the tail reads the residual from the ctx dict
    dh_mid = moe_mlp_tail_bwd(
        _kctx(), K, dims, dy, a, w, dw, False, acc, norm_bwd, resid_field="xo",
    )
    torch.cuda.synchronize()

    assert rel_l2(dh_mid.float(), rl.grad.float()) < 3e-2
    for name in dw:
        ref_g = leaves[name].grad
        assert ref_g is not None, name
        assert rel_l2(dw[name].float(), ref_g.float()) < 4e-2, name

    # determinism: bwd twice -> bitwise identical everything
    dw2 = {n: torch.full_like(v, 5.5) for n, v in dw.items()}

    def acc2(name, value):
        dw2[name].copy_(value.to(dw2[name].dtype))

    dh_mid2 = moe_mlp_tail_bwd(
        _kctx(), K, dims, dy, a, w, dw2, False, acc2, norm_bwd, resid_field="xo",
    )
    torch.cuda.synchronize()
    assert torch.equal(dh_mid, dh_mid2)
    for name in dw:
        assert torch.equal(dw[name], dw2[name]), name


def test_moe_tail_recompute_reproduces_ctx_bitwise():
    """Recompute-mode re-derives the WHOLE ctx from (x, W): int fields must
    be torch.equal, float fields bitwise too (same kernels, same stream)."""
    g = _gen(12)
    moe = MoESpec(n_experts=8, top_k=2, d_ff_expert=48, aux_coef=0.01)
    dims = _Dims(64, 128, moe)
    K = _kset()
    w = _harness_weights(64, moe, g)
    resid = _randb(128, 64, gen=g)
    a1 = _ctx_dict(dims, moe)
    y1 = _run_tail_fwd(K, dims, w, resid, a1)
    a2 = _ctx_dict(dims, moe)
    y2 = _run_tail_fwd(K, dims, w, resid, a2)
    torch.cuda.synchronize()
    assert torch.equal(y1, y2)
    for name in a1:
        assert torch.equal(a1[name], a2[name]), name


def test_moe_tail_eager_kernels_match_fused():
    g = _gen(13)
    moe = MoESpec(
        n_experts=8, top_k=2, d_ff_expert=48, aux_coef=0.01,
        n_shared_experts=1, d_ff_shared=32,
    )
    dims = _Dims(64, 160, moe)
    w = _harness_weights(64, moe, g)
    resid = _randb(160, 64, gen=g)
    outs = []
    for eager in (False, True):
        K = _kset(eager)
        a = _ctx_dict(dims, moe)
        outs.append(_run_tail_fwd(K, dims, w, resid, a))
    torch.cuda.synchronize()
    assert rel_l2(outs[0].float(), outs[1].float()) < 1e-2


# --- EP accounting -------------------------------------------------------------------


def test_partial_ownership_sizes_and_sharded_experts_math():
    g = _gen(14)
    e, k, f, d_model, t = 8, 2, 32, 64, 96
    full = MoESpec(n_experts=e, top_k=k, d_ff_expert=f)
    part = dataclasses.replace(full, expert_ids=(1, 4, 6))

    D = type("D", (), {"d_model": d_model, "tokens": t})()
    wspecs = dict(moe_weight_specs(D, part))
    assert wspecs["w13_experts"] == (3, d_model, 2 * f)
    assert wspecs["w2_experts"] == (3, f, d_model)
    assert wspecs["w_router"] == (d_model, e), "router stays GLOBAL width"
    cspecs = {n: s for n, s, _ in moe_context_specs(D, part)}
    assert cspecs["route_offsets"] == (part.n_local_experts + 1,)
    assert cspecs["router_logits"] == (t, e)
    assert moe_local_rows(full, t) == t * k
    assert moe_local_rows(part, t) == -(-t * k * 3 // e)

    # sharded experts stage == reference restricted to held experts:
    # run grouped fwd on a LOCAL segment buffer built for experts (1,4,6)
    K = _kset()
    w13_full = _randb(e, d_model, 2 * f, gen=g, scale=0.2)
    w13_local = w13_full[list(part.expert_ids)].contiguous()
    counts = [5, 40, 19]
    m = sum(counts)
    x = _randb(m, d_model, gen=g)
    out = torch.empty(m, 2 * f, dtype=torch.bfloat16, device="cuda")
    K.moe_grouped_mm_fwd(_kctx(), x, w13_local, _uneven_offsets(counts), out)
    torch.cuda.synchronize()
    lo = 0
    for slot, exp in enumerate(part.expert_ids):
        hi = lo + counts[slot]
        assert rel_l2(out[lo:hi].float(), (x[lo:hi] @ w13_full[exp]).float()) < 1e-2
        lo = hi


def test_spec_validation():
    with pytest.raises(ValueError):
        MoESpec(n_experts=8, top_k=9, d_ff_expert=32)
    with pytest.raises(ValueError):
        MoESpec(n_experts=8, top_k=2, d_ff_expert=32, routing_mode="sigmoid")
    with pytest.raises(ValueError):
        MoESpec(n_experts=8, top_k=2, d_ff_expert=32, dispatch_dtype="fp8_e4m3")
    with pytest.raises(ValueError):
        MoESpec(n_experts=8, top_k=2, d_ff_expert=32, expert_ids=(1, 1))
    with pytest.raises(ValueError):
        MoESpec(n_experts=8, top_k=2, d_ff_expert=32, n_shared_experts=2)
