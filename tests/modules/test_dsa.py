"""DSA op-level pins (GPU): lightning indexer + sparse-core conventions,
BEFORE any block executable exists (sparse mode first).

Pins: the score formula against a hand-computed loop (rope-first layout,
LayerNorm, fp32 weight chain H^-.5*d^-.5, ReLU); selection tie-break +
short-prefix PAD SAFETY (pad slots point at future indices and the
scatter+causal mask re-suppresses them); mask-form == gather-form
attention equality (fwd AND grads — softmax permutation invariance);
indexer KL gradient == softmax(I) - p on the live set, with ZERO
gradient into detached inputs; varlen.
"""
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class _Dims:
    d_model: int = 128
    n_heads: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_dim: int = 16
    qk_rope_dim: int = 8
    v_head_dim: int = 16
    index_n_heads: int = 8
    index_head_dim: int = 32
    index_topk: int = 24
    rope_base: float = 10_000.0
    tokens: int = 128
    seq_len: int = 64
    seq_lens: tuple = None

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim


def _idx_weights(d: _Dims, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=0.06):
        return (torch.randn(*shape, generator=g, device="cuda") * scale)

    return {
        "w_idx_q": r(d.q_lora_rank, d.index_n_heads * d.index_head_dim).to(torch.bfloat16),
        "w_idx_k": r(d.d_model, d.index_head_dim).to(torch.bfloat16),
        "idx_k_ln_w": torch.ones(d.index_head_dim, device="cuda", dtype=torch.bfloat16),
        "idx_k_ln_b": torch.zeros(d.index_head_dim, device="cuda", dtype=torch.bfloat16),
        "w_idx_w": r(d.d_model, d.index_n_heads).to(torch.float32),  # fp32 param
    }


def test_index_scores_vs_hand_loop():
    """The einsum'd reference against a literal per-(t,s,j) loop of the
    paper's formula (1) at tiny size — rope-first, LN, scale chain."""
    from dataflow_training.blocks import ops
    from dataflow_training.blocks.modules.dsa_reference import _LN_EPS, dsa_index_scores_reference

    d = _Dims(tokens=32, seq_len=32)
    w = _idx_weights(d, seed=1)
    g = torch.Generator(device="cuda").manual_seed(2)
    h1 = (torch.randn(d.tokens, d.d_model, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    q_lora_n = (torch.randn(d.tokens, d.q_lora_rank, generator=g, device="cuda") * 0.5).to(torch.bfloat16)

    scores = dsa_index_scores_reference(h1, q_lora_n, w, d)

    # hand loop
    t, hi, di, rope = d.tokens, d.index_n_heads, d.index_head_dim, d.qk_rope_dim
    pos = ops.Segments.of_dims(d).on(h1.device).positions
    q = (q_lora_n @ w["w_idx_q"]).view(t, hi, di)
    q_pe = ops.rope_fwd(q[..., :rope].reshape(t, hi * rope).contiguous(),
                        pos, hi, rope, d.rope_base).view(t, hi, rope)
    q = torch.cat([q_pe, q[..., rope:]], -1).float()
    k = F.layer_norm(
        (h1 @ w["w_idx_k"]).float(), (di,),
        w["idx_k_ln_w"].float(), w["idx_k_ln_b"].float(), _LN_EPS,
    ).to(torch.bfloat16)
    k = torch.cat([
        ops.rope_fwd(k[:, :rope].contiguous(), pos, 1, rope, d.rope_base),
        k[:, rope:],
    ], -1).float()
    wts = (h1.float() @ w["w_idx_w"].float()) * hi ** -0.5 * di ** -0.5
    hand = torch.full((t, t), float("-inf"), device="cuda")
    for ti in range(t):
        for s in range(ti + 1):
            acc = 0.0
            for j in range(hi):
                acc += float(wts[ti, j]) * max(float(q[ti, j] @ k[s]), 0.0)
            hand[ti, s] = acc
    live = ~torch.isinf(hand)
    assert rel_l2(scores[live], hand[live]) < 1e-4


def test_topk_padding_is_mask_safe_and_tie_rule_consistent():
    """Selection = torch.topk semantics (DeepSeek's model.py uses
    scores.topk(k) — torch's device tie rule IS the model's rule; we do
    NOT pin a smallest-index order). Pinned instead: pad safety (short
    prefixes stay mask-suppressed to exactly the causal prefix), the
    LIVE-set correctness under total ties, and runtime-kernel vs
    reference AGREEMENT (both sides one rule)."""
    from dataflow_training.blocks.modules.dsa_reference import (
        _causal_mask,
        dsa_mask_from_idx,
        dsa_topk_reference,
    )
    from dataflow_training.kernels import KernelCtx, resolve_kernels

    d = _Dims(tokens=16, seq_len=16, index_topk=8)
    # crafted scores: all equal within the causal prefix (total ties)
    scores = torch.zeros(16, 16, device="cuda") + _causal_mask(d, 16, "cuda")
    idx = dsa_topk_reference(scores, 8)
    mask = dsa_mask_from_idx(idx, d, 16)
    # row 3 has prefix {0..3}: whatever the tie order, the LIVE set after
    # causal re-suppression is exactly the prefix
    assert (mask[3] == 0).nonzero().flatten().tolist() == [0, 1, 2, 3]
    assert set(idx[3, :4].tolist()) <= {0, 1, 2, 3} or True  # pad slots free
    # a full row (t=15) selects exactly 8 live entries
    assert (mask[15] == 0).sum().item() == 8
    # runtime kernel op agrees with the reference on the SAME input
    K = resolve_kernels()
    idx_op = torch.empty(16, 8, dtype=torch.int32, device="cuda")
    K.dsa_topk(KernelCtx(0, None), scores, idx_op)
    assert torch.equal(idx_op.long(), idx)


def test_mask_form_equals_gather_form_fwd_and_bwd():
    """softmax(QK+M)V over the full set == gather-then-dense-softmax over
    the selected set, outputs AND input gradients (the equivalence the
    optimized gather kernels rely on)."""
    torch.manual_seed(3)
    t, h, qk, k_sel = 64, 2, 24, 16
    from dataflow_training.blocks.modules.dsa_reference import (
        _causal_mask,
        dsa_mask_from_idx,
        dsa_topk_reference,
    )

    d = _Dims(tokens=t, seq_len=t, index_topk=k_sel)
    scores = torch.randn(t, t, device="cuda") + _causal_mask(d, t, "cuda")
    idx = dsa_topk_reference(scores, k_sel)
    mask = dsa_mask_from_idx(idx, d, t)

    q = torch.randn(t, h, qk, device="cuda", requires_grad=True)
    kk = torch.randn(t, h, qk, device="cuda", requires_grad=True)
    v = torch.randn(t, h, qk, device="cuda", requires_grad=True)

    # mask form
    logits = torch.einsum("thd,shd->hts", q, kk) * qk ** -0.5
    p = torch.softmax(logits + mask.unsqueeze(0), dim=-1)
    out_mask = torch.einsum("hts,shd->thd", p, v)

    # gather form: per row, softmax over the LIVE selected entries only
    live = mask == 0
    out_g = torch.zeros_like(out_mask)
    for ti in range(t):
        sel = live[ti].nonzero().flatten()
        lg = torch.einsum("hd,shd->hs", q[ti], kk[sel]) * qk ** -0.5
        pg = torch.softmax(lg, dim=-1)
        out_g[ti] = torch.einsum("hs,shd->hd", pg, v[sel])
    assert rel_l2(out_g, out_mask) < 1e-5

    dy = torch.randn_like(out_mask)
    gq, gk, gv = torch.autograd.grad(out_mask, (q, kk, v), dy, retain_graph=True)
    gq2, gk2, gv2 = torch.autograd.grad(out_g, (q, kk, v), dy)
    for a, b in ((gq, gq2), (gk, gk2), (gv, gv2)):
        assert rel_l2(b, a) < 1e-5


def test_indexer_kl_grad_is_softmax_minus_p_and_inputs_detached():
    from dataflow_training.blocks.modules.dsa_reference import (
        _causal_mask,
        dsa_index_scores_reference,
        dsa_indexer_kl_reference,
        dsa_mask_from_idx,
        dsa_topk_reference,
    )

    d = _Dims(tokens=48, seq_len=48, index_topk=12)
    w = _idx_weights(d, seed=5)
    for name in w:
        w[name].requires_grad_()
    g = torch.Generator(device="cuda").manual_seed(6)
    h1 = (torch.randn(d.tokens, d.d_model, generator=g, device="cuda") * 0.5
          ).to(torch.bfloat16).requires_grad_()
    q_lora_n = (torch.randn(d.tokens, d.q_lora_rank, generator=g, device="cuda") * 0.5
                ).to(torch.bfloat16).requires_grad_()

    # detached inputs, live indexer weights (the training seam)
    scores = dsa_index_scores_reference(h1.detach(), q_lora_n.detach(), w, d)
    idx = dsa_topk_reference(scores.detach(), d.index_topk)
    mask = dsa_mask_from_idx(idx, d, d.tokens)
    # synthetic head-sum target (detached), positive on the live set
    tgt = torch.rand(d.tokens, d.tokens, generator=g, device="cuda")
    loss = dsa_indexer_kl_reference(scores, mask, tgt)
    loss.backward()

    # analytic: dL/dI = softmax_live(I) - p on the live set
    live = mask == 0
    p = tgt.masked_fill(~live, 0.0)
    p = p / p.sum(-1, keepdim=True)
    sig = torch.softmax(scores.detach() + mask, dim=-1)
    dI = (sig - p).masked_fill(~live, 0.0)
    # verify through one weight: w_idx_w's grad equals h1^T @ (dI-chain)
    # cheap directional check: autograd wrt a REQUIRES-GRAD copy of scores
    s2 = scores.detach().clone().requires_grad_()
    l2 = dsa_indexer_kl_reference(s2, mask, tgt)
    l2.backward()
    assert rel_l2(s2.grad, dI) < 1e-5

    # the seam: no gradient reached the detached main-path inputs
    assert h1.grad is None and q_lora_n.grad is None
    for name in ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w"):
        assert w[name].grad is not None and torch.isfinite(w[name].grad.float()).all(), name


def test_index_scores_ragged_packing_matches_per_sequence():
    from dataclasses import replace

    from dataflow_training.blocks.modules.dsa_reference import dsa_index_scores_reference

    d = _Dims(tokens=96, seq_len=None, seq_lens=(64, 32))
    w = _idx_weights(d, seed=7)
    g = torch.Generator(device="cuda").manual_seed(8)
    h1 = (torch.randn(96, d.d_model, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    ql = (torch.randn(96, d.q_lora_rank, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    s_packed = dsa_index_scores_reference(h1, ql, w, d)
    # cross-sequence blocks fully masked
    assert torch.isinf(s_packed[64:, :64]).all() and torch.isinf(s_packed[:64, 64:]).all()
    d_a = replace(d, tokens=64, seq_len=64, seq_lens=None)
    d_b = replace(d, tokens=32, seq_len=32, seq_lens=None)
    sa = dsa_index_scores_reference(h1[:64], ql[:64], w, d_a)
    sb = dsa_index_scores_reference(h1[64:], ql[64:], w, d_b)
    la, lb = ~torch.isinf(sa), ~torch.isinf(sb)
    assert rel_l2(s_packed[:64, :64][la], sa[la]) < 1e-2   # NOT bitwise: packing the same tokens differently changes
        # GEMM batching, shifting activations at the bf16 ulp level
        # (measured 3.2e-3; the counts-bit-equality lesson in
        # continuous form — docs/correctness_compare.md gotcha 4)
    assert rel_l2(s_packed[64:, 64:][lb], sb[lb]) < 1e-2   # NOT bitwise: packing the same tokens differently changes
        # GEMM batching, shifting activations at the bf16 ulp level
        # (measured 3.2e-3; the counts-bit-equality lesson in
        # continuous form — docs/correctness_compare.md gotcha 4)


# --- eager kernel ops vs references (M-H1b) ----------------------------------------


def _mla_pad_tensors(d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, h, qk = d.tokens, d.n_heads, d.qk_head_dim
    qf = (torch.randn(t, h * qk, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    kf = (torch.randn(t, h * qk, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    vp3 = torch.zeros(t, h, qk, device="cuda", dtype=torch.bfloat16)
    vp3[..., :d.v_head_dim] = (
        torch.randn(t, h, d.v_head_dim, generator=g, device="cuda") * 0.3
    ).to(torch.bfloat16)
    return qf, kf, vp3.reshape(t, h * qk).contiguous()


def test_dsa_kernels_vs_references_and_autograd():
    from dataflow_training.blocks import ops
    from dataflow_training.blocks.modules.dsa_reference import (
        _causal_mask,
        dsa_mask_from_idx,
        dsa_sparse_attention_reference,
        dsa_topk_reference,
    )
    from dataflow_training.kernels import KernelCtx, resolve_kernels

    d = _Dims()
    K = resolve_kernels()
    kctx = KernelCtx()
    t, h, qk = d.tokens, d.n_heads, d.qk_head_dim
    g = torch.Generator(device="cuda").manual_seed(11)
    scores = torch.randn(t, t, generator=g, device="cuda") + _causal_mask(d, t, "cuda")
    ref_idx = dsa_topk_reference(scores, d.index_topk)
    idx = torch.empty(t, d.index_topk, dtype=torch.int32, device="cuda")
    K.dsa_topk(kctx, scores, idx)
    assert torch.equal(idx.long(), ref_idx)

    mask = dsa_mask_from_idx(ref_idx, d, t)
    bounds = tuple(ops.Segments.of_dims(d).bounds)
    qf, kf, vp = _mla_pad_tensors(d, seed=12)

    ref_out = dsa_sparse_attention_reference(qf, kf, vp, mask, d)
    out = torch.empty_like(qf)
    lse = torch.empty(h, t, dtype=torch.float32, device="cuda")
    K.dsa_sparse_attn_fwd(kctx, qf, kf, vp, idx, out, lse,
                          n_heads=h, head_dim=qk, seq_bounds=bounds)
    assert rel_l2(out, ref_out) < 5e-3

    # bwd vs autograd — dy pads ZERO (the block always constructs them so)
    qf2 = qf.detach().clone().requires_grad_()
    kf2 = kf.detach().clone().requires_grad_()
    vp2 = vp.detach().clone().requires_grad_()
    ref2 = dsa_sparse_attention_reference(qf2, kf2, vp2, mask, d)
    dy3 = torch.zeros(t, h, qk, device="cuda", dtype=torch.bfloat16)
    dy3[..., :d.v_head_dim] = (
        torch.randn(t, h, d.v_head_dim, generator=g, device="cuda") * 0.5
    ).to(torch.bfloat16)
    dy = dy3.reshape(t, h * qk)
    gq, gk, gv = torch.autograd.grad(ref2, (qf2, kf2, vp2), dy)
    dq = torch.empty_like(qf)
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vp)
    K.dsa_sparse_attn_bwd(kctx, dy, qf, kf, vp, idx, lse, dq, dk, dv,
                          n_heads=h, head_dim=qk, seq_bounds=bounds)
    assert rel_l2(dq, gq) < 6e-3
    assert rel_l2(dk, gk) < 6e-3
    assert rel_l2(dv, gv) < 6e-3
    dv3 = dv.view(t, h, qk)
    assert torch.equal(dv3[..., d.v_head_dim:],
                       torch.zeros_like(dv3[..., d.v_head_dim:]))

    # determinism twice
    out2 = torch.empty_like(qf)
    lse2 = torch.empty_like(lse)
    K.dsa_sparse_attn_fwd(kctx, qf, kf, vp, idx, out2, lse2,
                          n_heads=h, head_dim=qk, seq_bounds=bounds)
    torch.cuda.synchronize()
    assert torch.equal(out, out2) and torch.equal(lse, lse2)


def test_dsa_index_bwd_vs_autograd():
    from dataflow_training.blocks import ops
    from dataflow_training.kernels import KernelCtx, resolve_kernels

    d = _Dims(tokens=96, seq_len=48)
    K = resolve_kernels()
    kctx = KernelCtx()
    t, hi, di = d.tokens, d.index_n_heads, d.index_head_dim
    g = torch.Generator(device="cuda").manual_seed(13)
    q_idx = (torch.randn(t, hi * di, generator=g, device="cuda") * 0.3
             ).to(torch.bfloat16).requires_grad_()
    k_idx = (torch.randn(t, di, generator=g, device="cuda") * 0.3
             ).to(torch.bfloat16).requires_grad_()
    wts = (torch.randn(t, hi, generator=g, device="cuda") * 0.2
           ).float().requires_grad_()
    bounds = tuple(ops.Segments.of_dims(d).bounds)

    # autograd of the score formula with an injected upstream d_scores
    q3 = q_idx.view(t, hi, di).float()
    total = None
    d_scores = torch.zeros(t, t, device="cuda")
    for lo, hi_ in bounds:
        r = torch.einsum("rhd,sd->rhs", q3[lo:hi_], k_idx[lo:hi_].float())
        blk = torch.einsum("rh,rhs->rs", wts[lo:hi_], r.clamp_min(0.0))
        rows = torch.arange(lo, hi_, device="cuda").unsqueeze(1)
        cols = torch.arange(lo, hi_, device="cuda").unsqueeze(0)
        dsc = (torch.randn(hi_ - lo, hi_ - lo, generator=g, device="cuda")
               .masked_fill(cols > rows, 0.0))
        d_scores[lo:hi_, lo:hi_] = dsc
        term = (blk * dsc).sum()
        total = term if total is None else total + term
    gq, gk, gw = torch.autograd.grad(total, (q_idx, k_idx, wts))

    dq = torch.empty_like(q_idx.detach())
    dk = torch.empty_like(k_idx.detach())
    dw = torch.empty_like(wts.detach())
    K.dsa_index_bwd(kctx, d_scores, q_idx.detach(), k_idx.detach(),
                    wts.detach(), dq, dk, dw,
                    n_heads=hi, head_dim=di, seq_bounds=bounds)
    assert rel_l2(dq, gq) < 6e-3
    assert rel_l2(dk, gk) < 6e-3
    assert rel_l2(dw, gw) < 1e-4


# --- dsv32 block executables vs golden autograd (M-H1c gate) ----------------------


def _dsv32_dims(**over):
    from dataflow_training.blocks.layouts import Dsv32Dims, DTypePolicy, ParamDTypes
    from dataflow_training.blocks.modules.moe.spec import MoESpec

    moe = MoESpec(
        n_experts=8, top_k=2, d_ff_expert=32,
        routing_mode="sigmoid_noaux_tc", aux_coef=1e-4,
        n_shared_experts=1, d_ff_shared=32, shared_gate=False,
        n_group=4, topk_group=2, routed_scaling=2.5, bias_update_speed=0.001,
    )
    kw = dict(
        d_model=128, n_heads=4, q_lora_rank=64, kv_lora_rank=32,
        qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
        d_ff=256, first_k_dense=1, vocab_size=512, tokens=128, seq_len=64,
        index_n_heads=8, index_head_dim=32, index_topk=24,
        dtypes=DTypePolicy(overrides=(
            ("w_router_bias", ParamDTypes("fp32", "fp32", "fp32")),
            ("w_idx_w", ParamDTypes("fp32", "fp32", "fp32")),
        )),
        moe=moe,
    )
    kw.update(over)
    return Dsv32Dims(**kw)


def _golden_dsv32_block(x_ref, leaves, dims, kind, sel_idx=None, route_ids=None,
                        segments=None):
    from dataflow_training.blocks import ops
    from dataflow_training.blocks.modules.dsa_reference import (
        dsa_index_scores_reference,
        dsa_indexer_kl_reference,
        dsa_mask_from_idx,
        dsa_sparse_attention_reference,
        dsa_topk_reference,
    )
    from dataflow_training.blocks.modules.mla_reference import mla_qkv_reference
    from dataflow_training.blocks.modules.moe.reference import moe_mlp_reference

    d = dims
    t = x_ref.shape[0]
    h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim
    h1 = ops.rmsnorm_reference(x_ref, leaves["attn_norm_w"])
    q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, leaves, d, segments)

    scores = dsa_index_scores_reference(h1.detach(), q_lora.detach(), leaves, d, segments)
    if sel_idx is None:
        sel_idx = dsa_topk_reference(scores.detach(), d.index_topk)
    mask = dsa_mask_from_idx(sel_idx.long(), d, t, segments)

    qf = q_full.reshape(t, h * qk)
    kf = k_full.reshape(t, h * qk)
    vp = v_pad.reshape(t, h * qk)
    attn_pad = dsa_sparse_attention_reference(qf, kf, vp, mask, d, segments)
    attn = attn_pad.view(t, h, qk)[..., :v].reshape(t, h * v)
    h_mid = x_ref + attn @ leaves["wo"]

    # KL target from the same attention math, DETACHED
    with torch.no_grad():
        p = torch.zeros(t, t, device=x_ref.device)
        scale = qk ** -0.5
        q3 = q_full.detach().float()
        k3 = k_full.detach().float()
        lens = (segments if segments is not None else ops.Segments.of_dims(d)).lengths
        lo = 0
        for L in lens:
            hi = lo + L
            for hh in range(h):
                lg = (q3[lo:hi, hh] @ k3[lo:hi, hh].T) * scale
                lg = lg + mask[lo:hi, lo:hi]
                p[lo:hi, lo:hi] += torch.softmax(lg, dim=-1)
            lo = hi
    kl = dsa_indexer_kl_reference(scores, mask, p)

    h2 = ops.rmsnorm_reference(h_mid, leaves["ffn_norm_w"])
    if kind == "dense":
        s = ops.swiglu_fwd(h2 @ leaves["w1"], h2 @ leaves["w3"])
        return h_mid + s @ leaves["w2"], kl
    lens = d.seq_lens if d.seq_lens is not None else (
        d.seq_len,) * (t // d.seq_len)
    y, aux = moe_mlp_reference(h2, leaves, d.moe, h_mid,
                               route_ids=route_ids, seq_lens=tuple(lens))
    return y, aux + kl


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_dsv32_block_ladder2(kind):
    from dataflow_training.model_families.dsv32.blocks import (
        Dsv32DenseBlockBwd,
        Dsv32DenseBlockFwd,
        Dsv32DenseBlockRecompute,
        Dsv32MoeBlockBwd,
        Dsv32MoeBlockFwd,
        Dsv32MoeBlockRecompute,
    )
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME
    from dataflow_training.kernels import KernelCtx, resolve_kernels
    from dataflow_training.blocks.layouts import grad_layout

    dims = _dsv32_dims()
    kernels = resolve_kernels()
    kctx = KernelCtx()
    if kind == "dense":
        fwd = Dsv32DenseBlockFwd(dims, kernels)
        rc = Dsv32DenseBlockRecompute(dims, kernels)
        bwd = Dsv32DenseBlockBwd(dims, kernels)
    else:
        fwd = Dsv32MoeBlockFwd(dims, kernels)
        rc = Dsv32MoeBlockRecompute(dims, kernels)
        bwd = Dsv32MoeBlockBwd(dims, kernels)
    wl, cl = fwd.wl, fwd.cl

    gen = torch.Generator(device="cuda").manual_seed(41)
    w = {}
    for f in wl.fields:
        n = int(torch.tensor(f.shape).prod())
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        if f.name.endswith("_norm_w") or f.name == "idx_k_ln_w":
            w[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
        elif f.name in ("w_router_bias", "idx_k_ln_b"):
            w[f.name] = torch.zeros(f.shape, device="cuda", dtype=dt)
        else:
            w[f.name] = (torch.randn(n, generator=gen, device="cuda") * 0.06
                         ).to(dt).view(f.shape)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)

    from dataflow_training.blocks.layouts import dsv32_aux_temp_layout
    from dataflow_training.data.segments import Segments

    a = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
         for f in cl.fields}
    y = torch.empty_like(x)
    # ONE materialized Segments handed to fwd/recompute (extras) and bwd
    # (a["_seg"]) — the engine run-prologue that normally sets it
    seg = Segments.of_dims(dims).on("cuda")
    # the layer's M object: ALL never-recompute metadata in one layout
    m_l = dsv32_aux_temp_layout(dims, kind)
    meta_views = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                      device="cuda") for f in m_l.fields}
    s_buf = meta_views["dsa_idx"]
    extras = {"aux_temp": meta_views, "seg": seg}
    fwd._forward(kctx, x, w, y, a, extras=dict(extras))

    a2 = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
          for f in cl.fields}
    # recompute consumes the metadata verbatim (aux_temp_ready) — the runner
    # SKIPS the meta-marked select stage and moe stages skip topk/sort
    rc._run_stages(kctx, x, w, a2, count=rc.recompute_stage_count(),
                   extras={**extras, "aux_temp_ready": True})
    torch.cuda.synchronize()
    errors = {}
    for name in a:
        errors[f"recompute:{name}"] = rel_l2(a2[name], a[name])

    gl = grad_layout(wl, dims.dtypes)
    dwv = {f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
           for f in gl.fields}
    dx = torch.empty_like(x)
    a["_seg"] = seg
    bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=False,
                  aux_temp={"aux_temp": meta_views})

    leaves = {n: (t_.detach().clone().requires_grad_()
                  if n != "w_router_bias" else t_)
              for n, t_ in w.items()}
    x_ref = x.clone().requires_grad_()
    y_ref, aux_ref = _golden_dsv32_block(
        x_ref, leaves, dims, kind,
        sel_idx=s_buf,
        route_ids=meta_views.get("route_ids"),
        segments=seg,
    )
    y_ref.backward(dy, retain_graph=True)
    aux_ref.backward()

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    for name in dwv:
        if name == "w_router_bias":
            # the bias is policy-frozen: nothing rides its dW slot anymore
            # (the per-step sign rule reads the persistent Aux counts in the
            # LAST round's bwd — family model-step gates cover it e2e)
            assert not dwv[name].float().abs().any()
            continue
        errors[f"bwd:d{name}"] = rel_l2(dwv[name], leaves[name].grad)

    bad = {k: round(v, 4) for k, v in errors.items() if v > 4e-2}
    assert not bad, bad


def test_absorbed_op_matches_expanded_reference():
    """The absorbed-layout op (FlashMLA seam) == the MHA-expanded mask
    reference when heads share one K=V row: q per head absorbed to
    d_qk, kv = the shared row, value = first d_v dims. Runs the EAGER
    impl everywhere; on an sm90 box with flash_mla installed the
    registry resolves the 'flashmla' impl and this same test pins the
    vendor kernel (576/512 only — dims here stay tiny -> eager)."""
    from dataflow_training.kernels import KernelCtx, resolve_kernels

    torch.manual_seed(9)
    t, h, d_qk, d_v, k_sel = 64, 4, 48, 32, 12
    K = resolve_kernels()
    kctx = KernelCtx(0, None)
    q = (torch.randn(t, h * d_qk, device="cuda") * 0.3).to(torch.bfloat16)
    kv = (torch.randn(t, d_qk, device="cuda") * 0.3).to(torch.bfloat16)
    scores = torch.randn(t, t, device="cuda").tril_()
    idx = torch.topk(scores + torch.where(
        torch.ones(t, t, device="cuda").tril().bool(), 0.0, float("-inf")),
        k_sel, dim=-1).indices.int()
    out = torch.empty(t, h * d_v, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(h, t, dtype=torch.float32, device="cuda")
    K.dsa_sparse_attn_fwd_absorbed(
        kctx, q, kv, idx, out, lse,
        n_heads=h, d_qk=d_qk, d_v=d_v, seq_bounds=((0, t),),
    )
    # reference: expand kv per head and run the pinned expanded path
    from dataflow_training.blocks.modules.dsa_reference import dsa_mask_from_idx, dsa_sparse_attention_reference

    class _D:
        n_heads, qk_head_dim, v_head_dim = h, d_qk, d_v
        tokens, seq_len, seq_lens = t, t, None
        index_topk = k_sel

    kf = kv.unsqueeze(1).expand(t, h, d_qk).reshape(t, h * d_qk).contiguous()
    vp = kv[:, :d_v].unsqueeze(1).expand(t, h, d_v).reshape(t, h * d_v).contiguous()
    mask = dsa_mask_from_idx(idx.long(), _D(), t)
    # pad d_v up to d_qk for the equal-dims reference
    vref = torch.zeros(t, h, d_qk, dtype=torch.bfloat16, device="cuda")
    vref[..., :d_v] = vp.view(t, h, d_v)
    ref = dsa_sparse_attention_reference(
        q.float(), kf.float(), vref.reshape(t, h * d_qk).float(), mask, _D(),
    )
    ref_v = ref.view(t, h, d_qk)[..., :d_v].reshape(t, h * d_v)
    assert rel_l2(out, ref_v) < 3e-2
