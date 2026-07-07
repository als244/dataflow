"""DSA op-level pins (GPU): lightning indexer + sparse-core conventions,
BEFORE any block executable exists (M-H1, sparse mode first).

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

from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

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
    def seq_spec(self):
        return self.seq_lens if self.seq_lens is not None else self.seq_len

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
    from dataflow.tasks import ops
    from dataflow.tasks.dsa_reference import _LN_EPS, dsa_index_scores_reference

    d = _Dims(tokens=32, seq_len=32)
    w = _idx_weights(d, seed=1)
    g = torch.Generator(device="cuda").manual_seed(2)
    h1 = (torch.randn(d.tokens, d.d_model, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    q_lora_n = (torch.randn(d.tokens, d.q_lora_rank, generator=g, device="cuda") * 0.5).to(torch.bfloat16)

    scores = dsa_index_scores_reference(h1, q_lora_n, w, d)

    # hand loop
    t, hi, di, rope = d.tokens, d.index_n_heads, d.index_head_dim, d.qk_rope_dim
    pos = ops.positions_for(d.seq_spec, t, h1.device)
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


def test_topk_padding_is_mask_safe_and_ties_smallest_index():
    from dataflow.tasks.dsa_reference import (
        _causal_mask,
        dsa_mask_from_idx,
        dsa_topk_reference,
    )

    d = _Dims(tokens=16, seq_len=16, index_topk=8)
    # crafted scores: all equal -> ties resolve to smallest indices
    scores = torch.zeros(16, 16, device="cuda") + _causal_mask(d, 16, "cuda")
    idx = dsa_topk_reference(scores, 8)
    # row 3 has prefix {0..3}: first 4 slots = 0..3 (ties smallest-index),
    # pad slots = FUTURE indices 4.. (ascending)
    assert idx[3, :4].tolist() == [0, 1, 2, 3]
    assert idx[3, 4:].tolist() == [4, 5, 6, 7]
    mask = dsa_mask_from_idx(idx, d, 16)
    # pad slots re-suppressed: row 3 live set is exactly {0,1,2,3}
    assert (mask[3] == 0).nonzero().flatten().tolist() == [0, 1, 2, 3]
    # a full row (t=15) selects exactly 8 smallest-index entries
    assert (mask[15] == 0).sum().item() == 8
    assert idx[15, :8].tolist() == list(range(8))


def test_mask_form_equals_gather_form_fwd_and_bwd():
    """softmax(QK+M)V over the full set == gather-then-dense-softmax over
    the selected set, outputs AND input gradients (the equivalence the
    optimized gather kernels rely on)."""
    torch.manual_seed(3)
    t, h, qk, k_sel = 64, 2, 24, 16
    from dataflow.tasks.dsa_reference import (
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
    from dataflow.tasks.dsa_reference import (
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

    from dataflow.tasks.dsa_reference import dsa_index_scores_reference

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
    assert rel_l2(s_packed[:64, :64][la], sa[la]) < 1e-6
    assert rel_l2(s_packed[64:, 64:][lb], sb[lb]) < 1e-6
