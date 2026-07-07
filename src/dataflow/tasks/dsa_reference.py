"""DSA (DeepSeek Sparse Attention) — REFERENCE forms only (DeepSeek-V3.2).

NOT a task/executable: pure-autograd anchors (like mla_reference.py) for
the lightning indexer, top-k selection, sparse-core attention, and the
indexer's KL training loss. Runtime executables live in dsv32_blocks.py.

Conventions pinned here (verified against deepseek-ai/DeepSeek-V3.2-Exp
inference/model.py + kernel.py + the tech report, 2026-07-07):

- Indexer scores (report eq. 1; ReLU confirmed in their fp8_index):
      I_{t,s} = sum_j w~_{t,j} * ReLU(q^I_{t,j} . k^I_s),  s <= t
  with w~ = weights_proj(h1).float() * H_I^-0.5 * d_I^-0.5 (their
  softmax_scale = head_dim^-0.5; fp8 q_scale == 1 in bf16 math).
- q^I from the SHARED post-norm q_lora latent (wq_b: q_lora -> H_I*d_I);
  k^I from h1 via wk then STANDARD LayerNorm (F.layer_norm: mean-
  subtracting, weight AND bias, fp32 internal).
- Rope-FIRST head layout for the indexer: q^I/k^I split as
  [rope_dims | rest] and rope applies to the FIRST qk_rope_dim dims
  (their split order `[rope_head_dim, head_dim - rope_head_dim]`) —
  the OPPOSITE of main MLA's nope-first layout. Non-interleaved
  (rotate-half, ops.rope_fwd). k^I is ONE shared 'head' per token.
- Selection: per-token top-min(k, t+1) of the causally-masked scores,
  stable-sort-descending => smallest-index tie-break. Stored (t, k)
  int32 STATIC; rows with prefix < k pad with FUTURE indices (the -inf
  causal entries, ascending) — pad-SAFE because the mask is built as
  scatter(0 at idx) + causal(-inf) (their construction: causal added
  AFTER scatter re-suppresses any pad slot).
- Sparse core == additive-mask attention (their prefill semantics):
  softmax(QK^T*scale + M) V with M in {0, -inf}. Mathematically equal
  to gather-form attention over S_t (softmax is permutation-invariant
  and pads are -inf).
- Indexer training (report eqs. 3/4): target p_t = main attention
  probs SUMMED OVER HEADS then L1-normalized (DETACHED); KL(p ||
  softmax(I)) over the full prefix (dense mode) or S_t (sparse mode).
  The indexer INPUT is detached: no gradient crosses the seam.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import ops

_LN_EPS = 1e-5  # repo-global norm eps (their 1e-6; standing delta note)


def dsa_index_scores_reference(
    h1: torch.Tensor, q_lora_n: torch.Tensor, w: dict, dims,
) -> torch.Tensor:
    """(t, t) fp32 index scores, -inf strictly above the causal diagonal.
    Autograd flows to the four indexer weights (and h1/q_lora_n — the
    caller detaches those for training parity)."""
    d = dims
    t = h1.shape[0]
    hi, di, rope = d.index_n_heads, d.index_head_dim, d.qk_rope_dim
    pos = ops.positions_for(d.seq_spec, t, h1.device)

    q = (q_lora_n @ w["w_idx_q"]).view(t, hi, di)
    q_pe = ops.rope_fwd(
        q[..., :rope].reshape(t, hi * rope).contiguous(), pos, hi, rope, d.rope_base,
    ).view(t, hi, rope)
    q = torch.cat([q_pe, q[..., rope:]], dim=-1)

    k_pre = h1 @ w["w_idx_k"]
    k = F.layer_norm(
        k_pre.float(), (di,), w["idx_k_ln_w"].float(), w["idx_k_ln_b"].float(),
        _LN_EPS,
    ).to(k_pre.dtype)
    k_pe = ops.rope_fwd(k[:, :rope].contiguous(), pos, 1, rope, d.rope_base)
    k = torch.cat([k_pe, k[:, rope:]], dim=-1)

    # their weights_proj(x.float()): the INPUT is cast to fp32, matmul in fp32
    wts = (h1.float() @ w["w_idx_w"].float()) * (hi ** -0.5) * (di ** -0.5)
    r = torch.einsum("thd,sd->ths", q.float(), k.float())
    scores = (wts.unsqueeze(-1) * r.clamp_min(0.0)).sum(1)           # (t, t)

    # causal (per sequence): s <= t within each sequence
    mask = _causal_mask(d, t, h1.device)
    return scores + mask


def _causal_mask(dims, t: int, device) -> torch.Tensor:
    """(t, t) additive mask: 0 on/below the per-sequence causal diagonal,
    -inf above it AND across sequence boundaries."""
    lens = ops.seq_lens_of(dims.seq_spec, t)
    m = torch.full((t, t), float("-inf"), device=device)
    lo = 0
    for L in lens:
        hi = lo + L
        m[lo:hi, lo:hi] = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1,
        )
        lo = hi
    return m


def dsa_topk_reference(scores: torch.Tensor, k_sel: int) -> torch.Tensor:
    """(t, k_sel) int64 selected indices. Stable descending sort =>
    smallest-index tie-break; short-prefix rows pad with the -inf slots
    (ascending future indices — pad-safe under scatter+causal masks)."""
    _, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    return idx[:, :k_sel]


def dsa_mask_from_idx(idx: torch.Tensor, dims, t: int) -> torch.Tensor:
    """Their construction: scatter 0 at selected, then ADD causal —
    pad slots (future indices) are re-suppressed."""
    m = torch.full((t, t), float("-inf"), device=idx.device)
    m.scatter_(-1, idx, 0.0)
    return m + _causal_mask(dims, t, idx.device)


def dsa_sparse_attention_reference(
    q_full: torch.Tensor, k_full: torch.Tensor, v_pad: torch.Tensor,
    add_mask: torch.Tensor, dims,
) -> torch.Tensor:
    """Masked-SDPA sparse core over the padded-v MLA tensors (t, h*qk):
    per-sequence SDPA with the additive {0,-inf} mask (causality lives
    in the mask). Output (t, h*qk); caller slices [:v]."""
    d = dims
    t = q_full.shape[0]
    h, qk = d.n_heads, d.qk_head_dim
    lens = ops.seq_lens_of(d.seq_spec, t)
    outs = []
    lo = 0
    for L in lens:
        hi = lo + L
        q4 = q_full[lo:hi].view(1, L, h, qk).transpose(1, 2)
        k4 = k_full[lo:hi].view(1, L, h, qk).transpose(1, 2)
        v4 = v_pad[lo:hi].view(1, L, h, qk).transpose(1, 2)
        m = add_mask[lo:hi, lo:hi].view(1, 1, L, L)
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m)
        outs.append(o.transpose(1, 2).reshape(L, h * qk))
        lo = hi
    return torch.cat(outs)


def dsa_indexer_kl_reference(
    index_scores: torch.Tensor, sel_mask: torch.Tensor,
    head_probs_sum: torch.Tensor,
) -> torch.Tensor:
    """L_I = sum_t KL(p_t || softmax(I_t)) over the mask's live set.

    ``sel_mask``: additive {0,-inf} (t, t) — full-causal for dense mode,
    scatter+causal for sparse mode. ``head_probs_sum``: (t, t) fp32 sum
    over main attention heads of the (masked) attention probabilities —
    DETACHED by the caller. Both p and sigma renormalize over the live
    set; -inf slots contribute nothing (0 log 0 := 0)."""
    live = sel_mask == 0
    p = head_probs_sum.masked_fill(~live, 0.0)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
    logsig = torch.log_softmax(index_scores + sel_mask, dim=-1)
    plogp = torch.where(p > 0, p * p.clamp_min(1e-20).log(), p.new_zeros(()))
    return (plogp - p * logsig.masked_fill(~live, 0.0)).sum()
