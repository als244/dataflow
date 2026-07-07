"""DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1.

Four ops, per-sequence chunked, all fp32-internal from bf16 storage,
all sync-free (spin-audited; no bincount-class aten traps — counts and
sorts stay on device):

- ``dsa_index_scores(kctx, q_idx (t,H*dI), k_idx (t,dI), wts (t,H) f32,
  scores_out (t,t) f32, *, n_heads, head_dim, seq_bounds)``
      I[t,s] = sum_j wts[t,j] * ReLU(q[t,j] . k[s]) for s<=t in-seq,
      -inf elsewhere (causal + cross-sequence).
- ``dsa_topk(kctx, scores (t,t) f32, idx_out (t,k) i32)``
      stable-desc sort rows -> first k (smallest-index ties; short
      prefixes pad with future indices — pad-safe under scatter+causal
      mask reconstruction).
- ``dsa_sparse_attn_fwd(kctx, q (t,h*qk), kf (t,h*qk), vp (t,h*qk),
  idx (t,k) i32, out (t,h*qk), lse_out (h,t) f32, *, n_heads, head_dim,
  seq_bounds)``
      masked softmax(QK^T*scale + M)V with M rebuilt from idx per
      sequence (scatter 0 + causal -inf); lse = MASKED logsumexp (the
      bwd + indexer-target anchor). Padded-v convention as main MLA.
- ``dsa_sparse_attn_bwd(kctx, d_attn, q, kf, vp, idx, lse, dq_out,
  dk_out, dv_out, *, n_heads, head_dim, seq_bounds)``
      dense-masked attention backward from saved lse (probs recomputed
      exp(logit-lse)); deterministic (no atomics — per-sequence dense
      GEMMs).
- ``dsa_index_bwd(kctx, d_scores (t,t) f32, q_idx, k_idx, wts,
  dq_out, dk_out, dwts_out, *, n_heads, head_dim, seq_bounds)``
      chain dI through the score formula (ReLU sign recomputed);
      d_scores is zero off the live set by construction.

The (t,t) score/prob matrices are PER-SEQUENCE (seq_bounds host ints,
plan-time constants) and row-chunked; nothing materializes across
sequences. Optimized triton/gather-absorbed/FlashMLA impls arrive in
M-H2 behind the same op names.
"""
from __future__ import annotations

import torch

from .registry import internal, register

_ROW_CHUNK = 1024


def _score_hint(*tensors) -> int:
    t = tensors[0].shape[0]
    return 4 * _ROW_CHUNK * t + 64 * 1024 * 1024


def _eager_index_scores(kctx, q_idx, k_idx, wts, scores_out, *,
                        n_heads, head_dim, seq_bounds):
    t = q_idx.shape[0]
    scores_out.fill_(float("-inf"))
    q3 = q_idx.view(t, n_heads, head_dim)
    for lo, hi in seq_bounds:
        kf = k_idx[lo:hi].float()                       # (L, dI)
        for r0 in range(lo, hi, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, hi)
            q = q3[r0:r1].float()                       # (R, H, dI)
            r = torch.einsum("rhd,sd->rhs", q, kf).clamp_min_(0.0)
            blk = torch.einsum("rh,rhs->rs", wts[r0:r1].float(), r)
            # causal within the sequence
            rows = torch.arange(r0, r1, device=blk.device).unsqueeze(1)
            cols = torch.arange(lo, hi, device=blk.device).unsqueeze(0)
            blk = blk.masked_fill(cols > rows, float("-inf"))
            scores_out[r0:r1, lo:hi] = blk


def _eager_topk(kctx, scores, idx_out):
    k = idx_out.shape[1]
    for r0 in range(0, scores.shape[0], _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, scores.shape[0])
        _, order = torch.sort(scores[r0:r1], dim=-1, descending=True, stable=True)
        idx_out[r0:r1].copy_(order[:, :k].to(idx_out.dtype))


def _mask_for(idx, lo, hi, rows_lo, rows_hi):
    """{0,-inf} (R, L) mask for query rows [rows_lo, rows_hi) of the
    sequence [lo, hi): scatter selections then re-add causal."""
    device = idx.device
    r = rows_hi - rows_lo
    m = torch.full((r, hi - lo), float("-inf"), device=device)
    sel = (idx[rows_lo:rows_hi].long() - lo).clamp_(0, hi - lo - 1)
    m.scatter_(-1, sel, 0.0)
    rows = torch.arange(rows_lo, rows_hi, device=device).unsqueeze(1)
    cols = torch.arange(lo, hi, device=device).unsqueeze(0)
    return m.masked_fill_(cols > rows, float("-inf"))


def _eager_sparse_attn_fwd(kctx, q, kf, vp, idx, out, lse_out, *,
                           n_heads, head_dim, seq_bounds):
    t = q.shape[0]
    scale = head_dim ** -0.5
    q3 = q.view(t, n_heads, head_dim)
    k3 = kf.view(t, n_heads, head_dim)
    v3 = vp.view(t, n_heads, head_dim)
    o3 = out.view(t, n_heads, head_dim)
    for lo, hi in seq_bounds:
        for r0 in range(lo, hi, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, hi)
            m = _mask_for(idx, lo, hi, r0, r1)                       # (R, L)
            lg = torch.einsum("rhd,shd->hrs", q3[r0:r1].float(),
                              k3[lo:hi].float()) * scale
            lg = lg + m.unsqueeze(0)
            lse = torch.logsumexp(lg, dim=-1)                        # (h, R)
            p = torch.exp(lg - lse.unsqueeze(-1))
            o3[r0:r1] = torch.einsum("hrs,shd->rhd", p, v3[lo:hi].float()
                                     ).to(o3.dtype)
            lse_out[:, r0:r1] = lse
            del lg, p, m


def _eager_sparse_attn_bwd(kctx, d_attn, q, kf, vp, idx, lse,
                           dq_out, dk_out, dv_out, *,
                           n_heads, head_dim, seq_bounds):
    t = q.shape[0]
    scale = head_dim ** -0.5
    q3 = q.view(t, n_heads, head_dim)
    k3 = kf.view(t, n_heads, head_dim)
    v3 = vp.view(t, n_heads, head_dim)
    da3 = d_attn.view(t, n_heads, head_dim)
    dq3 = dq_out.view(t, n_heads, head_dim)
    dk3 = dk_out.view(t, n_heads, head_dim)
    dv3 = dv_out.view(t, n_heads, head_dim)
    dq_out.zero_(), dk_out.zero_(), dv_out.zero_()
    for lo, hi in seq_bounds:
        kseq = k3[lo:hi].float()
        vseq = v3[lo:hi].float()
        dk_acc = torch.zeros(hi - lo, n_heads, head_dim,
                             dtype=torch.float32, device=q.device)
        dv_acc = torch.zeros_like(dk_acc)
        for r0 in range(lo, hi, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, hi)
            m = _mask_for(idx, lo, hi, r0, r1)
            lg = torch.einsum("rhd,shd->hrs", q3[r0:r1].float(), kseq) * scale
            p = torch.exp(lg + m.unsqueeze(0) - lse[:, r0:r1].unsqueeze(-1))
            da = da3[r0:r1].float()
            dp = torch.einsum("rhd,shd->hrs", da, vseq)
            row_dot = (dp * p).sum(-1, keepdim=True)
            ds = p * (dp - row_dot) * scale
            dq3[r0:r1] = torch.einsum("hrs,shd->rhd", ds, kseq).to(dq3.dtype)
            dk_acc += torch.einsum("hrs,rhd->shd", ds, q3[r0:r1].float())
            dv_acc += torch.einsum("hrs,rhd->shd", p, da)
            del lg, p, dp, ds, m
        dk3[lo:hi] = dk_acc.to(dk3.dtype)
        dv3[lo:hi] = dv_acc.to(dv3.dtype)


def _eager_index_bwd(kctx, d_scores, q_idx, k_idx, wts,
                     dq_out, dk_out, dwts_out, *,
                     n_heads, head_dim, seq_bounds):
    t = q_idx.shape[0]
    q3 = q_idx.view(t, n_heads, head_dim)
    dq3 = dq_out.view(t, n_heads, head_dim)
    dq_out.zero_(), dk_out.zero_(), dwts_out.zero_()
    for lo, hi in seq_bounds:
        kf = k_idx[lo:hi].float()
        dk_acc = torch.zeros(hi - lo, head_dim, dtype=torch.float32,
                             device=q_idx.device)
        for r0 in range(lo, hi, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, hi)
            dI = d_scores[r0:r1, lo:hi]                              # (R, L) f32
            q = q3[r0:r1].float()
            r = torch.einsum("rhd,sd->rhs", q, kf)
            relu_mask = (r > 0).float()
            relu_r = r.clamp_min(0.0)
            w = wts[r0:r1].float()                                   # (R, H)
            dwts_out[r0:r1] += torch.einsum("rs,rhs->rh", dI, relu_r
                                            ).to(dwts_out.dtype)
            g = dI.unsqueeze(1) * w.unsqueeze(-1) * relu_mask        # (R, H, L)
            dq3[r0:r1] = torch.einsum("rhs,sd->rhd", g, kf).to(dq3.dtype)
            dk_acc += torch.einsum("rhs,rhd->sd", g, q)
            del r, relu_mask, relu_r, g, dI
        dk_out[lo:hi] += dk_acc.to(dk_out.dtype)


for _op, _fn in (
    ("dsa_index_scores", _eager_index_scores),
    ("dsa_topk", _eager_topk),
    ("dsa_sparse_attn_fwd", _eager_sparse_attn_fwd),
    ("dsa_sparse_attn_bwd", _eager_sparse_attn_bwd),
    ("dsa_index_bwd", _eager_index_bwd),
):
    register(_op, "eager", deterministic=True, workspace=internal(_score_hint),
             priority=0, allocates="torch", fn=_fn)
