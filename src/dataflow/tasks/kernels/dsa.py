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
    # torch.topk, NOT stable sort: DeepSeek's own model.py selects via
    # scores.topk(k), so torch's device tie rule IS the model's rule (our
    # earlier smallest-index pin was a MoE-router convention imported by
    # mistake). Ties resolve identically in runtime and reference (both
    # torch.topk on CUDA); per-input determinism unchanged; -inf pad
    # picks remain mathematically irrelevant (scatter-then-causal). Also
    # ~1.4x faster than the radix SORT (selection needs fewer passes).
    k = idx_out.shape[1]
    for r0 in range(0, scores.shape[0], _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, scores.shape[0])
        _, order = torch.topk(scores[r0:r1], k, dim=-1, sorted=True)
        idx_out[r0:r1].copy_(order.to(idx_out.dtype))


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
                           n_heads, head_dim, seq_bounds, bits_by_seq=None,
                           v_head_dim=None):
    del bits_by_seq
    t = q.shape[0]
    dv = v_head_dim if v_head_dim is not None else head_dim
    scale = head_dim ** -0.5
    q3 = q.view(t, n_heads, head_dim)
    k3 = kf.view(t, n_heads, head_dim)
    v3 = vp.view(t, n_heads, dv)
    o3 = out.view(t, n_heads, dv)
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
                           n_heads, head_dim, seq_bounds, out=None,
                           bits_by_seq=None, v_head_dim=None):
    del out, bits_by_seq  # row_dot formulation needs only lse
    t = q.shape[0]
    dv = v_head_dim if v_head_dim is not None else head_dim
    scale = head_dim ** -0.5
    q3 = q.view(t, n_heads, head_dim)
    k3 = kf.view(t, n_heads, head_dim)
    v3 = vp.view(t, n_heads, dv)
    da3 = d_attn.view(t, n_heads, dv)
    dq3 = dq_out.view(t, n_heads, head_dim)
    dk3 = dk_out.view(t, n_heads, head_dim)
    dv3 = dv_out.view(t, n_heads, dv)
    dq_out.zero_(), dk_out.zero_(), dv_out.zero_()
    for lo, hi in seq_bounds:
        kseq = k3[lo:hi].float()
        vseq = v3[lo:hi].float()
        dk_acc = torch.zeros(hi - lo, n_heads, head_dim,
                             dtype=torch.float32, device=q.device)
        dv_acc = torch.zeros(hi - lo, n_heads, dv,
                             dtype=torch.float32, device=q.device)
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


# --- triton (sm120 default): masked-flash sparse core + indexer ---------------
#
# Selection rides a BITMASK: 64-bit words, one word per (row, 64-col tile)
# — built from dsa_idx by the caller (dsa_pack_bits). The sparse core is a
# standard flash tiling (tensor cores, online softmax) that applies the
# bits as an additive -inf mask per tile: at s4k/k=1024 the sparsity
# ratio is only 4x, so flash-grade compute on the dense grid beats
# per-query gather structures (which forfeit tl.dot); index-gather
# kernels become the win at long context (FlashMLA on sm90+, seam
# below). Backward is the two-pass flash pattern: row-major dq, column-
# major dk/dv with exclusive tile ownership — deterministic, no atomics.

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None


def _pack_bits_or(kctx, idx, bits_out, *, seq_bounds):
    """OR-correct bit packer: one-hot bool (R, L) then 64-bit pack via
    matmul-free reduction. (R, L) bool at s4k = 16 MB/seq transient."""
    bits_out.zero_()
    t = idx.shape[0]
    device = idx.device
    for lo, hi in seq_bounds:
        length = hi - lo
        r = hi - lo
        local = (idx[lo:hi].long() - lo).clamp_(0, length - 1)
        rows = torch.arange(r, device=device).unsqueeze(1)
        onehot = torch.zeros(r, length, dtype=torch.bool, device=device)
        onehot.scatter_(1, local, True)
        # causal re-suppression (pad indices point at future rows)
        cols = torch.arange(length, device=device).unsqueeze(0)
        onehot &= cols <= rows
        words = length // 64 if length % 64 == 0 else length // 64 + 1
        pad = words * 64 - length
        if pad:
            onehot = torch.nn.functional.pad(onehot, (0, pad))
        shifts = (torch.ones(64, dtype=torch.int64, device=device)
                  << torch.arange(64, dtype=torch.int64, device=device))
        packed = (onehot.view(r, words, 64).long() * shifts).sum(-1)
        bits_out[lo:hi, :words] = packed


register("dsa_pack_bits", "eager", deterministic=True,
         workspace=internal(_score_hint), priority=0, allocates="torch",
         fn=_pack_bits_or)


if triton is not None:

    @triton.jit
    def _mf_fwd_kernel(q_ptr, k_ptr, v_ptr, bits_ptr, o_ptr, lse_ptr,
                       L, n_words, scale,
                       sqt, sqh, skt, skh, svt, svh, sot, soh, slh, slt,
                       H: tl.constexpr, D: tl.constexpr,
                       DP: tl.constexpr,
                       DV: tl.constexpr, DVP: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        mmask = rm < L
        rd = tl.arange(0, DP)
        dmask = rd < D
        q = tl.load(q_ptr + rm[:, None] * sqt + pid_h * sqh + rd[None, :],
                    mask=mmask[:, None] & dmask[None, :], other=0.0)
        rv = tl.arange(0, DVP)
        vmask = rv < DV
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, DVP), tl.float32)
        hi_n = (pid_m + 1) * BM
        hi_n = tl.minimum(hi_n, L)
        for n0 in range(0, hi_n, BN):
            word = tl.load(bits_ptr + rm * n_words + (n0 >> 6),
                           mask=mmask, other=0)
            alive = tl.sum((word != 0).to(tl.int32), 0) > 0
            if alive:  # dead tile: zero live bits -> exact no-op for
                # online softmax (p = 0, m/l unchanged) — skip all work
                rn = n0 + tl.arange(0, BN)
                nmask = rn < L
                kt = tl.load(k_ptr + rn[:, None] * skt + pid_h * skh + rd[None, :],
                             mask=nmask[:, None] & dmask[None, :], other=0.0)
                s = tl.dot(q, tl.trans(kt)) * scale
                bit = (word[:, None] >> (rn[None, :] - n0)) & 1
                s = tl.where((bit == 1) & nmask[None, :], s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                vt = tl.load(v_ptr + rn[:, None] * svt + pid_h * svh + rv[None, :],
                             mask=nmask[:, None] & vmask[None, :], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
                m_i = m_new
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        o = acc / l_safe[:, None]
        tl.store(o_ptr + rm[:, None] * sot + pid_h * soh + rv[None, :],
                 o.to(o_ptr.dtype.element_ty),
                 mask=mmask[:, None] & vmask[None, :])
        lse = m_i + tl.log(l_safe)
        tl.store(lse_ptr + pid_h * slh + rm * slt, lse, mask=mmask)

    @triton.jit
    def _mf_bwd_dq_kernel(do_ptr, q_ptr, k_ptr, v_ptr, o_ptr, bits_ptr,
                          lse_ptr, dq_ptr, delta_ptr,
                          L, n_words, scale,
                          sqt, sqh, skt, skh, svt, svh, sot, soh, slh, slt,
                          H: tl.constexpr, D: tl.constexpr,
                          DP: tl.constexpr,
                          DV: tl.constexpr, DVP: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        mmask = rm < L
        rd = tl.arange(0, DP)
        dmask = rd < D
        rv = tl.arange(0, DVP)
        vmask = rv < DV
        q = tl.load(q_ptr + rm[:, None] * sqt + pid_h * sqh + rd[None, :],
                    mask=mmask[:, None] & dmask[None, :], other=0.0)
        do = tl.load(do_ptr + rm[:, None] * sot + pid_h * soh + rv[None, :],
                     mask=mmask[:, None] & vmask[None, :], other=0.0)
        o = tl.load(o_ptr + rm[:, None] * sot + pid_h * soh + rv[None, :],
                    mask=mmask[:, None] & vmask[None, :], other=0.0
                    ).to(tl.float32)
        delta = tl.sum(do.to(tl.float32) * o, 1)
        tl.store(delta_ptr + pid_h * L + rm, delta, mask=mmask)
        lse = tl.load(lse_ptr + pid_h * slh + rm * slt, mask=mmask, other=0.0)
        dq_acc = tl.zeros((BM, DP), tl.float32)
        hi_n = tl.minimum((pid_m + 1) * BM, L)
        for n0 in range(0, hi_n, BN):
            word = tl.load(bits_ptr + rm * n_words + (n0 >> 6),
                           mask=mmask, other=0)
            alive = tl.sum((word != 0).to(tl.int32), 0) > 0
            if alive:  # dead tile contributes exactly zero dq
                rn = n0 + tl.arange(0, BN)
                nmask = rn < L
                kt = tl.load(k_ptr + rn[:, None] * skt + pid_h * skh + rd[None, :],
                             mask=nmask[:, None] & dmask[None, :], other=0.0)
                s = tl.dot(q, tl.trans(kt)) * scale
                bit = (word[:, None] >> (rn[None, :] - n0)) & 1
                live = (bit == 1) & nmask[None, :]
                p = tl.where(live, tl.exp(s - lse[:, None]), 0.0)
                vt = tl.load(v_ptr + rn[:, None] * svt + pid_h * svh + rv[None, :],
                             mask=nmask[:, None] & vmask[None, :], other=0.0)
                dp = tl.dot(do.to(vt.dtype), tl.trans(vt))
                ds = p * (dp - delta[:, None]) * scale
                dq_acc += tl.dot(ds.to(kt.dtype), kt)
        tl.store(dq_ptr + rm[:, None] * sqt + pid_h * sqh + rd[None, :],
                 dq_acc.to(dq_ptr.dtype.element_ty),
                 mask=mmask[:, None] & dmask[None, :])

    @triton.jit
    def _mf_bwd_dkv_kernel(do_ptr, q_ptr, k_ptr, v_ptr, bits_ptr,
                           lse_ptr, dk_ptr, dv_ptr, delta_ptr,
                           L, n_words, scale,
                           sqt, sqh, skt, skh, svt, svh, sot, soh, slh, slt,
                           H: tl.constexpr, D: tl.constexpr,
                           DP: tl.constexpr,
                           DV: tl.constexpr, DVP: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        nmask = rn < L
        rd = tl.arange(0, DP)
        dmask = rd < D
        rv = tl.arange(0, DVP)
        vmask = rv < DV
        kt = tl.load(k_ptr + rn[:, None] * skt + pid_h * skh + rd[None, :],
                     mask=nmask[:, None] & dmask[None, :], other=0.0)
        vt = tl.load(v_ptr + rn[:, None] * svt + pid_h * svh + rv[None, :],
                     mask=nmask[:, None] & vmask[None, :], other=0.0)
        dk_acc = tl.zeros((BN, DP), tl.float32)
        dv_acc = tl.zeros((BN, DVP), tl.float32)
        for m0 in range(pid_n * BN, L, BM):
            rm = m0 + tl.arange(0, BM)
            mmask = rm < L
            word = tl.load(bits_ptr + rm * n_words + ((pid_n * BN) >> 6),
                           mask=mmask, other=0)
            alive = tl.sum((word != 0).to(tl.int32), 0) > 0
            if alive:  # dead (m, n) tile contributes zero dk/dv
                q = tl.load(q_ptr + rm[:, None] * sqt + pid_h * sqh + rd[None, :],
                            mask=mmask[:, None] & dmask[None, :], other=0.0)
                do = tl.load(do_ptr + rm[:, None] * sot + pid_h * soh + rv[None, :],
                             mask=mmask[:, None] & vmask[None, :], other=0.0)
                delta = tl.load(delta_ptr + pid_h * L + rm, mask=mmask,
                                other=0.0)
                lse = tl.load(lse_ptr + pid_h * slh + rm * slt, mask=mmask,
                              other=0.0)
                s = tl.dot(q, tl.trans(kt)) * scale
                bit = (word[:, None] >> (rn[None, :] - pid_n * BN)) & 1
                live = (bit == 1) & nmask[None, :] & mmask[:, None]
                p = tl.where(live, tl.exp(s - lse[:, None]), 0.0)
                dv_acc += tl.dot(tl.trans(p.to(vt.dtype)), do)
                dp = tl.dot(do, tl.trans(vt))
                ds = p * (dp - delta[:, None]) * scale
                dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)
        tl.store(dk_ptr + rn[:, None] * skt + pid_h * skh + rd[None, :],
                 dk_acc.to(dk_ptr.dtype.element_ty),
                 mask=nmask[:, None] & dmask[None, :])
        tl.store(dv_ptr + rn[:, None] * svt + pid_h * svh + rv[None, :],
                 dv_acc.to(dv_ptr.dtype.element_ty),
                 mask=nmask[:, None] & vmask[None, :])

    @register("dsa_sparse_attn_fwd", "triton", deterministic=True,
              workspace=internal(_score_hint),
              requires=lambda c: c.get("triton"), priority=10,
              allocates="torch")
    # static sweep at s4k mini dims (dead-tile skip active), 2026-07-07:
    # fwd (BM=64, w=4, s=3) 0.52 ms; dq (64, 8, 3) 0.77; dkv (128, 8, 2)
    # 0.97 — BN pinned at 64 by the bitmask word width
    def _triton_sparse_fwd(kctx, q, kf, vp, idx, out, lse_out, *,
                           n_heads, head_dim, seq_bounds, bits_by_seq=None,
                           v_head_dim=None):
        t = q.shape[0]
        dv = v_head_dim if v_head_dim is not None else head_dim
        BM, BN = 64, 64
        for si, (lo, hi) in enumerate(seq_bounds):
            length = hi - lo
            words = (length + 63) // 64
            if bits_by_seq is not None:
                bits = bits_by_seq[si]
            else:
                bits = torch.empty(length, words, dtype=torch.int64,
                                   device=q.device)
                _pack_local_bits(idx[lo:hi], lo, length, bits)
            grid = ((length + BM - 1) // BM, n_heads)
            _mf_fwd_kernel[grid](
                q[lo:hi], kf[lo:hi], vp[lo:hi], bits, out[lo:hi],
                lse_out[:, lo:hi],
                length, words, head_dim ** -0.5,
                q.stride(0), head_dim, kf.stride(0), head_dim,
                vp.stride(0), dv, out.stride(0), dv,
                lse_out.stride(0), 1,
                H=n_heads, D=head_dim, DP=triton.next_power_of_2(head_dim),
                DV=dv, DVP=triton.next_power_of_2(dv),
                BM=BM, BN=BN,
                num_warps=4, num_stages=3,
            )
        return out

    def _pack_local_bits(idx_slice, lo, length, bits_out):
        device = idx_slice.device
        r = idx_slice.shape[0]
        rows = torch.arange(r, device=device).unsqueeze(1)
        local = (idx_slice.long() - lo).clamp_(0, length - 1)
        # causal re-suppression of pad indices (point at future rows):
        # clamp them onto the row's own diagonal (always a legal bit)
        local = torch.minimum(local, rows)
        srt, _ = torch.sort(local, dim=1)
        uniq = torch.ones_like(srt, dtype=torch.bool)
        uniq[:, 1:] = srt[:, 1:] != srt[:, :-1]
        contrib = torch.where(
            uniq, torch.ones_like(srt) << (srt & 63), torch.zeros_like(srt),
        )
        bits_out.zero_()
        bits_out.scatter_add_(1, srt >> 6, contrib)

    @register("dsa_sparse_attn_bwd", "triton", deterministic=True,
              workspace=internal(_score_hint),
              requires=lambda c: c.get("triton"), priority=10,
              allocates="torch")
    def _triton_sparse_bwd(kctx, d_attn, q, kf, vp, idx, lse,
                           dq_out, dk_out, dv_out, *,
                           n_heads, head_dim, seq_bounds, out=None,
                           bits_by_seq=None, v_head_dim=None):
        BM, BN = 64, 64
        dv = v_head_dim if v_head_dim is not None else head_dim
        # delta = <dO, O> needs the forward output: rebuilt by the CALLER
        # from ctx attn_out when available (out=), else re-derived here
        if out is None:
            out = torch.empty_like(d_attn)
            lse_chk = torch.empty_like(lse)
            _triton_sparse_fwd(kctx, q, kf, vp, idx, out, lse_chk,
                               n_heads=n_heads, head_dim=head_dim,
                               seq_bounds=seq_bounds, bits_by_seq=bits_by_seq,
                               v_head_dim=v_head_dim)
        for si, (lo, hi) in enumerate(seq_bounds):
            length = hi - lo
            words = (length + 63) // 64
            if bits_by_seq is not None:
                bits = bits_by_seq[si]
            else:
                bits = torch.empty(length, words, dtype=torch.int64,
                                   device=q.device)
                _pack_local_bits(idx[lo:hi], lo, length, bits)
            delta = torch.empty(n_heads, length, dtype=torch.float32,
                                device=q.device)
            grid_m = ((length + BM - 1) // BM, n_heads)
            _mf_bwd_dq_kernel[grid_m](
                d_attn[lo:hi], q[lo:hi], kf[lo:hi], vp[lo:hi], out[lo:hi],
                bits, lse[:, lo:hi], dq_out[lo:hi], delta,
                length, words, head_dim ** -0.5,
                q.stride(0), head_dim, kf.stride(0), head_dim,
                vp.stride(0), dv, d_attn.stride(0), dv,
                lse.stride(0), 1,
                H=n_heads, D=head_dim, DP=triton.next_power_of_2(head_dim),
                DV=dv, DVP=triton.next_power_of_2(dv),
                BM=BM, BN=BN,
                num_warps=8, num_stages=3,
            )
            grid_n = ((length + BN - 1) // BN, n_heads)
            _mf_bwd_dkv_kernel[grid_n](
                d_attn[lo:hi], q[lo:hi], kf[lo:hi], vp[lo:hi],
                bits, lse[:, lo:hi], dk_out[lo:hi], dv_out[lo:hi],
                delta,
                length, words, head_dim ** -0.5,
                q.stride(0), head_dim, kf.stride(0), head_dim,
                vp.stride(0), dv, d_attn.stride(0), dv,
                lse.stride(0), 1,
                H=n_heads, D=head_dim, DP=triton.next_power_of_2(head_dim),
                DV=dv, DVP=triton.next_power_of_2(dv),
                BM=128, BN=BN,
                num_warps=8, num_stages=2,
            )

    @triton.jit
    def _idx_scores_kernel(q_ptr, k_ptr, w_ptr, s_ptr,
                           L, s_stride,
                           HI: tl.constexpr, DI: tl.constexpr,
                           DIP: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        if pid_n * BN > (pid_m + 1) * BM:  # fully above causal diag
            return
        mmask = rm < L
        nmask = rn < L
        rd = tl.arange(0, DIP)
        dmask = rd < DI
        acc = tl.zeros((BM, BN), tl.float32)
        for h in range(HI):
            q = tl.load(q_ptr + rm[:, None] * (HI * DI) + h * DI + rd[None, :],
                        mask=mmask[:, None] & dmask[None, :], other=0.0)
            k = tl.load(k_ptr + rn[:, None] * DI + rd[None, :],
                        mask=nmask[:, None] & dmask[None, :], other=0.0)
            r = tl.dot(q, tl.trans(k))
            r = tl.maximum(r, 0.0)
            w = tl.load(w_ptr + rm * HI + h, mask=mmask, other=0.0)
            acc += w[:, None] * r
        causal = rn[None, :] <= rm[:, None]
        acc = tl.where(causal & nmask[None, :], acc, float("-inf"))
        tl.store(s_ptr + rm[:, None] * s_stride + rn[None, :], acc,
                 mask=mmask[:, None])

    @register("dsa_index_scores", "triton", deterministic=True,
              workspace=internal(_score_hint),
              requires=lambda c: c.get("triton"), priority=10,
              allocates="torch")
    def _triton_index_scores(kctx, q_idx, k_idx, wts, scores_out, *,
                             n_heads, head_dim, seq_bounds):
        BM, BN = 64, 64
        if scores_out.shape[0] != (seq_bounds[-1][1] - seq_bounds[0][0]) \
                or len(seq_bounds) > 1:
            scores_out.fill_(float("-inf"))  # cross-sequence blocks
        for lo, hi in seq_bounds:
            length = hi - lo
            s_slice = scores_out[lo:hi, lo:hi] if scores_out.shape[0] != length \
                else scores_out
            grid = ((length + BM - 1) // BM, (length + BN - 1) // BN)
            _idx_scores_kernel[grid](
                q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], s_slice,
                length, s_slice.stride(0),
                HI=n_heads, DI=head_dim,
                DIP=triton.next_power_of_2(head_dim), BM=BM, BN=BN,
                num_warps=4, num_stages=2,
            )


if triton is not None:

    @triton.jit
    def _idx_bwd_dq_kernel(ds_ptr, q_ptr, k_ptr, w_ptr, dq_ptr, dw_ptr,
                           L, s_stride,
                           HI: tl.constexpr, DI: tl.constexpr,
                           DIP: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr):
        # row-major: dq[m,h,:] and dw[m,h] owned by this program — exclusive
        pid_m = tl.program_id(0)
        rm = pid_m * BM + tl.arange(0, BM)
        mmask = rm < L
        rd = tl.arange(0, DIP)
        dmask = rd < DI
        hi_n = tl.minimum((pid_m + 1) * BM, L)
        for h in range(HI):
            q = tl.load(q_ptr + rm[:, None] * (HI * DI) + h * DI + rd[None, :],
                        mask=mmask[:, None] & dmask[None, :], other=0.0)
            w = tl.load(w_ptr + rm * HI + h, mask=mmask, other=0.0)
            dq_acc = tl.zeros((BM, DIP), tl.float32)
            dw_acc = tl.zeros((BM,), tl.float32)
            for n0 in range(0, hi_n, BN):
                rn = n0 + tl.arange(0, BN)
                nmask = rn < L
                dI = tl.load(ds_ptr + rm[:, None] * s_stride + rn[None, :],
                             mask=mmask[:, None] & nmask[None, :], other=0.0)
                k = tl.load(k_ptr + rn[:, None] * DI + rd[None, :],
                            mask=nmask[:, None] & dmask[None, :], other=0.0)
                r = tl.dot(q, tl.trans(k))
                relu_r = tl.maximum(r, 0.0)
                dw_acc += tl.sum(dI * relu_r, 1)
                g = tl.where(r > 0, dI * w[:, None], 0.0)
                dq_acc += tl.dot(g.to(k.dtype), k)
            tl.store(dq_ptr + rm[:, None] * (HI * DI) + h * DI + rd[None, :],
                     dq_acc.to(dq_ptr.dtype.element_ty),
                     mask=mmask[:, None] & dmask[None, :])
            tl.store(dw_ptr + rm * HI + h, dw_acc, mask=mmask)

    @triton.jit
    def _idx_bwd_dk_kernel(ds_ptr, q_ptr, k_ptr, w_ptr, dk_ptr,
                           L, s_stride,
                           HI: tl.constexpr, DI: tl.constexpr,
                           DIP: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr):
        # column-major: dk[n,:] owned exclusively; heads accumulated inside
        pid_n = tl.program_id(0)
        rn = pid_n * BN + tl.arange(0, BN)
        nmask = rn < L
        rd = tl.arange(0, DIP)
        dmask = rd < DI
        dk_acc = tl.zeros((BN, DIP), tl.float32)
        for h in range(HI):
            k = tl.load(k_ptr + rn[:, None] * DI + rd[None, :],
                        mask=nmask[:, None] & dmask[None, :], other=0.0)
            for m0 in range(pid_n * BN, L, BM):
                rm = m0 + tl.arange(0, BM)
                mmask = rm < L
                dI = tl.load(ds_ptr + rm[:, None] * s_stride + rn[None, :],
                             mask=mmask[:, None] & nmask[None, :], other=0.0)
                q = tl.load(q_ptr + rm[:, None] * (HI * DI) + h * DI + rd[None, :],
                            mask=mmask[:, None] & dmask[None, :], other=0.0)
                w = tl.load(w_ptr + rm * HI + h, mask=mmask, other=0.0)
                r = tl.dot(q, tl.trans(k))
                g = tl.where(r > 0, dI * w[:, None], 0.0)
                dk_acc += tl.dot(tl.trans(g.to(q.dtype)), q)
        tl.store(dk_ptr + rn[:, None] * DI + rd[None, :],
                 dk_acc.to(dk_ptr.dtype.element_ty),
                 mask=nmask[:, None] & dmask[None, :])

    @register("dsa_index_bwd", "triton", deterministic=True,
              workspace=internal(_score_hint),
              requires=lambda c: c.get("triton"), priority=10,
              allocates="torch")
    def _triton_index_bwd(kctx, d_scores, q_idx, k_idx, wts,
                          dq_out, dk_out, dwts_out, *,
                          n_heads, head_dim, seq_bounds):
        BM, BN = 64, 64
        for lo, hi in seq_bounds:
            length = hi - lo
            ds = d_scores[lo:hi, lo:hi] if d_scores.shape[0] != length \
                else d_scores
            grid_m = ((length + BM - 1) // BM,)
            _idx_bwd_dq_kernel[grid_m](
                ds, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi],
                dq_out[lo:hi], dwts_out[lo:hi],
                length, ds.stride(0),
                HI=n_heads, DI=head_dim,
                DIP=triton.next_power_of_2(head_dim), BM=BM, BN=BN,
                num_warps=4, num_stages=4,
            )
            grid_n = ((length + BN - 1) // BN,)
            _idx_bwd_dk_kernel[grid_n](
                ds, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], dk_out[lo:hi],
                length, ds.stride(0),
                HI=n_heads, DI=head_dim,
                DIP=triton.next_power_of_2(head_dim), BM=BM, BN=BN,
                num_warps=4, num_stages=2,
            )

    @triton.jit
    def _probs_sum_kernel(q_ptr, k_ptr, bits_ptr, lse_ptr, p_ptr,
                          L, n_words, p_stride, scale,
                          sqt, sqh, skt, skh, slh, slt,
                          H: tl.constexpr, D: tl.constexpr,
                          DP: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr):
        # head-summed masked attention probabilities: one (BM,BN) tile per
        # program, ALL heads accumulated inside — exclusive tile ownership
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_n * BN > (pid_m + 1) * BM:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mmask = rm < L
        nmask = rn < L
        rd = tl.arange(0, DP)
        dmask = rd < D
        word = tl.load(bits_ptr + rm * n_words + ((pid_n * BN) >> 6),
                       mask=mmask, other=0)
        alive = tl.sum((word != 0).to(tl.int32), 0) > 0
        if not alive:
            return  # p_out pre-zeroed by the launcher
        bit = (word[:, None] >> (rn[None, :] - pid_n * BN)) & 1
        live = (bit == 1) & mmask[:, None] & nmask[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for h in range(H):
            q = tl.load(q_ptr + rm[:, None] * sqt + h * sqh + rd[None, :],
                        mask=mmask[:, None] & dmask[None, :], other=0.0)
            k = tl.load(k_ptr + rn[:, None] * skt + h * skh + rd[None, :],
                        mask=nmask[:, None] & dmask[None, :], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            lse = tl.load(lse_ptr + h * slh + rm * slt, mask=mmask, other=0.0)
            acc += tl.where(live, tl.exp(s - lse[:, None]), 0.0)
        tl.store(p_ptr + rm[:, None] * p_stride + rn[None, :], acc,
                 mask=mmask[:, None] & nmask[None, :])

    def _dsa_probs_sum(kctx, q, kf, idx, lse, p_out, *,
                       n_heads, head_dim, seq_bounds, bits_by_seq=None):
        """Head-summed masked attention probs (t-local (L,L) fp32 per seq)
        — the indexer KL target's expensive half, at flash-tile speeds."""
        BM, BN = 64, 64
        for si, (lo, hi) in enumerate(seq_bounds):
            length = hi - lo
            words = (length + 63) // 64
            if bits_by_seq is not None:
                bits = bits_by_seq[si]
            else:
                bits = torch.empty(length, words, dtype=torch.int64,
                                   device=q.device)
                _pack_local_bits(idx[lo:hi], lo, length, bits)
            p_slice = p_out[lo:hi, lo:hi] if p_out.shape[0] != length else p_out
            p_slice.zero_()
            grid = ((length + BM - 1) // BM, (length + BN - 1) // BN)
            _probs_sum_kernel[grid](
                q[lo:hi], kf[lo:hi], bits, lse[:, lo:hi], p_slice,
                length, words, p_slice.stride(0), head_dim ** -0.5,
                q.stride(0), head_dim, kf.stride(0), head_dim,
                lse.stride(0), 1,
                H=n_heads, D=head_dim, DP=triton.next_power_of_2(head_dim),
                BM=BM, BN=BN, num_warps=4, num_stages=2,
            )

    register("dsa_probs_sum", "triton", deterministic=True,
             workspace=internal(_score_hint),
             requires=lambda c: c.get("triton"), priority=10,
             allocates="torch", fn=_dsa_probs_sum)


def _eager_probs_sum(kctx, q, kf, idx, lse, p_out, *,
                     n_heads, head_dim, seq_bounds, bits_by_seq=None):
    del bits_by_seq
    t = q.shape[0]
    scale = head_dim ** -0.5
    q3 = q.view(t, n_heads, head_dim)
    k3 = kf.view(t, n_heads, head_dim)
    for lo, hi in seq_bounds:
        length = hi - lo
        p_slice = p_out[lo:hi, lo:hi] if p_out.shape[0] != length else p_out
        m = _mask_for(idx, lo, hi, lo, hi)
        acc = torch.zeros(length, length, device=q.device)
        for hh in range(n_heads):
            lg = (q3[lo:hi, hh].float() @ k3[lo:hi, hh].float().T) * scale
            acc += torch.exp(lg + m - lse[hh, lo:hi].unsqueeze(1))
        p_slice.copy_(acc.masked_fill(m != 0, 0.0))


register("dsa_probs_sum", "eager", deterministic=True,
         workspace=internal(_score_hint), priority=0, allocates="torch",
         fn=_eager_probs_sum)
