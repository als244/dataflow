"""FlashMLA seam: DeepSeek's sm90/sm100 sparse-attention forward for DSA.

Verified against the upstream repo (github.com/deepseek-ai/FlashMLA,
fetched 2026-07-07):

- ``flash_mla_sparse_fwd(q, kv, indices, sm_scale) -> (out, max_logits,
  lse)`` — the SPARSE PREFILL kernel, MQA-ABSORBED layout: ``q``
  [s_q, h_q, d_qk], ``kv`` [s_kv, h_kv=1, d_qk], ``indices``
  [s_q, 1, topk]; specialized at d_qk=576 / d_v=512 — EXACTLY the
  absorbed dims of the 671B config (kv_lora 512 + rope 64); no batch
  dim (per-sequence, matching our seq_bounds loop). sm90 + sm100,
  CUDA 12.8+.
- The sparse kernels are FORWARD-ONLY (inference release). Training
  integration therefore uses FlashMLA for fwd + recompute and keeps our
  deterministic masked-flash kernels for the backward.
- sm100 additionally ships dense MHA prefill fwd+bwd — relevant to the
  M-H3 dense-warmup mode on that hardware, not wired here.

This module lands the seam, not the full absorbed pipeline: a NEW op
``dsa_sparse_attn_fwd_absorbed`` in the absorbed layout, registered
behind ``requires=flash_mla`` so it resolves only on capable machines.
The dsv32 blocks keep the MHA-expanded stages on sm120; the
absorbed-mode block variant (latents stay compressed end-to-end,
q-absorption GEMMs, absorbed-space backward + un-absorption chains) is
the big-machine milestone that consumes this op (plan M-H2b).

Assumptions to pin ON the sm90 machine (parity test below, auto-skipped
elsewhere):
- pad slots in ``indices`` are neutralized as -1 (kernel skips
  negatives); if upstream instead requires in-range indices, switch the
  adapter to duplicate-of-self plus a post-hoc lse correction — the
  parity test will say so loudly.
- returned ``lse`` is the natural-log masked logsumexp in [s_q, h_q]
  layout (transposed into our (h, t) convention here).
"""
from __future__ import annotations

import torch

from .registry import internal, register


def _ws_hint(*tensors) -> int:
    return 64 * 1024 * 1024


def _eager_absorbed_fwd(kctx, q_abs, kv, idx, out, lse_out, *,
                        n_heads, d_qk, d_v, seq_bounds):
    """Eager anchor for the absorbed layout (any dims, any device):
    mask-form attention where every head shares ONE (d_qk) key row and
    the value is the first d_v dims of that row. Defines the op's
    semantics so the FlashMLA impl has a golden to match on sm90."""
    t = q_abs.shape[0]
    scale = d_qk ** -0.5
    q3 = q_abs.view(t, n_heads, d_qk)
    for lo, hi in seq_bounds:
        length = hi - lo
        rows = torch.arange(length, device=q_abs.device).unsqueeze(1)
        cols = torch.arange(length, device=q_abs.device).unsqueeze(0)
        m = torch.full((length, length), float("-inf"), device=q_abs.device)
        sel = (idx[lo:hi].long() - lo).clamp_(0, length - 1)
        m.scatter_(-1, sel, 0.0)
        m.masked_fill_(cols > rows, float("-inf"))
        kseq = kv[lo:hi].float()                          # (L, d_qk)
        lg = torch.einsum("rhd,sd->hrs", q3[lo:hi].float(), kseq) * scale
        lg = lg + m.unsqueeze(0)
        lse = torch.logsumexp(lg, dim=-1)
        p = torch.exp(lg - lse.unsqueeze(-1))
        o = torch.einsum("hrs,sd->rhd", p, kseq[:, :d_v])
        out.view(t, n_heads, d_v)[lo:hi] = o.to(out.dtype)
        lse_out[:, lo:hi] = lse


register("dsa_sparse_attn_fwd_absorbed", "eager", deterministic=True,
         workspace=internal(_ws_hint), priority=0, allocates="torch",
         fn=_eager_absorbed_fwd)


def _flashmla_absorbed_fwd(kctx, q_abs, kv, idx, out, lse_out, *,
                           n_heads, d_qk, d_v, seq_bounds):
    import flash_mla

    if (d_qk, d_v) != (576, 512):
        raise ValueError(
            f"flash_mla_sparse_fwd is specialized at (576, 512); got "
            f"({d_qk}, {d_v}) — mini-scale absorbed dims (288/256) must "
            f"zero-pad for validation runs or use the triton/eager impls"
        )
    t = q_abs.shape[0]
    topk = idx.shape[1]
    for lo, hi in seq_bounds:
        length = hi - lo
        q = q_abs[lo:hi].view(length, n_heads, d_qk)
        kv_seq = kv[lo:hi].view(length, 1, d_qk)
        rows = torch.arange(length, device=idx.device).unsqueeze(1)
        local = idx[lo:hi].long() - lo
        # pads (future/self-duplicate slots) -> -1: skipped by the kernel
        local = torch.where(local > rows, local.new_full((), -1), local)
        ind = local.view(length, 1, topk).int().contiguous()
        o, _maxlog, lse = flash_mla.flash_mla_sparse_fwd(
            q.contiguous(), kv_seq.contiguous(), ind, d_qk ** -0.5,
        )
        out.view(t, n_heads, d_v)[lo:hi] = o.view(length, n_heads, d_v)
        lse_out[:, lo:hi] = lse.view(length, n_heads).transpose(0, 1).float()


register("dsa_sparse_attn_fwd_absorbed", "flashmla", deterministic=True,
         workspace=internal(_ws_hint),
         requires=lambda c: c.get("flash_mla"), priority=20,
         allocates="vendor", fn=_flashmla_absorbed_fwd)
