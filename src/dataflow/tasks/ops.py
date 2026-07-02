"""Eager-torch op library for llama-family blocks.

Each op is a pair of plain functions over torch tensors:

- the *launch* form used by executables (writes into provided `out` tensors
  where torch supports it; anything torch self-allocates goes through the
  bounded scratch lane and is measured by the profiling harness), and
- a *reference* form — pure, autograd-able — used by the gradcheck helpers
  and by the composer-derived task references.

Numerics: parameters/activations bf16; reductions and normalization
statistics in fp32 (matching the golden reference model bit-for-bit in
structure, so parity tolerances stay tight).
"""
from __future__ import annotations

import math

import torch

# --- rmsnorm -------------------------------------------------------------------

RMS_EPS = 1e-5


def rmsnorm_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, rstd_out: torch.Tensor) -> None:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1) + RMS_EPS)
    rstd_out.copy_(rstd)
    out.copy_((xf * rstd.unsqueeze(-1)).to(x.dtype) * w)


def rmsnorm_apply(x: torch.Tensor, rstd: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Recompute the normalized output from a saved rstd (cheap, exact)."""
    return (x.float() * rstd.unsqueeze(-1)).to(x.dtype) * w


def rmsnorm_bwd(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (dx, dw)."""
    xf = x.float()
    dyf = dy.float()
    wf = w.float()
    xhat = xf * rstd.unsqueeze(-1)
    dxhat = dyf * wf
    dw = (dyf * xhat).sum(0)
    d = x.shape[-1]
    dx = rstd.unsqueeze(-1) * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdim=True))
    # note: mean over last dim == sum/d; xhat*rstd² == x*rstd³ folded via xhat
    return dx.to(x.dtype), dw.to(w.dtype)


def rmsnorm_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (xf * rstd).to(x.dtype) * w


# --- rope (llama rotate-half) ----------------------------------------------------

def _rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_fwd(x: torch.Tensor, seq_len: int, n_heads: int, head_dim: int, base: float) -> torch.Tensor:
    """x: (tokens, n_heads*head_dim), tokens == seq_len (batch folded upstream)."""
    cos, sin = _rope_cos_sin(seq_len, head_dim, base, x.device)
    xh = x.view(seq_len, n_heads, head_dim).float()
    out = xh * cos[:, None, :] + _rotate_half(xh) * sin[:, None, :]
    return out.to(x.dtype).view(seq_len, n_heads * head_dim)


def rope_bwd(dx: torch.Tensor, seq_len: int, n_heads: int, head_dim: int, base: float) -> torch.Tensor:
    """Gradient through the rotation = rotation by -theta (its transpose)."""
    cos, sin = _rope_cos_sin(seq_len, head_dim, base, dx.device)
    dh = dx.view(seq_len, n_heads, head_dim).float()
    out = dh * cos[:, None, :] - _rotate_half(dh) * sin[:, None, :]
    return out.to(dx.dtype).view(seq_len, n_heads * head_dim)


# --- flash attention (aten low-level fwd/bwd split) ------------------------------

def flash_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (t, d); k, v: (t, kv). Returns (attn_out (t, d), lse (h, t))."""
    t = q.shape[0]
    rep = n_heads // n_kv_heads
    q4 = q.view(1, t, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out, lse, *_rest = torch.ops.aten._scaled_dot_product_flash_attention(
        q4, k4, v4, 0.0, True, return_debug_mask=False
    )
    return (
        out.transpose(1, 2).reshape(t, n_heads * head_dim),
        lse.view(n_heads, t),
    )


def flash_bwd(
    d_attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_out: torch.Tensor, lse: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dq (t,d), dk (t,kv), dv (t,kv)); GQA head grads reduced."""
    t = q.shape[0]
    rep = n_heads // n_kv_heads
    q4 = q.view(1, t, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out4 = attn_out.view(1, t, n_heads, head_dim).transpose(1, 2)
    d4 = d_attn.view(1, t, n_heads, head_dim).transpose(1, 2)
    lse4 = lse.view(1, n_heads, t)
    # dense (non-varlen) path: cum_seqs are undefined; philox is unused with
    # dropout 0 but must be a uint64 pair tensor
    philox = torch.zeros(2, dtype=torch.uint64, device=q.device)
    dq4, dk4, dv4 = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        d4, q4, k4, v4, out4, lse4, None, None, t, t, 0.0, True, philox, philox,
    )
    dq = dq4.transpose(1, 2).reshape(t, n_heads * head_dim)
    dk = dk4.transpose(1, 2).reshape(t, n_heads, head_dim).view(t, n_kv_heads, rep, head_dim).sum(2)
    dv = dv4.transpose(1, 2).reshape(t, n_heads, head_dim).view(t, n_kv_heads, rep, head_dim).sum(2)
    return dq, dk.reshape(t, -1), dv.reshape(t, -1)


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    t = q.shape[0]
    rep = n_heads // n_kv_heads
    q4 = q.view(1, t, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(1, t, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    return out.transpose(1, 2).reshape(t, n_heads * head_dim)


# --- swiglu ---------------------------------------------------------------------

def swiglu_fwd(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3


def swiglu_bwd(ds: torch.Tensor, x1: torch.Tensor, x3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x1f = x1.float()
    sig = torch.sigmoid(x1f)
    silu = x1f * sig
    dsilu = sig * (1 + x1f * (1 - sig))
    dx1 = (ds.float() * x3.float() * dsilu).to(x1.dtype)
    dx3 = (ds.float() * silu).to(x3.dtype)
    return dx1, dx3


# --- fused cross-entropy loss ------------------------------------------------------

def ce_loss_fwd_bwd(
    logits: torch.Tensor, targets: torch.Tensor, loss_out: torch.Tensor, dlogits_out: torch.Tensor,
) -> None:
    """Mean CE over tokens; writes fp32 scalar loss and bf16 dlogits."""
    lf = logits.float()
    lse = torch.logsumexp(lf, dim=-1, keepdim=True)
    t = logits.shape[0]
    nll = (lse.squeeze(-1) - lf.gather(1, targets.long().unsqueeze(1)).squeeze(1)).mean()
    loss_out.copy_(nll.reshape(loss_out.shape))
    soft = torch.exp(lf - lse)
    soft.scatter_add_(
        1, targets.long().unsqueeze(1),
        torch.full((t, 1), -1.0, device=logits.device, dtype=torch.float32),
    )
    dlogits_out.copy_((soft / t).to(dlogits_out.dtype))


def ce_loss_reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.float(), targets.long())


# --- adamw ---------------------------------------------------------------------

def adamw_step(
    w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    *, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int,
) -> None:
    """In-place AdamW on flat views. States bf16, math fp32 (the golden
    reference implements the identical update, incl. the bf16 state
    round-trip, so parity is exact in structure)."""
    gf = g.float()
    mf = m.float().mul_(beta1).add_(gf, alpha=1 - beta1)
    vf = v.float().mul_(beta2).addcmul_(gf, gf, value=1 - beta2)
    m.copy_(mf.to(m.dtype))
    v.copy_(vf.to(v.dtype))
    # bias-corrected using the ROUND-TRIPPED states (what the next step sees)
    mhat = m.float() / (1 - beta1 ** step)
    vhat = v.float() / (1 - beta2 ** step)
    wf = w.float()
    wf -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * wf)
    w.copy_(wf.to(w.dtype))


# --- embedding ---------------------------------------------------------------------

def embed_fwd(tokens: torch.Tensor, w_embed: torch.Tensor, out: torch.Tensor) -> None:
    torch.index_select(w_embed, 0, tokens.int(), out=out)


def embed_bwd_accum(tokens: torch.Tensor, dy: torch.Tensor, dw_embed: torch.Tensor, *, zero_first: bool) -> None:
    if zero_first:
        dw_embed.zero_()
    dw_embed.index_add_(0, tokens.int(), dy)
