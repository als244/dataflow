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
    """Returns (dx, dw). Row-chunked (per-row math is unchanged; only the dw
    accumulation order differs, at fp32 noise level)."""
    dx = torch.empty_like(x)
    dw_acc = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
    wf = w.float()
    t = x.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        xf = x[lo:hi].float()
        dyf = dy[lo:hi].float()
        xhat = xf * rstd[lo:hi].unsqueeze(-1)
        dxhat = dyf * wf
        dw_acc += (dyf * xhat).sum(0)
        dx[lo:hi] = (
            rstd[lo:hi].unsqueeze(-1) * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        ).to(x.dtype)
    return dx, dw_acc.to(w.dtype)


def rmsnorm_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (xf * rstd).to(x.dtype) * w


def rmsnorm_noweight(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's FINAL norm before the LM head (weightless here; llama
    initializes that weight to 1 — adequate for runtime work, documented).
    Returns (normalized bf16, rstd fp32)."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1) + RMS_EPS)
    return (xf * rstd.unsqueeze(-1)).to(x.dtype), rstd


def rmsnorm_noweight_reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (xf * rstd).to(x.dtype)


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
    """x: (tokens, n_heads*head_dim); tokens = batch x seq_len (positions
    restart every seq_len rows)."""
    b = x.shape[0] // seq_len
    cos, sin = _rope_cos_sin(seq_len, head_dim, base, x.device)
    xh = x.view(b, seq_len, n_heads, head_dim).float()
    out = xh * cos[None, :, None, :] + _rotate_half(xh) * sin[None, :, None, :]
    return out.to(x.dtype).view(b * seq_len, n_heads * head_dim)


def rope_bwd(dx: torch.Tensor, seq_len: int, n_heads: int, head_dim: int, base: float) -> torch.Tensor:
    """Gradient through the rotation = rotation by -theta (its transpose)."""
    b = dx.shape[0] // seq_len
    cos, sin = _rope_cos_sin(seq_len, head_dim, base, dx.device)
    dh = dx.view(b, seq_len, n_heads, head_dim).float()
    out = dh * cos[None, :, None, :] - _rotate_half(dh) * sin[None, :, None, :]
    return out.to(dx.dtype).view(b * seq_len, n_heads * head_dim)


# --- flash attention (aten low-level fwd/bwd split) ------------------------------

def flash_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (t, d); k, v: (t, kv); t = batch x seq_len (causal PER SEQUENCE).
    Returns (attn_out (t, d), lse (batch*n_heads, seq_len))."""
    t = q.shape[0]
    s = seq_len or t
    b = t // s
    rep = n_heads // n_kv_heads
    q4 = q.view(b, s, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out, lse, *_rest = torch.ops.aten._scaled_dot_product_flash_attention(
        q4, k4, v4, 0.0, True, return_debug_mask=False
    )
    return (
        out.transpose(1, 2).reshape(t, n_heads * head_dim),
        lse.reshape(b * n_heads, s),
    )


def flash_bwd(
    d_attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_out: torch.Tensor, lse: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dq (t,d), dk (t,kv), dv (t,kv)); GQA head grads reduced."""
    t = q.shape[0]
    s = seq_len or t
    b = t // s
    rep = n_heads // n_kv_heads
    q4 = q.view(b, s, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out4 = attn_out.view(b, s, n_heads, head_dim).transpose(1, 2)
    d4 = d_attn.view(b, s, n_heads, head_dim).transpose(1, 2)
    lse4 = lse.view(b, n_heads, s)
    # dense (non-varlen) path: cum_seqs are undefined; philox is unused with
    # dropout 0 but must be a uint64 pair tensor
    philox = torch.zeros(2, dtype=torch.uint64, device=q.device)
    dq4, dk4, dv4 = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        d4, q4, k4, v4, out4, lse4, None, None, s, s, 0.0, True, philox, philox,
    )
    dq = dq4.transpose(1, 2).reshape(t, n_heads * head_dim)
    dk = dk4.transpose(1, 2).reshape(t, n_heads, head_dim).view(t, n_kv_heads, rep, head_dim).sum(2)
    dv = dv4.transpose(1, 2).reshape(t, n_heads, head_dim).view(t, n_kv_heads, rep, head_dim).sum(2)
    return dq, dk.reshape(t, -1), dv.reshape(t, -1)


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int | None = None,
) -> torch.Tensor:
    t = q.shape[0]
    s = seq_len or t
    b = t // s
    rep = n_heads // n_kv_heads
    q4 = q.view(b, s, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    return out.transpose(1, 2).reshape(t, n_heads * head_dim)


# --- swiglu ---------------------------------------------------------------------

def swiglu_fwd(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """Pure form (autograd-able) — golden references compose this."""
    return torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3


ROWWISE_CHUNK = 2048  # bounds fp32 temporaries of t x d_ff ops to ~chunk x d_ff


def swiglu_fwd_out(x1: torch.Tensor, x3: torch.Tensor, out: torch.Tensor) -> None:
    """Launch form: row-chunked into a preallocated output (an unchunked
    version holds several t x d_ff fp32 temporaries — GBs at batched scale)."""
    t = x1.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        out[lo:hi] = torch.nn.functional.silu(x1[lo:hi].float()).to(x1.dtype) * x3[lo:hi]


def swiglu_bwd(
    ds: torch.Tensor, x1: torch.Tensor, x3: torch.Tensor,
    dx1_out: torch.Tensor | None = None, dx3_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-chunked backward (elementwise: chunking is numerics-identical)."""
    dx1 = dx1_out if dx1_out is not None else torch.empty_like(x1)
    dx3 = dx3_out if dx3_out is not None else torch.empty_like(x3)
    t = x1.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        x1f = x1[lo:hi].float()
        sig = torch.sigmoid(x1f)
        dsf = ds[lo:hi].float()
        dx1[lo:hi] = (dsf * x3[lo:hi].float() * (sig * (1 + x1f * (1 - sig)))).to(x1.dtype)
        dx3[lo:hi] = (dsf * (x1f * sig)).to(x3.dtype)
    return dx1, dx3


# --- fused cross-entropy loss ------------------------------------------------------

CE_CHUNK_ROWS = 1024  # bounds fp32 softmax temporaries to ~2 x chunk x vocab


def ce_loss_fwd_bwd(
    logits: torch.Tensor, targets: torch.Tensor, loss_out: torch.Tensor, dlogits_out: torch.Tensor,
) -> None:
    """Mean CE over tokens; writes fp32 scalar loss and bf16 dlogits.

    Row-chunked: per-row math (logsumexp/softmax) is unchanged; only the
    final mean accumulates across chunks. Unchunked, the fp32 temporaries
    are ~2 x tokens x vocab x 4 bytes (4+ GB at llama vocab)."""
    t = logits.shape[0]
    nll_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    tl = targets.long()
    for lo in range(0, t, CE_CHUNK_ROWS):
        hi = min(lo + CE_CHUNK_ROWS, t)
        lf = logits[lo:hi].float()
        lse = torch.logsumexp(lf, dim=-1, keepdim=True)
        tc = tl[lo:hi]
        nll_sum += (lse.squeeze(-1) - lf.gather(1, tc.unsqueeze(1)).squeeze(1)).sum()
        soft = torch.exp(lf - lse)
        soft.scatter_add_(
            1, tc.unsqueeze(1),
            torch.full((hi - lo, 1), -1.0, device=logits.device, dtype=torch.float32),
        )
        dlogits_out[lo:hi].copy_((soft / t).to(dlogits_out.dtype))
    loss_out.copy_((nll_sum / t).reshape(loss_out.shape))


def ce_loss_reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.float(), targets.long())


# --- adamw ---------------------------------------------------------------------

ADAMW_CHUNK_ELEMS = 1 << 24  # 16.7M elems -> ~400MB of fp32 temporaries max


def adamw_step(
    w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    *, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int,
) -> None:
    """In-place AdamW on flat views. States bf16, math fp32 (the golden
    reference implements the identical update, incl. the bf16 state
    round-trip, so parity is exact in structure).

    Chunked: the update is elementwise, so processing the flat parameter in
    chunks is bit-identical while bounding fp32 temporaries (an unchunked
    embed/head update materializes ~6x param bytes — 12+ GB at vocab scale)."""
    n = w.numel()
    for lo in range(0, n, ADAMW_CHUNK_ELEMS):
        hi = min(lo + ADAMW_CHUNK_ELEMS, n)
        wc, gc, mc, vc = w[lo:hi], g[lo:hi], m[lo:hi], v[lo:hi]
        gf = gc.float()
        mf = mc.float().mul_(beta1).add_(gf, alpha=1 - beta1)
        vf = vc.float().mul_(beta2).addcmul_(gf, gf, value=1 - beta2)
        mc.copy_(mf.to(mc.dtype))
        vc.copy_(vf.to(vc.dtype))
        # bias-corrected using the ROUND-TRIPPED states (what the next step sees)
        mhat = mc.float() / (1 - beta1 ** step)
        vhat = vc.float() / (1 - beta2 ** step)
        wf = wc.float()
        wf -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * wf)
        wc.copy_(wf.to(wc.dtype))


# --- embedding ---------------------------------------------------------------------

def embed_fwd(tokens: torch.Tensor, w_embed: torch.Tensor, out: torch.Tensor) -> None:
    torch.index_select(w_embed, 0, tokens.int(), out=out)


def embed_bwd_accum(tokens: torch.Tensor, dy: torch.Tensor, dw_embed: torch.Tensor, *, zero_first: bool) -> None:
    if zero_first:
        dw_embed.zero_()
    dw_embed.index_add_(0, tokens.int(), dy)
