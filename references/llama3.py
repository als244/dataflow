"""Independent llama3 — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

This is the correctness GROUND TRUTH for the pretraining parity study. It is
deliberately isolated: it imports ONLY ``torch`` (nothing from ``dataflow``),
reads like a normal transformer, and lets autograd derive the backward pass.
The engine must reproduce this model's loss curve from a byte-identical init
on the identical data stream; a from-scratch second implementation also
guards against a shared bug in the engine's hand-written reference ops.

Numeric conventions are chosen to MATCH the engine (so the curves track to
within bf16 kernel-order noise), not a textbook fp32 model:
  - weights/activations bf16; RMSNorm, RoPE, softmax and the CE loss reduce
    in fp32 then cast back (standard mixed precision, and exactly what the
    engine's kernels do);
  - RMSNorm eps 1e-5; RoPE is llama rotate-half with base 500000;
  - GQA via key/value head repeat; causal attention per sequence;
  - SwiGLU MLP (silu(gate) * up); untied LM head.

Weight ORIENTATION: projections are ``nn.Linear`` (weight ``(out, in)``), so
``linear(x) == x @ packed_weight`` when the bridge loads
``linear.weight = packed_weight.T`` (the engine stores ``(in, out)``). The
embedding and LM-head tables are ``(vocab, d)`` and load directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5


@dataclass(frozen=True)
class Llama3Config:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    rope_base: float = 500_000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((xf * rstd).to(x.dtype) * self.weight).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(seq_len: int, head_dim: int, base: float, device,
                dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) each ``(seq_len, head_dim)`` — positions reset per sequence
    (every row is an independent length-``seq_len`` sequence)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                       dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: ``(B, T, H, head_dim)``; cos/sin: ``(T, head_dim)`` fp32."""
    xf = x.float()
    out = xf * cos[None, :, None, :] + _rotate_half(xf) * sin[None, :, None, :]
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, KV, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = apply_rope(self.wq(x).view(B, T, H, hd), cos, sin)
        k = apply_rope(self.wk(x).view(B, T, KV, hd), cos, sin)
        v = self.wv(x).view(B, T, KV, hd)
        rep = H // KV
        q = q.transpose(1, 2)                                   # (B, H, T, hd)
        k = k.repeat_interleave(rep, dim=2).transpose(1, 2)
        v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(o)


class MLP(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # down

    def forward(self, x):
        gate, up = self.w1(x), self.w3(x)
        act = (F.silu(gate.float()).to(gate.dtype) * up)
        return self.w2(act)


class Block(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        h = x + self.attn(self.attn_norm(x), cos, sin)
        return h + self.mlp(self.ffn_norm(h))


class Llama3(nn.Module):
    """Untied-embedding llama3. ``forward`` takes ``(B, T)`` int tokens where
    each row is an independent causal sequence (uniform packing)."""

    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # recompute each block in the backward (activation checkpointing) to
        # trade compute for memory — only needed for the largest model on a
        # single card; off by default.
        self.grad_checkpoint = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.embed(tokens)
        cos, sin = rope_tables(T, self.cfg.head_dim, self.cfg.rope_base,
                               x.device)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.lm_head(self.final_norm(x))

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization. ``tokens``/``targets`` are ``(B, T)``
        int; targets are the next-token ids."""
        logits = self.forward(tokens)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
