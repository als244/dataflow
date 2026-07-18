"""Independent GPT-2 — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the gpt2 family: the classic GPT-2
architecture at the llm.c / nanoGPT pretraining convention (the nanogpt
speedrun's baseline model). Deliberately isolated: imports ONLY ``torch``,
reads like a normal transformer, autograd derives the backward.

Architecture (GPT-2 124M shape, radford2019):
  - LEARNED positional embeddings (wpe) added to token embeddings — no rope;
  - pre-LN blocks with LayerNorm (learned gain AND bias, eps 1e-5);
  - fused QKV projection (c_attn, one (d, 3d) matrix + bias), full MHA
    (no GQA), causal softmax attention at 1/sqrt(head_dim);
  - MLP: c_fc (d, 4d) + bias, GELU (tanh approximation — the GPT-2/llm.c
    form), c_proj (4d, d) + bias;
  - final LayerNorm; LM head UNTIED by default (the repo convention shared
    with the llama3 baselines; modded-nanogpt also unties) — ``tied=True``
    restores the classic GPT-2 weight tying as a config option.

Numeric conventions MATCH the engine (bf16 storage, fp32 reductions):
  - LayerNorm reduces in fp32; the normalized value rounds to the storage
    dtype BEFORE the affine (the repo's norm-kernel convention — one ulp
    from ``nn.LayerNorm``'s round-after-affine, applied consistently on
    both sides of the parity gates);
  - GELU/softmax/CE compute at torch opmath (fp32 internally for bf16);
  - mean CE over all tokens in fp32.

Weight ORIENTATION: projections are ``nn.Linear`` (weight ``(out, in)``),
so the bridge loads ``linear.weight = packed_weight.T`` (the engine stores
``(in, out)``); biases and the (vocab, d)/(n_ctx, d) tables load directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-5


@dataclass(frozen=True)
class Gpt2Config:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    vocab_size: int
    n_ctx: int              # learned-position table rows (max sequence length)
    tied: bool = False      # tie lm_head to wte (classic GPT-2; option here)
    # biases in Linears AND LayerNorms (the nanoGPT flag): True = classic
    # GPT-2; False = the bias-free variant (like llama3; the speedrun's
    # own first simplification)
    use_bias: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class LayerNorm(nn.Module):
    """fp32 mean/var; normalized value rounds to the input dtype BEFORE
    the affine (matching the engine's layernorm kernels)."""

    def __init__(self, dim: int, eps: float = LN_EPS, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(-1, keepdim=True)
        rstd = torch.rsqrt((xf - mean).pow(2).mean(-1, keepdim=True) + self.eps)
        xhat = ((xf - mean) * rstd).to(x.dtype)
        out = xhat * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


def packed_positions(seq_lens, device) -> torch.Tensor:
    """Per-token positions for a PACKED round: every sequence restarts
    at 0 — the wpe rows each token reads (varlen mode)."""
    return torch.cat([torch.arange(n, device=device) for n in seq_lens])


def block_causal_mask(seq_lens, device) -> torch.Tensor:
    """(T, T) additive {0, -inf} fp32 mask for a packed round: causal
    WITHIN each sequence, -inf across sequences."""
    t = int(sum(seq_lens))
    m = torch.full((t, t), float("-inf"), device=device)
    lo = 0
    for n in seq_lens:
        m[lo:lo + n, lo:lo + n] = torch.triu(
            torch.full((n, n), float("-inf"), device=device), diagonal=1)
        lo += n
    return m


class Attention(nn.Module):
    def __init__(self, cfg: Gpt2Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.use_bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_bias)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        H, hd = self.n_heads, self.head_dim
        q, k, v = self.c_attn(x).split(d, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)                 # (B, H, T, hd)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:       # packed varlen: block-diagonal causality lives in the mask
            o = F.scaled_dot_product_attention(q, k, v,
                                               attn_mask=mask.to(q.dtype))
        o = o.transpose(1, 2).reshape(B, T, d)
        return self.c_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: Gpt2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.use_bias)
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.use_bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: Gpt2Config):
        super().__init__()
        self.attn_norm = LayerNorm(cfg.d_model, bias=cfg.use_bias)
        self.attn = Attention(cfg)
        self.ffn_norm = LayerNorm(cfg.d_model, bias=cfg.use_bias)
        self.mlp = MLP(cfg)

    def forward(self, x, mask=None):
        h = x + self.attn(self.attn_norm(x), mask)
        return h + self.mlp(self.ffn_norm(h))


class Gpt2(nn.Module):
    """Untied-embedding GPT-2. ``forward`` takes ``(B, T)`` int tokens where
    each row is an independent causal sequence starting at position 0
    (uniform packing); pass ``seq_lens`` with a ``(1, sum(seq_lens))``
    packed row for native varlen (per-sequence positions + block-diagonal
    attention — the engine's packed-round semantics)."""

    SUPPORTS_PACKED = True

    def __init__(self, cfg: Gpt2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = LayerNorm(cfg.d_model, bias=cfg.use_bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied:
            self.lm_head.weight = self.wte.weight

    def forward(self, tokens: torch.Tensor,
                seq_lens: tuple[int, ...] | None = None) -> torch.Tensor:
        B, T = tokens.shape
        if seq_lens is None:
            if T > self.cfg.n_ctx:
                raise ValueError(f"sequence length {T} exceeds n_ctx "
                                 f"{self.cfg.n_ctx} (learned positions)")
            pos = torch.arange(T, device=tokens.device)
            mask = None
        else:
            if B != 1 or T != int(sum(seq_lens)):
                raise ValueError(f"packed mode expects (1, sum(seq_lens)) "
                                 f"tokens; got {tuple(tokens.shape)} for "
                                 f"{seq_lens}")
            if max(seq_lens) > self.cfg.n_ctx:
                raise ValueError(f"segment length {max(seq_lens)} exceeds "
                                 f"n_ctx {self.cfg.n_ctx} (learned positions)")
            pos = packed_positions(seq_lens, tokens.device)
            mask = block_causal_mask(seq_lens, tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lm_head(self.final_norm(x))

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             seq_lens: tuple[int, ...] | None = None) -> torch.Tensor:
        """Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization."""
        logits = self.forward(tokens, seq_lens=seq_lens)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
