"""Independent OLMoE — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the OLMoE arm of the pretraining parity
study, reimplemented from scratch. Like the sibling references it is
deliberately isolated: it imports ONLY ``torch`` (nothing from ``dataflow``,
nothing from the other ``reference_models/`` files — RMSNorm/RoPE/SwiGLU AND the
MoE router+experts+combine are reimplemented locally even though that
duplicates code, because isolation is the point). It reads like a normal MoE
transformer and lets autograd derive the backward pass; a from-scratch second
implementation guards against a shared bug in the engine's reference ops.

OLMoE is qwen3-shaped attention with a routed-SwiGLU MoE FFN in EVERY layer:

  - **attention** — RMSNorm -> wq/wk/wv -> FULL-ROW qk-norm (one RMSNorm over
    the ENTIRE ``(T, q_dim)`` / ``(T, kv_dim)`` projection row: one rstd per
    token and a ``(q_dim,)`` / ``(kv_dim,)`` gain — NOT the per-head qk-norm
    of qwen3) -> full rotate-half RoPE (base 1e4) -> causal attention. No GQA
    at 7B (``n_kv_heads == n_heads``), though the kv-head repeat is kept
    general.
  - **MoE FFN** — router logits (bf16 GEMM) -> routing mode
    SOFTMAX_THEN_TOPK (full softmax over all E experts, then take the top-K
    probabilities UNnormalized, so the K routing weights sum to <= 1;
    smallest-index tie-break) -> DROPLESS per-expert SwiGLU over packed expert
    weights ``w13_experts (E, d, 2F)`` [gate|up] and ``w2_experts (E, F, d)``
    -> fp32-accumulated combine onto the post-attention residual. No shared
    expert. E=64, K=8 at 7B.

Numeric conventions MATCH the engine (curves track to within bf16
kernel-order noise, not a divergent fp32 model): weights/activations bf16;
RMSNorm, RoPE, the router softmax and the CE loss reduce in fp32 then cast
back; RMS eps 1e-5; SwiGLU rounds ``silu(gate)`` to bf16 before the product;
the MoE combine accumulates routed contributions in fp32 and rounds
``(residual + routed)`` to bf16 exactly once. Untied LM head.

Aux load-balance loss (OPTIONAL): ``loss()`` returns mean CE; pass
``aux_coef>0`` to add the MoE load-balancing auxiliary loss
``α·E·Σ_e f_e·p̄_e`` (α=``aux_coef``), summed over MoE layers — the standard
Switch/GShard term (matches the engine). ``f_e`` is the fraction of the top-K
assignments routed to expert e (discrete) and ``p̄_e`` the mean full-E router
probability for e (gradient flows through ``p̄``). The default ``aux_coef=0``
leaves the CE path bit-identical.

Weight ORIENTATION (for the parity bridge): the dense projections and the
router are ``nn.Linear`` (weight ``(out, in)``), so ``linear(x) == x @ packed``
when the bridge loads ``linear.weight = packed.T`` (the engine stores
``(in, out)``). The stacked expert weights ``w13_experts`` / ``w2_experts``
are raw parameters already in the engine's packed ``(E, ...)`` orientation and
load directly (``h2 @ w13_experts[e]``), as do the ``(vocab, d)`` embedding
and LM-head tables; the 1-D RMSNorm/qk-norm gains load directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5


@dataclass(frozen=True)
class OlmoeConfig:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int          # decoupled: n_heads*head_dim need not equal d_model
    n_experts: int         # E (router width, softmax space)
    top_k: int             # K experts routed per token
    d_ff_expert: int       # F: per-expert SwiGLU hidden width
    vocab_size: int
    rope_base: float = 10_000.0

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


class RMSNorm(nn.Module):
    """RMSNorm over the last dim: reduce in fp32, cast back, scale by gain.
    Used for the two block norms, the model's final norm, AND the FULL-ROW
    qk-norm (with ``dim == q_dim`` / ``kv_dim``, one rstd per token)."""

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
    (every ``(B, T)`` row is an independent length-``T`` causal sequence)."""
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
    """qwen3-shaped attention with FULL-ROW qk-norm applied BEFORE rope: one
    RMSNorm over the whole ``(T, q_dim)`` / ``(T, kv_dim)`` projection row
    (one rstd per token, ``(q_dim,)`` / ``(kv_dim,)`` gain) — then split into
    heads, rope, causal SDPA. No GQA at 7B (kept general via the kv repeat)."""

    def __init__(self, cfg: OlmoeConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.q_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.q_norm = RMSNorm(cfg.q_dim)    # FULL-ROW gain (whole q row), not per-head
        self.k_norm = RMSNorm(cfg.kv_dim)   # FULL-ROW gain (whole k row)
        self.wo = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)

    def forward(self, h1, cos, sin):
        B, T, _ = h1.shape
        H, KV, hd = self.n_heads, self.n_kv_heads, self.head_dim
        # full-row qk-norm over the ENTIRE projection row, then reshape to heads
        qn = self.q_norm(self.wq(h1))                     # (B, T, q_dim)
        kn = self.k_norm(self.wk(h1))                     # (B, T, kv_dim)
        q = apply_rope(qn.view(B, T, H, hd), cos, sin)
        k = apply_rope(kn.view(B, T, KV, hd), cos, sin)
        v = self.wv(h1).view(B, T, KV, hd)
        rep = H // KV
        q = q.transpose(1, 2)                             # (B, H, T, hd)
        k = k.repeat_interleave(rep, dim=2).transpose(1, 2)   # rep == 1 at 7B (no GQA)
        v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(o)


class MoE(nn.Module):
    """Routed-SwiGLU MoE FFN (dropless): router -> softmax_then_topk selection
    -> per-expert SwiGLU -> fp32-accumulated combine onto the residual. No
    shared expert. Expert weights are packed: ``w13_experts (E, d, 2F)`` holds
    [gate | up] and ``w2_experts (E, F, d)`` the down projection. Each forward
    also stashes ``self.aux_lbl`` — this layer's load-balancing term
    ``E·Σ_e f_e·p̄_e`` — for ``Olmoe.load_balance_loss`` (optional; it does NOT
    enter the routed output)."""

    def __init__(self, cfg: OlmoeConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.d_ff_expert = cfg.d_ff_expert
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.w13_experts = nn.Parameter(
            torch.empty(cfg.n_experts, cfg.d_model, 2 * cfg.d_ff_expert))
        self.w2_experts = nn.Parameter(
            torch.empty(cfg.n_experts, cfg.d_ff_expert, cfg.d_model))
        # fan-in init so a from-scratch (non-bridged) model is well-conditioned;
        # the parity bridge overwrites both stacks with the engine's init.
        nn.init.normal_(self.w13_experts, std=cfg.d_model ** -0.5)
        nn.init.normal_(self.w2_experts, std=cfg.d_ff_expert ** -0.5)
        # this layer's load-balancing term, populated each forward (None until
        # the first forward). Read via Olmoe.load_balance_loss().
        self.aux_lbl: torch.Tensor | None = None

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        B, T, d = h2.shape
        f = self.d_ff_expert
        # router: bf16 GEMM logits, then ALL routing math in fp32
        logits = self.router(h2)                          # (B, T, E) bf16
        probs = torch.softmax(logits.float(), dim=-1)     # full-E softmax, fp32
        # top-K by stable descending sort => smallest-index tie-break (torch.topk
        # tie-breaks differently); weights are the raw probs, kept UNNORMALIZED.
        vals, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
        weights = vals[..., :self.top_k]                  # (B, T, K) fp32, differentiable
        ids = idx[..., :self.top_k]                       # (B, T, K) discrete selection
        # load-balancing aux term for this layer (Switch/GShard), fp32 scalar, no α:
        #   L = E * Σ_e f_e·p̄_e ; f_e = frac of top-K assignments on e (discrete,
        #   via bincount => detached), p̄_e = mean full-E router prob for e (grad
        #   flows through p̄). Stashed only — does NOT touch the routed output below.
        n_tok = B * T
        counts = torch.bincount(ids.reshape(-1),
                                minlength=self.n_experts).float()
        frac = counts / (n_tok * self.top_k)              # (E,) f_e, detached fraction
        p_bar = probs.reshape(n_tok, self.n_experts).mean(0)   # (E,) grad-carrying
        self.aux_lbl = self.n_experts * (frac * p_bar).sum()   # fp32 scalar
        # dropless masked E-loop (autograd-able; correctness over speed)
        routed = torch.zeros(B, T, d, dtype=torch.float32, device=h2.device)
        for e in range(self.n_experts):
            coef = (weights * (ids == e)).sum(-1)         # (B, T) fp32; <= 1 hit per token
            h13 = h2 @ self.w13_experts[e]                # (B, T, 2F) bf16
            gate, up = h13[..., :f], h13[..., f:]
            act = F.silu(gate.float()).to(gate.dtype) * up            # bf16
            routed = routed + coef[..., None] * (act @ self.w2_experts[e]).float()
        # combine convention: fp32 routed accumulator + bf16 residual, round once
        return (resid.float() + routed).to(h2.dtype)


class Block(nn.Module):
    def __init__(self, cfg: OlmoeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        h_mid = x + self.attn(self.attn_norm(x), cos, sin)
        # the MoE tail folds the residual into its fp32 combine (returns h_mid + routed)
        return self.moe(self.ffn_norm(h_mid), h_mid)


class Olmoe(nn.Module):
    """Untied-embedding OLMoE. ``forward`` takes ``(B, T)`` int tokens where
    each row is an independent causal sequence (uniform packing)."""

    def __init__(self, cfg: OlmoeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # recompute each block in the backward (activation checkpointing) to
        # trade compute for memory — only needed at the largest scale; off by default.
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

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the per-layer load-balancing term
        ``E·Σ_e f_e·p̄_e`` stashed by the most recent forward (no α — scale by
        ``aux_coef`` at the call site). ``f_e`` is the discrete fraction of
        top-K assignments on expert e; ``p̄_e`` the mean full-E router prob
        (gradient flows through ``p̄``). Returns a ``0.0`` scalar if no forward
        has populated the per-layer terms yet."""
        terms = [blk.moe.aux_lbl for blk in self.blocks
                 if blk.moe.aux_lbl is not None]
        if not terms:
            return torch.zeros((), device=self.lm_head.weight.device)
        return torch.stack(terms).sum()

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0) -> torch.Tensor:
        """Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization. ``tokens``/``targets`` are ``(B, T)``
        int; targets are the next-token ids.

        ``loss()`` returns mean CE; pass ``aux_coef>0`` to add the MoE
        load-balancing auxiliary loss ``α·E·Σ_e f_e·p̄_e`` (α=``aux_coef``),
        summed over MoE layers — the standard Switch/GShard term (matches the
        engine). The default ``aux_coef=0`` returns CE alone, bit-identical to
        the CE-only path."""
        logits = self.forward(tokens)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
        if aux_coef > 0:
            return ce + aux_coef * self.load_balance_loss()
        return ce
