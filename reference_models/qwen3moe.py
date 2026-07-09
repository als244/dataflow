"""Independent Qwen3-MoE — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the Qwen3-MoE family (Qwen3-30B-A3B /
235B-A22B), reimplemented from scratch. It is deliberately isolated: it
imports ONLY ``torch`` (nothing from ``dataflow``, nothing from the other
``reference_models/`` files — RMSNorm/RoPE/SwiGLU and the whole MoE router +
experts + combine are reimplemented locally), reads like a normal
transformer, and lets autograd derive the backward pass. The engine must
reproduce this model's loss curve from a byte-identical init on the same
data stream; a second from-scratch implementation also guards against a
shared bug in the engine's hand-written reference ops.

Architecture — qwen3's attention VERBATIM, only the FFN is MoE:

  - **attention** — RMSNorm -> wq/wk/wv -> PER-HEAD qk-norm (an RMSNorm over
    each head_dim-wide row, one shared ``(head_dim,)`` gain for all q heads
    and one for all k heads, applied BEFORE rope) -> full RoPE (base 1e6) ->
    GQA causal attention -> output proj. No biases.
  - **MoE FFN** — RMSNorm -> router logits ``h2 @ w_router`` -> routing mode
    ``topk_then_softmax`` (select the top-K experts by router logit, then
    softmax over JUST those K logits — weights sum to 1, i.e. renormalized
    over K / norm_topk_prob=true) -> dropless per-expert SwiGLU (packed
    ``w13_experts (E,d,2F)`` = [gate|up], ``w2_experts (E,F,d)``) ->
    weighted combine folded onto the post-attention residual. NO shared
    expert; all layers sparse. Untied LM head.

The MoE also supports an OPTIONAL load-balancing auxiliary loss (LBL):
``loss()`` returns mean CE; pass ``aux_coef>0`` to add the MoE
load-balancing auxiliary loss α·E·Σ_e f_e·p̄_e (α=aux_coef), summed over MoE
layers — the standard Switch/GShard term (matches the engine). The default
``aux_coef=0`` leaves the reported CE and its gradient exactly as a
pure-CE model's.

Numeric conventions MATCH the engine (so curves track within bf16
kernel-order noise, not a textbook fp32 model): weights/activations bf16;
RMSNorm, RoPE, the router softmax/top-k math and the CE loss reduce in fp32
then cast back; RMS eps 1e-5; RoPE is llama rotate-half with base 1e6. The
router GEMM runs in storage dtype and ALL routing math is fp32 from those
logits; SwiGLU rounds silu to storage dtype before the product; the routed
contributions accumulate in fp32 and the (residual + routed) sum rounds
once at the end.

Weight ORIENTATION (for the parity bridge): attention/head projections are
``nn.Linear`` (weight ``(out, in)``), so ``linear(x) == x @ packed.T`` when
the bridge loads ``linear.weight = packed.T`` (the engine stores
``(in, out)``). The MoE ``w_router (d, E)`` and the packed expert stacks
``w13_experts (E, d, 2F)`` / ``w2_experts (E, F, d)`` are held in the
engine's own orientation and load directly. Embedding / LM-head tables are
``(vocab, d)`` and load directly; RMSNorm gains are 1-D and load directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5


@dataclass(frozen=True)
class Qwen3MoeConfig:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int          # decoupled from d_model // n_heads (qwen3 convention)
    d_ff_expert: int       # per-expert SwiGLU hidden width (F)
    n_experts: int         # global expert count (E)
    top_k: int             # experts routed per token (K)
    vocab_size: int
    rope_base: float = 1_000_000.0

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


class RMSNorm(nn.Module):
    """RMSNorm over the last dim, fp32 statistics, gain in the given width.
    Reused for the block norms (width d_model) and, over head_dim-wide rows,
    for the per-head qk-norm (gain ``(head_dim,)``)."""

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


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, with silu ROUNDED to the storage dtype before the
    product (matches the engine's swiglu kernel)."""
    return F.silu(gate.float()).to(gate.dtype) * up


class Attention(nn.Module):
    """qwen3 attention: per-head qk-norm before full RoPE, GQA, causal."""

    def __init__(self, cfg: Qwen3MoeConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.q_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)   # per-head qk-norm gains
        self.k_norm = RMSNorm(cfg.head_dim)
        self.wo = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, KV, hd = self.n_heads, self.n_kv_heads, self.head_dim
        # PER-HEAD qk-norm (RMSNorm over each head_dim row) THEN rope
        q = apply_rope(self.q_norm(self.wq(x).view(B, T, H, hd)), cos, sin)
        k = apply_rope(self.k_norm(self.wk(x).view(B, T, KV, hd)), cos, sin)
        v = self.wv(x).view(B, T, KV, hd)
        rep = H // KV
        q = q.transpose(1, 2)                                   # (B, H, T, hd)
        k = k.repeat_interleave(rep, dim=2).transpose(1, 2)
        v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(o)


class MoE(nn.Module):
    """Routed SwiGLU MoE FFN — router -> top-K softmax weights -> per-expert
    SwiGLU -> weighted combine onto the residual. Dropless (every routed
    assignment is computed); no shared expert. The masked per-expert loop
    mirrors the engine's reference (each token contributes to expert ``e``
    with its routing weight when selected, else exactly zero).

    Each forward also stashes ``self.aux_lbl`` — this layer's load-balancing
    aux term ``E·Σ_e f_e·p̄_e`` (no coefficient) — for the model's OPTIONAL
    ``aux_coef>0`` objective; it does not affect the returned activation."""

    def __init__(self, cfg: Qwen3MoeConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.d_ff_expert = cfg.d_ff_expert
        d, e, f = cfg.d_model, cfg.n_experts, cfg.d_ff_expert
        # engine orientation: out = x @ w; experts packed [gate | up] over 2F.
        self.w_router = nn.Parameter(torch.empty(d, e))
        self.w13_experts = nn.Parameter(torch.empty(e, d, 2 * f))
        self.w2_experts = nn.Parameter(torch.empty(e, f, d))
        nn.init.normal_(self.w_router, std=d ** -0.5)
        nn.init.normal_(self.w13_experts, std=d ** -0.5)
        nn.init.normal_(self.w2_experts, std=f ** -0.5)

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        """h2: post-ffn-norm ``(B, T, d)``; resid: post-attention residual."""
        f = self.d_ff_expert
        logits = h2 @ self.w_router                      # (B, T, E) storage-dtype GEMM
        lf = logits.float()                              # fp32 routing math
        # top-K by logit; smallest-index tie-break via stable descending sort
        vals, idx = torch.sort(lf, dim=-1, descending=True, stable=True)
        ids = idx[..., :self.top_k]                      # (B, T, K)
        route_w = torch.softmax(vals[..., :self.top_k], dim=-1)   # (B,T,K) fp32, sums to 1
        # OPTIONAL load-balancing aux loss (LBL), stashed per layer, no α:
        #   L_layer = E · Σ_e f_e·p̄_e   (the Switch/GShard term, matches engine)
        # f_e = count_e/(T·K) from the DISCRETE top-K ids (bincount; detached by
        # construction); p̄_e = mean over tokens of the FULL-E softmax(logits) —
        # full E even though routing is topk_then_softmax. Grad flows through p̄
        # only; pure side effect — the returned activation is unchanged.
        E = self.n_experts
        counts = torch.bincount(ids.reshape(-1).long(), minlength=E)   # (E,)
        f_e = counts.float() / ids.numel()                            # count_e/(T·K)
        pbar = torch.softmax(lf, dim=-1).reshape(-1, E).mean(0)        # (E,) full-E
        self.aux_lbl = E * (f_e * pbar).sum()                         # fp32 scalar
        routed = torch.zeros_like(h2, dtype=torch.float32)
        for e in range(self.n_experts):
            coef = (route_w * (ids == e)).sum(-1)        # (B, T) fp32; <=1 hit/token
            h13 = h2 @ self.w13_experts[e]               # (B, T, 2F)
            act = swiglu(h13[..., :f], h13[..., f:])
            routed = routed + coef[..., None] * (act @ self.w2_experts[e]).float()
        # residual + routed in fp32, rounded once (engine combine convention)
        return (resid.float() + routed).to(h2.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Qwen3MoeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        h_mid = x + self.attn(self.attn_norm(x), cos, sin)
        # the second residual is folded into the MoE combine (h_mid + routed)
        return self.moe(self.ffn_norm(h_mid), h_mid)


class Qwen3Moe(nn.Module):
    """Untied-embedding Qwen3-MoE. ``forward`` takes ``(B, T)`` int tokens
    where each row is an independent causal sequence (uniform packing)."""

    def __init__(self, cfg: Qwen3MoeConfig):
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

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the load-balancing aux term ``E·Σ_e f_e·p̄_e``
        (no coefficient) stashed by each MoE forward; ``0.0`` when the model has
        no MoE layers. Call after ``forward``/``loss``; scale by α and add to the
        CE (``loss`` does this when ``aux_coef>0``)."""
        total = torch.zeros((), dtype=torch.float32,
                            device=self.lm_head.weight.device)
        for blk in self.blocks:
            aux = getattr(blk.moe, "aux_lbl", None)
            if aux is not None:
                total = total + aux
        return total

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0) -> torch.Tensor:
        """loss() returns mean CE; pass ``aux_coef>0`` to add the MoE
        load-balancing auxiliary loss α·E·Σ_e f_e·p̄_e (α=aux_coef), summed over
        MoE layers — the standard Switch/GShard term (matches the engine).

        Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization. ``tokens``/``targets`` are ``(B, T)``
        int; targets are the next-token ids. The default ``aux_coef=0`` returns
        pure CE, bit-identical to the model without the aux term."""
        logits = self.forward(tokens)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
        if aux_coef > 0:
            return ce + aux_coef * self.load_balance_loss()
        return ce
