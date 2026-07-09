"""Independent DeepSeek-V3.2 (DSV3.2) — a plain PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the DeepSeek-V3.2 arm of the pretraining
parity study, reimplemented from scratch. Like the sibling references it is
deliberately isolated: it imports ONLY ``torch`` (nothing from ``dataflow``,
nothing from the other ``reference_models/`` files — RMSNorm / RoPE / SwiGLU AND
the whole MLA + DSA indexer + MoE are reimplemented locally). It reads like a
normal transformer and lets autograd derive the backward pass; a second
from-scratch implementation guards against a shared bug in the engine's
hand-written reference ops.

DSV3.2 is the DeepSeek-V3 backbone (MLA attention + sigmoid-noaux top-k MoE,
mixed dense/MoE depth) with **DeepSeek Sparse Attention (DSA)** in EVERY
layer's attention:

  - **MLA (Multi-head Latent Attention)** — q via a low-rank stack
    (``d -> q_lora_rank``, RMSNorm, ``-> n_heads*(nope+rope)``); kv via
    ``d -> (kv_lora_rank + rope)`` where the LAST ``rope`` columns are the ONE
    decoupled key-rope vector shared by every head; latent RMSNorm
    ``-> n_heads*(nope+v)``. Rotate-half RoPE on the ``rope`` dims only
    (nope-FIRST head layout). Per head ``qk = nope+rope``; v is zero-padded to
    ``qk`` so one shared head_dim runs SDPA (softmax scale ``qk**-0.5``) and
    the output is sliced back to ``v``.
  - **DSA lightning indexer + top-k selection** — a cheap indexer scores every
    (query ``t``, key ``s<=t``) pair:
        ``I[t,s] = sum_h w_h(t) * ReLU(q_idx[t,h] . k_idx[s])``
    with per-head weights ``w = (h1 @ w_idx_w) * H_I**-0.5 * d_I**-0.5``.
    ``q_idx`` taps the SHARED post-norm ``q_lora`` latent (``-> H_I`` heads of
    ``d_I``); ``k_idx`` is ONE shared key per token
    ``= rope(LayerNorm(h1 @ w_idx_k))``. rope-FIRST head layout (the first
    ``rope`` dims), the OPPOSITE of MLA's nope-first. The causal mask is added
    to ``I`` BEFORE selection; per query the top ``min(index_topk, seq_len)``
    keys are kept. The attention mask = scatter(selected) + causal, which
    re-suppresses the future-padded slots that a short prefix necessarily
    selects. MLA attention then runs restricted to that (causal) live set.
    When ``seq_len <= index_topk`` the selection keeps ALL causal keys — a
    dense causal prefix (the correct degenerate).
  - **FFN** — the first ``first_k_dense`` layers use a dense SwiGLU MLP; the
    rest use the MoE: ``sigmoid_noaux_tc`` routing (sigmoid scores; group-
    limited selection on ``score + bias``; weights = the selected RAW sigmoid
    scores renormalized to sum 1, times ``routed_scaling``) with an UNGATED
    additive shared expert (DeepSeek-V3 style).

SCOPE — this reference implements the SPARSE path ONLY. ``loss()`` returns
mean CE; pass ``aux_coef>0`` to add the routed-expert load-balancing auxiliary
loss α·E·Σ_e f_e·p̄_e (α=aux_coef, p̄ = mean normalized-sigmoid router prob),
summed over MoE layers — matches the engine's balance loss; the shared expert
is excluded. (The DSA indexer-KL objective remains omitted.) Still deliberately
OMITTED: the dense-warm-up mode; the indexer's KL training objective (so the
indexer weights receive no gradient here and stay at their init — they still
drive selection in the forward); and the noaux balance-bias step rule
(``w_router_bias`` is a fixed zero buffer, so selection is by raw sigmoid
score).

Numeric conventions MATCH the engine (so curves track to within bf16
kernel-order noise, not a divergent fp32 model): weights/activations bf16;
RMSNorm, the indexer LayerNorm, RoPE, softmax, the routing math and the CE
loss reduce in fp32 then cast back; the MoE combine accumulates routed
contributions in fp32 and rounds ``(residual + routed)`` once. RMS eps 1e-5,
indexer LayerNorm eps 1e-5, RoPE base 1e4, untied LM head. The lightning
indexer's per-head weights ``w_idx_w`` and the balance bias ``w_router_bias``
are fp32 (engine dtype policy); keep them fp32 (do not downcast) in a bf16 run.

Weight ORIENTATION (for the parity bridge): MLA / indexer / dense-MLP / head
projections are ``nn.Linear`` (weight ``(out, in)``), so ``linear(x) ==
x @ packed`` when the bridge loads ``linear.weight = packed.T`` (the engine
stores ``(in, out)``). The MoE router / expert stacks / shared expert and
``w_idx_w`` are raw parameters already in the engine's ``x @ w`` orientation
(``w_router (d, E)``, ``w13_experts (E, d, 2F)`` [gate|up], ``w2_experts
(E, F, d)``, ``w_s13 (d, 2Fs)``, ``w_s2 (Fs, d)``, ``w_idx_w (d, H_I)``) and
load directly, as do the ``(vocab, d)`` embedding / LM-head tables and the
1-D RMSNorm / LayerNorm gains.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5
LN_EPS = 1e-5   # indexer key LayerNorm


@dataclass(frozen=True)
class Dsv32Config:
    n_layers: int
    d_model: int
    # --- MLA ---
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_head_dim: int
    # --- FFN: first_k_dense dense-SwiGLU layers, the rest MoE ---
    d_ff_dense: int
    first_k_dense: int
    n_experts: int
    top_k: int
    d_ff_expert: int
    n_group: int
    topk_group: int
    n_shared_experts: int
    d_ff_shared: int
    # --- DSA lightning indexer ---
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # --- shared ---
    vocab_size: int
    routed_scaling: float = 2.5
    rope_base: float = 10_000.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.qk_head_dim

    @property
    def v_dim(self) -> int:
        return self.n_heads * self.v_head_dim

    def kind_of(self, layer: int) -> str:
        return "dense" if layer < self.first_k_dense else "moe"


class RMSNorm(nn.Module):
    """RMSNorm over the last dim: reduce in fp32, cast back, scale by gain.
    Reused for the block norms, the model's final norm, and the two MLA
    mid-stack latent norms (widths ``q_lora_rank`` / ``kv_lora_rank``)."""

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


def rope_tables(seq_len: int, rope_dim: int, base: float, device,
                dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) each ``(seq_len, rope_dim)`` — positions reset per sequence
    (every ``(B, T)`` row is an independent length-``T`` causal sequence). One
    table (``rope_dim == qk_rope_dim``) serves MLA q, the shared MLA key-rope,
    and both indexer q/k rope slices (all rotate the same ``qk_rope_dim``)."""
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device,
                                       dtype=torch.float32) / rope_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the last dim. x: ``(B, T, H, rope_dim)``; cos/sin:
    ``(T, rope_dim)`` fp32. Math in fp32, cast back to x's dtype."""
    xf = x.float()
    out = xf * cos[None, :, None, :] + _rotate_half(xf) * sin[None, :, None, :]
    return out.to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, with silu ROUNDED to the storage dtype before the
    product (matches the engine's swiglu kernel)."""
    return F.silu(gate.float()).to(gate.dtype) * up


class DsaAttention(nn.Module):
    """MLA expansion + DeepSeek Sparse Attention.

    Takes the post-attn-norm activation ``h1`` and returns the attention
    output already projected by ``wo`` (the block adds the residual). The
    lightning indexer + top-k selection run under ``no_grad`` — selection is
    non-differentiable and, with the KL objective omitted, the indexer sees no
    gradient; the sparse core over the selected keys carries the CE gradient.
    """

    def __init__(self, cfg: Dsv32Config):
        super().__init__()
        self.cfg = cfg
        h, qk = cfg.n_heads, cfg.qk_head_dim
        # --- MLA projections (nn.Linear: bridge loads weight = packed.T) ---
        self.w_q_a = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank)
        self.w_q_b = nn.Linear(cfg.q_lora_rank, h * qk, bias=False)
        self.w_kv_a = nn.Linear(cfg.d_model, cfg.kv_lora_rank + cfg.qk_rope_dim,
                                bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_kv_b = nn.Linear(cfg.kv_lora_rank,
                                h * (cfg.qk_nope_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(h * cfg.v_head_dim, cfg.d_model, bias=False)
        # --- DSA lightning indexer ---
        hi, di = cfg.index_n_heads, cfg.index_head_dim
        self.w_idx_q = nn.Linear(cfg.q_lora_rank, hi * di, bias=False)
        self.w_idx_k = nn.Linear(cfg.d_model, di, bias=False)
        self.idx_k_ln_w = nn.Parameter(torch.ones(di))          # key LayerNorm gain
        self.idx_k_ln_b = nn.Parameter(torch.zeros(di))         # key LayerNorm bias
        # per-head score weights: fp32, engine orientation (h1 @ w_idx_w)
        self.w_idx_w = nn.Parameter(torch.empty(cfg.d_model, hi, dtype=torch.float32))
        nn.init.normal_(self.w_idx_w, std=cfg.d_model ** -0.5)

    def _mla_qkv(self, h1, cos, sin):
        """MLA q/kv expansion -> (q_lora, q_full, k_full, v_pad); the last
        three are ``(B, T, h, qk)`` with v zero-padded to qk width."""
        c = self.cfg
        B, T, _ = h1.shape
        h, nope, rope, v = c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim
        qk = nope + rope
        # q stack
        q_lora = self.q_a_norm(self.w_q_a(h1))                  # (B,T,q_lora_rank)
        q = self.w_q_b(q_lora).view(B, T, h, qk)
        q_full = torch.cat([q[..., :nope], apply_rope(q[..., nope:], cos, sin)],
                           dim=-1)                              # nope-first
        # kv stack: latent + ONE shared decoupled key-rope
        kv = self.w_kv_a(h1)                                    # (B,T,kv_lora+rope)
        latent = self.kv_a_norm(kv[..., :c.kv_lora_rank])
        k_rope = apply_rope(kv[..., c.kv_lora_rank:].unsqueeze(2), cos, sin)  # (B,T,1,rope)
        kvb = self.w_kv_b(latent).view(B, T, h, nope + v)
        k_full = torch.cat([kvb[..., :nope], k_rope.expand(B, T, h, rope)], dim=-1)
        v_pad = torch.cat([kvb[..., nope:], kvb.new_zeros(B, T, h, qk - v)], dim=-1)
        return q_lora, q_full, k_full, v_pad

    def _index_scores(self, h1, q_lora, cos, sin, causal):
        """(B, T, T) fp32 lightning-indexer scores, causal-masked. rope-FIRST
        head layout; ``k_idx`` is one shared key per token."""
        c = self.cfg
        B, T, _ = h1.shape
        hi, di, rope = c.index_n_heads, c.index_head_dim, c.qk_rope_dim
        q = self.w_idx_q(q_lora).view(B, T, hi, di)
        q = torch.cat([apply_rope(q[..., :rope], cos, sin), q[..., rope:]],
                      dim=-1).float()                          # (B,T,hi,di)
        k = F.layer_norm(self.w_idx_k(h1).float(), (di,),
                         self.idx_k_ln_w.float(), self.idx_k_ln_b.float(), LN_EPS)
        k = torch.cat([apply_rope(k[..., :rope].unsqueeze(2), cos, sin).squeeze(2),
                       k[..., rope:]], dim=-1)                  # (B,T,di) fp32
        wts = (h1.float() @ self.w_idx_w.float()) * (hi ** -0.5) * (di ** -0.5)
        r = torch.einsum("bthd,bsd->bths", q, k)               # (B,T,hi,T)
        scores = (wts.unsqueeze(-1) * r.clamp_min(0.0)).sum(2)  # sum over heads
        return scores + causal

    def _select_mask(self, scores, causal):
        """Top-``min(index_topk, T)`` per query -> additive {0,-inf} mask.
        scatter(selected)=0 then + causal re-suppresses future-padded slots."""
        B, T, _ = scores.shape
        kk = min(self.cfg.index_topk, T)
        idx = torch.topk(scores, kk, dim=-1, sorted=True).indices  # (B,T,kk)
        m = torch.full((B, T, T), float("-inf"), device=scores.device)
        m.scatter_(-1, idx, 0.0)
        return m + causal

    def _sparse_core(self, q_full, k_full, v_pad, mask):
        """Masked SDPA over the padded-v MLA tensors; causality is in the mask
        (no is_causal). Output sliced back to the true v width."""
        B, T, h, qk = q_full.shape
        v = self.cfg.v_head_dim
        q4 = q_full.transpose(1, 2)                            # (B,h,T,qk)
        k4 = k_full.transpose(1, 2)
        v4 = v_pad.transpose(1, 2)
        m = mask.unsqueeze(1).to(q4.dtype)                     # (B,1,T,T)
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m)  # scale qk**-0.5
        o = o.transpose(1, 2)                                  # (B,T,h,qk)
        return o[..., :v].reshape(B, T, h * v)

    def forward(self, h1, cos, sin, causal):
        q_lora, q_full, k_full, v_pad = self._mla_qkv(h1, cos, sin)
        with torch.no_grad():                                 # indexer: selection only
            scores = self._index_scores(h1, q_lora, cos, sin, causal)
            mask = self._select_mask(scores, causal)
        attn = self._sparse_core(q_full, k_full, v_pad, mask)
        return self.wo(attn)


class DenseMLP(nn.Module):
    """Dense SwiGLU MLP for the first ``first_k_dense`` layers."""

    def __init__(self, cfg: Dsv32Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff_dense, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff_dense, bias=False)   # up
        self.w2 = nn.Linear(cfg.d_ff_dense, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class MoE(nn.Module):
    """DeepSeek-V3 sigmoid-noaux top-k MoE + ungated additive shared expert.

    Router logits are a storage-dtype GEMM; ALL routing math is fp32. Selection
    (``sigmoid(logit) + bias``) is group-limited (``n_group`` score groups
    ranked by the sum of each group's top-2 selection scores; only the best
    ``topk_group`` groups stay eligible) then greedy top-K; the routing weights
    are the selected RAW sigmoid scores renormalized to sum 1, times
    ``routed_scaling``. Combine: routed contributions accumulate in fp32, the
    (residual + shared) base adds in storage dtype, the sum rounds once."""

    def __init__(self, cfg: Dsv32Config):
        super().__init__()
        self.cfg = cfg
        # most-recent-forward load-balancing aux (fp32 scalar); see forward.
        self.aux_lbl: torch.Tensor | None = None
        d, E, f = cfg.d_model, cfg.n_experts, cfg.d_ff_expert
        # engine orientation: out = x @ w; experts packed [gate | up] over 2F.
        self.w_router = nn.Parameter(torch.empty(d, E))
        # noaux balance bias: fp32, selection-only, NON-gradient. The per-step
        # sign-update rule is omitted, so it stays zero (the optional LBL adds
        # gradient to the router weights, not to this selection bias).
        self.register_buffer("w_router_bias", torch.zeros(E, dtype=torch.float32))
        self.w13_experts = nn.Parameter(torch.empty(E, d, 2 * f))
        self.w2_experts = nn.Parameter(torch.empty(E, f, d))
        nn.init.normal_(self.w_router, std=d ** -0.5)
        nn.init.normal_(self.w13_experts, std=d ** -0.5)
        nn.init.normal_(self.w2_experts, std=f ** -0.5)
        if cfg.n_shared_experts:
            fs = cfg.d_ff_shared
            self.w_s13 = nn.Parameter(torch.empty(d, 2 * fs))
            self.w_s2 = nn.Parameter(torch.empty(fs, d))
            nn.init.normal_(self.w_s13, std=d ** -0.5)
            nn.init.normal_(self.w_s2, std=fs ** -0.5)

    def _route(self, logits):
        """(weights fp32 (N,K), ids int64 (N,K)) for sigmoid_noaux_tc. Weights
        stay differentiable via the selected raw sigmoid scores; ids discrete.
        Smallest-index tie-break via stable descending sort (expert AND group)."""
        c = self.cfg
        scores = torch.sigmoid(logits.float())                # (N, E)
        N, E = scores.shape
        ng, kg = c.n_group, c.topk_group
        with torch.no_grad():
            sel = scores + self.w_router_bias                 # selection scores
            g = sel.view(N, ng, E // ng)
            g_sorted, _ = torch.sort(g, dim=-1, descending=True, stable=True)
            gscore = g_sorted[..., :min(2, E // ng)].sum(-1)  # (N, ng)
            _, gidx = torch.sort(gscore, dim=-1, descending=True, stable=True)
            gmask = torch.zeros(N, ng, dtype=torch.bool, device=logits.device)
            gmask.scatter_(1, gidx[:, :kg], True)
            emask = gmask.repeat_interleave(E // ng, dim=1)   # (N, E)
            masked = sel.masked_fill(~emask, float("-inf"))
            ids = torch.sort(masked, dim=-1, descending=True,
                             stable=True).indices[:, :c.top_k]
        picked = scores.gather(1, ids)                        # raw sigmoid scores
        w = picked / picked.sum(-1, keepdim=True) * c.routed_scaling
        return w, ids

    def _load_balance_loss(self, logits: torch.Tensor,
                           ids: torch.Tensor) -> torch.Tensor:
        """Routed-expert load-balancing aux (DeepSeek sigmoid_noaux_tc), no α:
        ``E · Σ_e f_e·p̄_e`` (fp32 scalar). ``f_e = count_e/(T·K)`` from the
        DISCRETE top-K routed ids (bincount over the flattened batch; detached);
        ``p̄_e`` = mean over tokens of the FULL-E normalized-sigmoid router prob
        ``p = s / s.sum(-1)`` — gradient flows through ``p̄``. Shared expert
        EXCLUDED; ``Σ_e f_e = Σ_e p̄_e = 1`` so uniform routing ⇒ L = 1."""
        E, K = self.cfg.n_experts, self.cfg.top_k
        s = torch.sigmoid(logits.float())                     # (N, E)
        pbar = (s / s.sum(-1, keepdim=True)).mean(0)          # (E,) grad via p̄
        with torch.no_grad():                                 # f from discrete ids
            counts = torch.bincount(ids.reshape(-1), minlength=E).to(pbar.dtype)
            f = counts / (ids.shape[0] * K)                   # (E,) detached, Σf=1
        return E * (f * pbar).sum()

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        B, T, d = h2.shape
        f = c.d_ff_expert
        x = h2.reshape(B * T, d)
        logits = x @ self.w_router                            # (N, E) storage-dtype
        w, ids = self._route(logits)
        self.aux_lbl = self._load_balance_loss(logits, ids)   # fp32 scalar, no α
        routed = torch.zeros(B * T, d, dtype=torch.float32, device=h2.device)
        for e in range(c.n_experts):                          # dropless masked E-loop
            coef = (w * (ids == e)).sum(-1)                   # (N,) fp32; <=1 hit/token
            h13 = x @ self.w13_experts[e]                     # (N, 2F)
            act = swiglu(h13[:, :f], h13[:, f:])
            routed = routed + coef[:, None] * (act @ self.w2_experts[e]).float()
        base = resid.reshape(B * T, d)
        if c.n_shared_experts:                                # ungated additive (V3)
            s13 = x @ self.w_s13
            s_act = swiglu(s13[:, :c.d_ff_shared], s13[:, c.d_ff_shared:])
            base = base + s_act @ self.w_s2
        y = (base.float() + routed).to(h2.dtype)
        return y.reshape(B, T, d)


class Block(nn.Module):
    def __init__(self, cfg: Dsv32Config, layer: int):
        super().__init__()
        self.kind = cfg.kind_of(layer)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = DsaAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.mlp = DenseMLP(cfg) if self.kind == "dense" else MoE(cfg)

    def forward(self, x, cos, sin, causal):
        h_mid = x + self.attn(self.attn_norm(x), cos, sin, causal)
        h2 = self.ffn_norm(h_mid)
        if self.kind == "dense":
            return h_mid + self.mlp(h2)
        # the MoE tail folds the residual into its fp32 combine
        return self.mlp(h2, h_mid)


class Dsv32(nn.Module):
    """Untied-embedding DeepSeek-V3.2. ``forward`` takes ``(B, T)`` int tokens
    where each row is an independent causal sequence (uniform packing)."""

    def __init__(self, cfg: Dsv32Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # recompute each block in the backward (activation checkpointing) to
        # trade compute for memory — only needed at the largest scale; off by default.
        self.grad_checkpoint = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.embed(tokens)
        cos, sin = rope_tables(T, self.cfg.qk_rope_dim, self.cfg.rope_base, x.device)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, causal, use_reentrant=False)
            else:
                x = blk(x, cos, sin, causal)
        return self.lm_head(self.final_norm(x))

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the routed-expert load-balancing aux
        (``MoE.aux_lbl``, stashed by the most recent forward); ``0.0`` if the
        model has no MoE layers or has not run a forward yet. Differentiable
        through each layer's ``p̄`` back to that layer's router weights."""
        total = torch.zeros((), dtype=torch.float32,
                            device=self.lm_head.weight.device)
        for blk in self.blocks:
            aux = getattr(blk.mlp, "aux_lbl", None)           # None for dense MLP
            if aux is not None:
                total = total + aux
        return total

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0) -> torch.Tensor:
        """``loss()`` returns mean CE; pass ``aux_coef>0`` to add the routed-
        expert load-balancing auxiliary loss α·E·Σ_e f_e·p̄_e (α=aux_coef,
        p̄ = mean normalized-sigmoid router prob), summed over MoE layers —
        matches the engine's balance loss; the shared expert is excluded. (The
        DSA indexer-KL objective remains omitted.)

        Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization. ``tokens``/``targets`` are ``(B, T)``
        int; targets are the next-token ids. With the default ``aux_coef=0`` the
        returned value (and its autograd graph) is exactly the pure-CE result."""
        logits = self.forward(tokens)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
        if aux_coef > 0:
            return ce + aux_coef * self.load_balance_loss()
        return ce
