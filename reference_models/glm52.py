"""Independent GLM-5.2 — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the GLM-5.2 family, reimplemented from
scratch: it imports ONLY ``torch`` (nothing from ``dataflow``, nothing from
the other ``reference_models/`` files), reads like a normal transformer, and lets
autograd derive the backward pass.

GLM-5.2 is a DSA variant of DeepSeek-V3. Relative to plain DeepSeek-V3 (MLA +
mixed dense/MoE depth) it adds two things, and this file mirrors both:

  - **DSA (DeepSeek Sparse Attention).** Before each attention, a *lightning
    indexer* scores every causal (query, key) pair:
    ``I[t,s] = sum_h w~[t,h] * ReLU(q^I[t,h] . k^I[s])``  (report eq. 1),
    with ``q^I`` tapped from the SHARED post-norm q-latent (rope-FIRST head
    layout, the opposite of main MLA's nope-first) and ``k^I`` one shared
    LayerNorm'd key per token. Each query keeps its top-``min(index_topk,
    prefix)`` keys; the main MLA attention then runs as masked SDPA over only
    that selected set. A sequence shorter than ``index_topk`` selects ALL its
    causal keys — a dense causal prefix, the correct degenerate.

  - **IndexShare (cross-layer index reuse, arXiv 2603.12201).** Layers carry a
    role ``full`` | ``shared`` (``indexer_types``, greedy-searched upstream, so
    an explicit list). A ``full`` layer (LEADER) runs its own indexer and emits
    its selection; the trailing run of ``shared`` layers (FOLLOWERS) carry NO
    indexer weights and REUSE the nearest preceding full layer's selection mask
    (with their own q/k/v). Layer 0 is always full. Kinds: ``gdl`` (dense FFN +
    indexer), ``gml`` (MoE + indexer), ``gmf`` (MoE, shared / no indexer). The
    first ``first_k_dense`` layers use a dense SwiGLU FFN; the rest are MoE.

MoE FFN: DeepSeek-V3 ``sigmoid_noaux_tc`` routing (sigmoid scores, group-
limited top-k selection on score+bias, weights = selected raw sigmoid scores
renormalized x routed_scaling) + one UNGATED additive shared expert.

SCOPE — ``loss`` returns mean CE; pass ``aux_coef>0`` to add the routed-expert
load-balancing auxiliary loss ``α·E·Σ_e f_e·p̄_e`` (α=``aux_coef``, ``p̄`` =
mean normalized-sigmoid router prob), summed over MoE layers — this matches the
engine's balance loss, and the shared expert is excluded. Two other
TRAINING-TIME injections that never enter the CE value stay deliberately
OMITTED (engine gradient/heuristic paths, not part of the parity study here):
(1) the indexer's KL distillation objective and its IndexShare multi-layer
target (``train_indexer`` / L^I_multi) — the indexer here only PRODUCES the
selection, it is not trained; (2) the noaux router-bias update rule
(``w_router_bias`` is a fixed zero buffer). The paper's dense warm-up mode is
also out of scope — only the sparse path exists.

Numeric conventions MATCH the engine (bf16-parity, not a textbook fp32 model):
weights/activations bf16 with fp32 reductions; RMSNorm eps 1e-5; the indexer,
all softmaxes, all norms and the CE loss reduce in fp32; RoPE is rotate-half.
Weight ORIENTATION follows llama3 (projections are ``nn.Linear``, weight
``(out, in)``); the stacked expert tensors are raw ``(E, in, out)`` params.
``(B, T)`` int tokens — each row an independent causal sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5
LN_EPS = 1e-5   # indexer key LayerNorm (repo-global norm eps)


@dataclass(frozen=True)
class Glm52Config:
    n_layers: int
    d_model: int
    n_heads: int
    # MLA low-rank stacks
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_head_dim: int
    # FFN: first_k_dense dense-SwiGLU layers, the rest MoE
    d_ff: int
    first_k_dense: int
    # MoE
    n_experts: int
    top_k: int
    d_ff_expert: int
    n_group: int
    topk_group: int
    routed_scaling: float
    n_shared_experts: int
    d_ff_shared: int
    # DSA lightning indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # IndexShare per-layer roles ("full" leader | "shared" follower)
    indexer_types: tuple[str, ...]
    vocab_size: int
    rope_base: float = 8_000_000.0

    def __post_init__(self) -> None:
        if len(self.indexer_types) != self.n_layers:
            raise ValueError("indexer_types must have one entry per layer")
        if self.indexer_types[0] != "full":
            raise ValueError("layer 0 must be 'full' (it seeds the indices)")
        if any(r not in ("full", "shared") for r in self.indexer_types):
            raise ValueError("indexer_types entries must be full|shared")
        if any(self.indexer_types[i] != "full" for i in range(self.first_k_dense)):
            raise ValueError("dense-FFN layers must all be full (leaders)")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    def is_leader(self, layer: int) -> bool:
        return self.indexer_types[layer] == "full"

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_k_dense


# --- shared primitives (reimplemented locally) --------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((xf * rstd).to(x.dtype) * self.weight).to(x.dtype)


def rope_tables(seq_len: int, rope_dim: int, base: float, device,
                dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) each ``(seq_len, rope_dim)`` — positions reset per row (every
    ``(B, T)`` row is an independent length-T sequence)."""
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device,
                                       dtype=torch.float32) / rope_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    emb = torch.cat((torch.outer(pos, inv),) * 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the last dim. x: ``(B, T, H, rope_dim)``;
    cos/sin: ``(T, rope_dim)`` fp32."""
    xf = x.float()
    out = xf * cos[None, :, None, :] + _rotate_half(xf) * sin[None, :, None, :]
    return out.to(x.dtype)


def swiglu(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, silu rounded to storage dtype before the product."""
    return F.silu(x1.float()).to(x1.dtype) * x3


def _causal_mask(t: int, device) -> torch.Tensor:
    """(t, t) additive mask: 0 on/below the diagonal, -inf strictly above."""
    return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)


# --- attention: MLA + DSA sparse selection ------------------------------------

class Attention(nn.Module):
    """MLA (low-rank q + low-rank kv + decoupled rope) with DSA sparse
    selection. LEADER layers own the lightning indexer and compute the
    group's selection mask; FOLLOWER layers reuse the passed-in mask and
    carry no indexer weights."""

    def __init__(self, cfg: Glm52Config, leader: bool):
        super().__init__()
        self.cfg = cfg
        self.leader = leader
        d, h = cfg.d_model, cfg.n_heads
        qk, v = cfg.qk_head_dim, cfg.v_head_dim
        self.w_q_a = nn.Linear(d, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank)
        self.w_q_b = nn.Linear(cfg.q_lora_rank, h * qk, bias=False)
        self.w_kv_a = nn.Linear(d, cfg.kv_lora_rank + cfg.qk_rope_dim, bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_kv_b = nn.Linear(cfg.kv_lora_rank, h * (cfg.qk_nope_dim + v), bias=False)
        self.wo = nn.Linear(h * v, d, bias=False)
        if leader:
            self.w_idx_q = nn.Linear(cfg.q_lora_rank,
                                     cfg.index_n_heads * cfg.index_head_dim, bias=False)
            self.w_idx_k = nn.Linear(d, cfg.index_head_dim, bias=False)
            self.idx_k_ln_w = nn.Parameter(torch.ones(cfg.index_head_dim))
            self.idx_k_ln_b = nn.Parameter(torch.zeros(cfg.index_head_dim))
            # per-head selection weights (fp32 by the engine's dtype policy)
            self.w_idx_w = nn.Parameter(torch.zeros(d, cfg.index_n_heads))

    @torch.no_grad()
    def _select(self, h1: torch.Tensor, q_lora: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Lightning indexer -> per-token causal top-k -> additive {0,-inf}
        selection mask ``(B, T, T)``. Non-differentiable (the CE never reaches
        the indexer — the paper's seam); computed once by the group leader and
        reused by its followers. fp32 throughout."""
        c = self.cfg
        B, T, _ = h1.shape
        hi, di, r = c.index_n_heads, c.index_head_dim, c.qk_rope_dim
        # q^I from the shared post-norm q-latent; rope-FIRST head layout
        q = self.w_idx_q(q_lora).view(B, T, hi, di)
        q = torch.cat([apply_rope(q[..., :r], cos, sin), q[..., r:]], dim=-1)
        # k^I: one shared key per token, standard LayerNorm then rope-first
        k = F.layer_norm(self.w_idx_k(h1).float(), (di,),
                         self.idx_k_ln_w.float(), self.idx_k_ln_b.float(), LN_EPS)
        k = torch.cat([apply_rope(k[..., :r].unsqueeze(2), cos, sin).squeeze(2),
                       k[..., r:]], dim=-1)                         # (B, T, di)
        wts = (h1.float() @ self.w_idx_w.float()) * (hi ** -0.5) * (di ** -0.5)
        r_th = torch.einsum("bqhd,bkd->bqhk", q.float(), k)         # (B,T,hi,T)
        scores = (wts.unsqueeze(-1) * r_th.clamp_min(0.0)).sum(2)   # (B,T,T)
        causal = _causal_mask(T, h1.device)
        scores = scores + causal
        kk = min(c.index_topk, T)
        idx = torch.topk(scores, kk, dim=-1).indices               # (B,T,kk)
        m = torch.full((B, T, T), float("-inf"), device=h1.device)
        m.scatter_(-1, idx, 0.0)
        # re-add causal so any future-index pad slots are re-suppressed
        return m + causal

    def forward(self, h1: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                shared_mask: torch.Tensor | None):
        c = self.cfg
        B, T, _ = h1.shape
        h, nope, r, v = c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim
        qk = c.qk_head_dim

        q_lora = self.q_a_norm(self.w_q_a(h1))
        q = self.w_q_b(q_lora).view(B, T, h, qk)
        q_full = torch.cat([q[..., :nope], apply_rope(q[..., nope:], cos, sin)], dim=-1)

        kv_a = self.w_kv_a(h1)
        latent = self.kv_a_norm(kv_a[..., :c.kv_lora_rank])
        k_rope = apply_rope(kv_a[..., c.kv_lora_rank:].unsqueeze(2), cos, sin).squeeze(2)
        kvb = self.w_kv_b(latent).view(B, T, h, nope + v)
        k_full = torch.cat(
            [kvb[..., :nope], k_rope.unsqueeze(2).expand(B, T, h, r)], dim=-1)
        v_heads = kvb[..., nope:]                                    # (B,T,h,v)

        mask = self._select(h1, q_lora, cos, sin) if self.leader else shared_mask

        # masked SDPA sparse core: causality + selection both live in the mask;
        # scale = qk^-0.5 (v carried at its native width — equal to the padded-v
        # convention: softmax(QK^T)@[V|0] sliced to v is softmax(QK^T)@V).
        qh, kh, vh = (t.transpose(1, 2) for t in (q_full, k_full, v_heads))
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[:, None].to(qh.dtype))
        o = o.transpose(1, 2).reshape(B, T, h * v)
        return self.wo(o), mask


# --- FFN: dense SwiGLU and MoE ------------------------------------------------

class DenseMLP(nn.Module):
    def __init__(self, cfg: Glm52Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, h2: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(h2), self.w3(h2)))


class MoE(nn.Module):
    """DeepSeek-V3 sigmoid_noaux_tc MoE + one ungated additive shared expert.
    Operates on flattened ``(N, d)`` tokens; returns the full block output
    (residual included, per the pinned combine convention)."""

    def __init__(self, cfg: Glm52Config):
        super().__init__()
        self.cfg = cfg
        d, E, f, fs = cfg.d_model, cfg.n_experts, cfg.d_ff_expert, cfg.d_ff_shared
        self.w_router = nn.Linear(d, E, bias=False)
        # non-gradient balance bias: fixed zero here (the noaux bias-update rule
        # is a training-time heuristic and stays omitted — it is separate from
        # the optional load-balancing loss, which just reads the router scores)
        self.register_buffer("w_router_bias", torch.zeros(E))
        self.w13_experts = nn.Parameter(torch.empty(E, d, 2 * f))
        self.w2_experts = nn.Parameter(torch.empty(E, f, d))
        self.w_s13 = nn.Linear(d, 2 * fs, bias=False)
        self.w_s2 = nn.Linear(fs, d, bias=False)
        # per-layer routed load-balancing aux term (fp32 scalar), recomputed each
        # forward and summed by ``Glm52.load_balance_loss``; None until first run
        self.aux_lbl: torch.Tensor | None = None

    def _route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(weights fp32 (N,K), ids int64). Weights are differentiable through
        the selected sigmoid scores; ids are the discrete group-limited top-k
        (smallest-index tie-break via stable descending sort)."""
        c = self.cfg
        scores = torch.sigmoid(logits.float())                      # (N, E)
        with torch.no_grad():
            sel = scores + self.w_router_bias                       # selection scores
            N, E = sel.shape
            g = sel.view(N, c.n_group, E // c.n_group)
            g_sorted, _ = torch.sort(g, dim=-1, descending=True, stable=True)
            group_score = g_sorted[..., :min(2, g.shape[-1])].sum(-1)
            _, g_idx = torch.sort(group_score, dim=-1, descending=True, stable=True)
            group_mask = torch.zeros(N, c.n_group, dtype=torch.bool, device=logits.device)
            group_mask.scatter_(1, g_idx[:, :c.topk_group], True)
            expert_mask = group_mask.repeat_interleave(E // c.n_group, dim=1)
            masked = sel.masked_fill(~expert_mask, float("-inf"))
            _, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
            ids = idx[:, :c.top_k]
        picked = scores.gather(1, ids)                              # raw sigmoid scores
        w = picked / picked.sum(-1, keepdim=True) * c.routed_scaling
        return w, ids

    def _load_balance(self, logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Routed-expert load-balancing auxiliary term for this layer (fp32
        scalar, NO coefficient): ``E · Σ_e f_e·p̄_e``.

        ``f_e`` = fraction of the discrete top-K routed assignments landing on
        expert ``e`` (``count_e/(T·K)`` via ``bincount``, detached — the
        selection carries no gradient). ``p̄_e`` = batch-mean of the full-E
        normalized-sigmoid router probability (``s = sigmoid(logits)``,
        ``p = s/s.sum(-1)``, ``p̄ = p.mean(tokens)``; gradient flows through
        ``p̄``). Matches the engine's ``moe_seq_aux_loss_reference`` P; the
        shared expert is excluded. Uniform routing gives ``1.0``; correlated
        (imbalanced) routing gives ``> 1``."""
        E = self.cfg.n_experts
        with torch.no_grad():
            counts = torch.bincount(ids.reshape(-1), minlength=E).float()   # (E,)
            f = counts / ids.numel()                                # count_e/(T·K)
        s = torch.sigmoid(logits.float())                           # (T, E)
        p = s / s.sum(-1, keepdim=True)
        p_bar = p.mean(0)                                           # (E,) over tokens
        return E * (f * p_bar).sum()

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        f = c.d_ff_expert
        logits = self.w_router(h2)                                 # (T, E) routed
        route_w, ids = self._route(logits)
        self.aux_lbl = self._load_balance(logits, ids)             # fp32 scalar, no α
        routed = torch.zeros_like(h2, dtype=torch.float32)          # fp32 accumulate
        for e in range(c.n_experts):
            coef = (route_w * (ids == e)).sum(-1)                   # (N,) <=1 hit/row
            h13 = h2 @ self.w13_experts[e]
            act = swiglu(h13[:, :f], h13[:, f:])
            routed = routed + coef[:, None] * (act @ self.w2_experts[e]).float()
        s13 = self.w_s13(h2)
        shared = self.w_s2(swiglu(s13[:, :c.d_ff_shared], s13[:, c.d_ff_shared:]))
        base = resid + shared                                       # V3: plain additive
        return (base.float() + routed).to(h2.dtype)


# --- block + model ------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: Glm52Config, layer: int):
        super().__init__()
        self.cfg = cfg
        self.leader = cfg.is_leader(layer)
        self.moe = cfg.is_moe(layer)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, leader=self.leader)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = MoE(cfg) if self.moe else DenseMLP(cfg)

    def forward(self, x, cos, sin, shared_mask):
        a, mask = self.attn(self.attn_norm(x), cos, sin, shared_mask)
        h_mid = x + a
        h2 = self.ffn_norm(h_mid)
        if self.moe:
            B, T, d = h_mid.shape
            y = self.ffn(h2.reshape(B * T, d), h_mid.reshape(B * T, d)).reshape(B, T, d)
        else:
            y = h_mid + self.ffn(h2)
        return y, mask


class Glm52(nn.Module):
    """Untied-embedding GLM-5.2. ``forward`` takes ``(B, T)`` int tokens where
    each row is an independent causal sequence. The group leader's DSA
    selection mask is threaded across its trailing shared layers."""

    def __init__(self, cfg: Glm52Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # recompute each block in the backward (activation checkpointing) to
        # trade compute for memory; off by default.
        self.grad_checkpoint = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.embed(tokens)
        cos, sin = rope_tables(T, self.cfg.qk_rope_dim, self.cfg.rope_base, x.device)
        mask = None   # each group opens with a full (leader) layer that sets it
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x, mask = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, mask, use_reentrant=False)
            else:
                x, mask = blk(x, cos, sin, mask)
        return self.lm_head(self.final_norm(x))

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the per-layer routed-expert load-balancing
        auxiliary term ``E·Σ_e f_e·p̄_e`` stashed by the most recent forward
        (fp32 scalar; ``0.0`` if the model has no MoE layer or has not been
        run). No coefficient is applied here — ``loss`` scales it by
        ``aux_coef``."""
        terms = [blk.ffn.aux_lbl for blk in self.blocks
                 if blk.moe and blk.ffn.aux_lbl is not None]
        if not terms:
            return torch.zeros((), device=self.lm_head.weight.device)
        return torch.stack(terms).sum()

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0) -> torch.Tensor:
        """Mean cross-entropy over all tokens (fp32). ``tokens``/``targets`` are
        ``(B, T)`` int; targets are the next-token ids.

        ``loss()`` returns mean CE; pass ``aux_coef>0`` to add the routed-expert
        load-balancing auxiliary loss ``α·E·Σ_e f_e·p̄_e`` (α=``aux_coef``,
        ``p̄`` = mean normalized-sigmoid router prob), summed over MoE layers —
        matches the engine's balance loss; the shared expert is excluded. (The
        DSA indexer-KL objective remains omitted.) ``aux_coef=0`` (the default)
        is exactly the pure-CE value."""
        logits = self.forward(tokens)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
        if aux_coef > 0:
            ce = ce + aux_coef * self.load_balance_loss()
        return ce
