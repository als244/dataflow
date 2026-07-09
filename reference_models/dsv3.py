"""Independent DeepSeek-V3 — a plain, idiomatic PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the DeepSeek-V3 arm of the pretraining parity
study, reimplemented from scratch and FULLY SELF-CONTAINED: this file imports
ONLY ``torch`` — nothing from ``dataflow`` and nothing from the sibling
``reference_models/`` modules (RMSNorm/RoPE/SwiGLU AND the full MLA attention + the
MoE router/experts/combine are reimplemented locally even though that
duplicates code, because isolation is the point). It reads like a normal MoE
transformer and lets autograd derive the backward pass; a from-scratch second
implementation guards against a shared bug in the engine's reference ops.

DeepSeek-V3 = MLA (Multi-head Latent Attention) + hybrid depth (``first_k_dense``
dense-SwiGLU layers, then MoE layers):

  - **MLA attention** — ``RMSNorm(x) = h1``, then two low-rank stacks:
      q : ``h1 @ w_q_a`` (d -> ``q_lora_rank``) -> RMSNorm -> ``@ w_q_b``
          (-> ``n_heads*(qk_nope_dim + qk_rope_dim)``);
      kv: ``h1 @ w_kv_a`` (d -> ``kv_lora_rank + qk_rope_dim``); the
          ``kv_lora_rank`` part -> RMSNorm(latent) -> ``@ w_kv_b``
          (-> ``n_heads*(qk_nope_dim + v_head_dim)``); the trailing
          ``qk_rope_dim`` columns are the ONE decoupled RoPE key shared by
          every head.
    RoPE (rotate-half, base 1e4) applies ONLY to the ``qk_rope_dim`` slices
    (q's per-head rope part + the shared k rope). Per-head causal attention
    with q/k head dim ``qk = qk_nope_dim + qk_rope_dim`` and v head dim
    ``v_head_dim``; softmax scale ``qk**-0.5`` (SDPA's default at the q/k head
    dim — V3's native-training scale; the engine's padded-V identity
    ``softmax(QK^T) @ [V|0] == [softmax(QK^T) @ V | 0]`` is left unpadded here,
    exact). Output proj ``wo`` (``n_heads*v_head_dim`` -> d).
  - **dense FFN** (layers < ``first_k_dense``) — standard SwiGLU MLP.
  - **MoE FFN** (the rest) — router GEMM -> ``sigmoid_noaux_tc`` selection:
    ``scores = sigmoid(logits)``; SELECTION on ``(scores + balance_bias)``
    under the GROUP LIMIT (``n_group`` score groups ranked by the sum of each
    group's top-2 selection scores, only the best ``topk_group`` groups stay
    eligible), then greedy top-K (smallest-index tie-break via a stable
    descending sort); WEIGHTS = the selected RAW sigmoid scores renormalized
    to sum 1, times ``routed_scaling``. Masked per-expert SwiGLU over packed
    ``w13_experts (E, d, 2F)`` [gate|up] / ``w2_experts (E, F, d)``,
    fp32-accumulated, plus ONE UNGATED (plain additive) shared expert (V3 has
    no shared-gate field). Combine: fp32 routed accumulator + bf16
    (residual + shared), rounded once.

LOAD-BALANCE AUX (optional) — ``loss()`` returns mean CE; pass ``aux_coef>0``
to add the routed-expert load-balancing auxiliary loss ``α·E·Σ_e f_e·p̄_e``
(``α = aux_coef``; ``f_e`` = fraction of the discrete top-K assignments landing
on expert ``e``, from the selected ids; ``p̄`` = mean normalized-sigmoid router
prob over the batch's tokens, with the gradient flowing through ``p̄``), summed
over the MoE layers — this matches the engine's balance loss. The SHARED expert
is excluded (it has no router). Default ``aux_coef=0`` → pure CE.

SEPARATE mechanism, left out (a training-dynamics concern, NOT the aux above and
not modeled by this pure forward): the DeepSeek aux-free bias-update rule — the
NON-GRADIENT balance bias is a zero-initialized buffer read by selection ONLY;
its per-step sign-rule update (``b += speed*sign(mean - count)`` on the step's
expert counts) is an optimizer-time concern owned by the training harness.

Numeric conventions MATCH the engine (curves track within bf16 kernel-order
noise, not a divergent fp32 model): weights/activations bf16; RMSNorm, RoPE,
the sigmoid/softmax routing math and the CE loss reduce in fp32 then cast
back; RMS eps 1e-5 (the engine keeps the global 1e-5, NOT HF DeepSeek's 1e-6;
standing note); SwiGLU rounds ``silu(gate)`` to bf16 before the product; the
MoE combine accumulates routed contributions in fp32 and rounds
``(base + routed)`` to bf16 exactly once. Untied LM head.

Weight ORIENTATION (for the parity bridge): the dense/MLA projections, the
router and the shared expert are ``nn.Linear`` (weight ``(out, in)``), so
``linear(x) == x @ packed`` when the bridge loads ``linear.weight = packed.T``
(the engine stores ``(in, out)``). The stacked expert weights
``w13_experts`` / ``w2_experts`` are raw parameters already in the engine's
packed ``(E, ...)`` orientation and load directly (``h2 @ w13_experts[e]``),
as do the ``(vocab, d)`` embedding and LM-head tables; the 1-D RMSNorm gains
and the balance bias load directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5


@dataclass(frozen=True)
class Dsv3Config:
    n_layers: int
    d_model: int
    # MLA attention
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_dim: int       # per-head non-rope q/k dim
    qk_rope_dim: int       # per-head (and shared-key) rope dim
    v_head_dim: int        # per-head value dim
    # hybrid depth: first_k_dense dense-SwiGLU layers, then MoE
    first_k_dense: int
    d_ff_dense: int        # dense-kind SwiGLU hidden width
    # MoE tail (sigmoid_noaux_tc routing + one ungated shared expert)
    n_experts: int         # E (router width / softmax-sigmoid space)
    top_k: int             # K experts routed per token
    d_ff_expert: int       # F: per-expert SwiGLU hidden width
    n_group: int           # score groups for the group-limited selection
    topk_group: int        # groups kept eligible per token
    n_shared_experts: int  # 0 or 1 (V3: 1)
    d_ff_shared: int       # shared-expert SwiGLU hidden width
    vocab_size: int
    routed_scaling: float = 1.0
    rope_base: float = 10_000.0
    # the engine's load-balance α — a suggested default to pass through to
    # ``loss(aux_coef=...)``; the forward itself never reads this field
    # (``loss`` defaults to ``aux_coef=0`` = pure CE, see the module docstring).
    aux_coef: float = 1e-4
    # the balance-bias sign-rule update speed: a SEPARATE optimizer-time
    # mechanism (see the module docstring), not modeled by this pure forward.
    bias_update_speed: float = 0.001

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    def kind_of(self, layer: int) -> str:
        return "dense" if layer < self.first_k_dense else "moe"


# --- primitives (self-contained; no cross-module imports) ---------------------


class RMSNorm(nn.Module):
    """RMSNorm over the last dim: reduce in fp32, cast back, scale by gain.
    Used for the block norms, the model's final norm, and the MLA mid-stack
    q/kv-latent norms."""

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
    (every ``(B, T)`` row is an independent length-``T`` causal sequence)."""
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device,
                                       dtype=torch.float32) / rope_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the rope slice. x: ``(B, T, H, rope_dim)``;
    cos/sin: ``(T, rope_dim)`` fp32 (broadcast over B and the head axis)."""
    xf = x.float()
    out = xf * cos[None, :, None, :] + _rotate_half(xf) * sin[None, :, None, :]
    return out.to(x.dtype)


def swiglu(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """SwiGLU: ``silu(gate)`` rounded to the storage dtype BEFORE the product
    (matches the engine's swiglu_fwd — silu in fp32, cast, then * value)."""
    return F.silu(x1.float()).to(x1.dtype) * x3


# --- MLA attention ------------------------------------------------------------


class MLA(nn.Module):
    """Multi-head Latent Attention: low-rank q + low-rank kv with a single
    decoupled RoPE key shared across heads. Takes the post-attn-norm input
    ``h1`` and returns the wo-projected attention output (the block adds the
    residual)."""

    def __init__(self, cfg: Dsv3Config):
        super().__init__()
        self.cfg = cfg
        h, qk = cfg.n_heads, cfg.qk_head_dim
        self.w_q_a = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank)
        self.w_q_b = nn.Linear(cfg.q_lora_rank, h * qk, bias=False)
        self.w_kv_a = nn.Linear(cfg.d_model, cfg.kv_lora_rank + cfg.qk_rope_dim, bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_kv_b = nn.Linear(cfg.kv_lora_rank, h * (cfg.qk_nope_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(h * cfg.v_head_dim, cfg.d_model, bias=False)

    def forward(self, h1: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        B, T, _ = h1.shape
        H, nope, rope, v = c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim
        qk = nope + rope

        # q low-rank stack: down -> RMSNorm -> up -> (nope | rope) per head
        q_lora = self.q_a_norm(self.w_q_a(h1))                 # (B, T, q_lora_rank)
        q = self.w_q_b(q_lora).view(B, T, H, qk)
        q_rope = apply_rope(q[..., nope:], cos, sin)           # rope on the rope slice
        q = torch.cat([q[..., :nope], q_rope], dim=-1)         # (B, T, H, qk)

        # kv low-rank stack: latent (RMSNorm'd) + ONE shared decoupled rope key
        kv_a = self.w_kv_a(h1)                                 # (B, T, kv_lora + rope)
        latent = self.kv_a_norm(kv_a[..., :c.kv_lora_rank])
        k_rope = apply_rope(kv_a[..., c.kv_lora_rank:].unsqueeze(2), cos, sin)  # (B, T, 1, rope)
        kvb = self.w_kv_b(latent).view(B, T, H, nope + v)
        k = torch.cat([kvb[..., :nope], k_rope.expand(B, T, H, rope)], dim=-1)  # (B, T, H, qk)
        vv = kvb[..., nope:]                                   # (B, T, H, v)

        # per-head causal attention; SDPA scale = qk**-0.5 (default at head_dim=qk)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2), is_causal=True,
        )                                                      # (B, H, T, v)
        o = o.transpose(1, 2).reshape(B, T, H * v)
        return self.wo(o)


# --- FFN kinds ----------------------------------------------------------------


class DenseMLP(nn.Module):
    """Dense SwiGLU MLP (the first_k_dense layers). Folds the residual into
    its output (``resid + swiglu @ w2``) so the block stays uniform."""

    def __init__(self, cfg: Dsv3Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff_dense, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff_dense, bias=False)   # up
        self.w2 = nn.Linear(cfg.d_ff_dense, cfg.d_model, bias=False)   # down

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        act = swiglu(self.w1(h2), self.w3(h2))
        return resid + self.w2(act)


class MoEMLP(nn.Module):
    """DeepSeek-V3 MoE tail: sigmoid_noaux_tc router -> masked per-expert
    SwiGLU -> fp32 combine + ONE ungated (plain additive) shared expert.

    Golden weight-name map (orientation ``out = x @ w`` throughout):
      router       -> "w_router"       (d, E)
      balance bias -> "w_router_bias"  (E,)   NON-GRADIENT buffer (init 0)
      w13          -> "w13_experts"    (E, d, 2F)  [gate | up] packed on 2F
      w2           -> "w2_experts"     (E, F, d)
      shared up    -> "w_s13"          (d, 2Fs)    [gate | up]
      shared down  -> "w_s2"           (Fs, d)

    Selection is non-differentiable (runs under no_grad, and the bias enters
    selection ONLY); the routing WEIGHTS stay differentiable through the raw
    sigmoid scores of the selected experts. After routing, the per-layer
    load-balance aux term ``E·Σ_e f_e·p̄_e`` is computed and stashed on
    ``self.aux_lbl`` (``f`` from the selected ids, ``p̄`` from the bias-free
    normalized sigmoid of the router logits); the model's ``load_balance_loss()``
    sums it and ``loss()`` adds ``α·`` it only when ``aux_coef>0``. The
    balance-bias update rule is a SEPARATE optimizer-time mechanism, not computed
    here (see the module docstring)."""

    def __init__(self, cfg: Dsv3Config):
        super().__init__()
        self.cfg = cfg
        d, E, F_ = cfg.d_model, cfg.n_experts, cfg.d_ff_expert
        self.router = nn.Linear(d, E, bias=False)
        # NON-GRADIENT balance bias: zero-initialized, read by selection only;
        # the per-step sign-rule update is an optimizer-time concern (see docstring).
        self.register_buffer("router_bias", torch.zeros(E))
        # per-layer routed-expert load-balance aux term (E·Σ_e f_e·p̄_e),
        # (re)computed and stashed by every forward; 0 until the first forward.
        self.aux_lbl = torch.zeros(())
        self.w13 = nn.Parameter(torch.empty(E, d, 2 * F_))
        self.w2 = nn.Parameter(torch.empty(E, F_, d))
        # fan-in init so a from-scratch (non-bridged) model is well-conditioned;
        # the parity bridge overwrites both stacks with the engine's init.
        nn.init.normal_(self.w13, std=d ** -0.5)
        nn.init.normal_(self.w2, std=F_ ** -0.5)
        if cfg.n_shared_experts:
            fs = cfg.d_ff_shared
            self.w_s13 = nn.Linear(d, 2 * fs, bias=False)
            self.w_s2 = nn.Linear(fs, d, bias=False)

    def _route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sigmoid_noaux_tc: group-limited biased selection, raw-sigmoid
        renormalized weights x routed_scaling. Returns (weights fp32 (N, K),
        ids int64 (N, K)). Smallest-index tie-break via stable descending sort
        (expert AND group level)."""
        c = self.cfg
        scores = torch.sigmoid(logits.float())                 # (N, E) fp32
        N, E = scores.shape
        ng, tg, K = c.n_group, c.topk_group, c.top_k
        with torch.no_grad():
            sel = scores + self.router_bias.float()            # bias enters SELECTION only
            g = sel.view(N, ng, E // ng)
            g_sorted, _ = torch.sort(g, dim=-1, descending=True, stable=True)
            group_score = g_sorted[..., :min(2, E // ng)].sum(-1)   # (N, ng) top-2 per group
            _, g_idx = torch.sort(group_score, dim=-1, descending=True, stable=True)
            keep = g_idx[:, :tg]                               # best topk_group groups
            gmask = torch.zeros(N, ng, dtype=torch.bool, device=logits.device)
            gmask.scatter_(1, keep, True)
            emask = gmask.repeat_interleave(E // ng, dim=1)
            masked = sel.masked_fill(~emask, float("-inf"))
            _, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
            ids = idx[:, :K]
        picked = scores.gather(1, ids)                         # raw sigmoid scores (differentiable)
        weights = picked / picked.sum(-1, keepdim=True) * c.routed_scaling
        return weights, ids

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        # h2: post-ffn-norm activation; resid: post-attention residual stream.
        c = self.cfg
        B, T, d = h2.shape
        F_ = c.d_ff_expert
        hf = h2.reshape(B * T, d)
        rf = resid.reshape(B * T, d)

        logits = self.router(hf)                               # bf16 GEMM (N, E)
        weights, ids = self._route(logits)                     # (N, K) fp32 / int

        # routed-expert load-balance aux term  L = E·Σ_e f_e·p̄_e  (fp32, no α;
        # load_balance_loss() sums it, loss() adds α· it only when aux_coef>0).
        #   f_e : fraction of the n_tok·K discrete top-K assignments on expert e,
        #         from the selected ids (bincount, non-differentiable);
        #   p̄_e : mean over tokens of the FULL-E normalized-sigmoid router prob
        #         (bias-free logits; gradient flows through p̄).
        n_tok, K = hf.shape[0], ids.shape[1]
        counts = torch.bincount(ids.reshape(-1), minlength=c.n_experts).float()
        f = counts / (n_tok * K)                               # (E,) sums to 1
        s = torch.sigmoid(logits.float())
        p_bar = (s / s.sum(-1, keepdim=True)).mean(0)          # (E,) grad flows
        self.aux_lbl = c.n_experts * (f * p_bar).sum()         # fp32 scalar

        # dropless masked E-loop: at most one top-K slot hits each expert per row
        routed = torch.zeros(B * T, d, dtype=torch.float32, device=hf.device)
        for e in range(c.n_experts):
            coef = (weights * (ids == e)).sum(-1)              # (N,) fp32
            h13 = hf @ self.w13[e]                             # (N, 2F) bf16
            act = swiglu(h13[:, :F_], h13[:, F_:])
            routed = routed + coef[:, None] * (act @ self.w2[e]).float()

        base = rf
        if c.n_shared_experts:                                 # V3: ungated, plain additive
            fs = c.d_ff_shared
            s13 = self.w_s13(hf)
            s_act = swiglu(s13[:, :fs], s13[:, fs:])
            base = rf + self.w_s2(s_act)                       # bf16 add

        out = (base.float() + routed).to(h2.dtype)
        return out.reshape(B, T, d)


# --- block / model ------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: Dsv3Config, layer: int):
        super().__init__()
        self.kind = cfg.kind_of(layer)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = DenseMLP(cfg) if self.kind == "dense" else MoEMLP(cfg)

    def forward(self, x, cos, sin):
        h_mid = x + self.attn(self.attn_norm(x), cos, sin)
        # both FFN kinds fold the residual into their output (return h_mid + ffn)
        return self.ffn(self.ffn_norm(h_mid), h_mid)


class Dsv3(nn.Module):
    """Untied-embedding DeepSeek-V3. ``forward`` takes ``(B, T)`` int tokens
    where each row is an independent causal sequence (uniform packing)."""

    def __init__(self, cfg: Dsv3Config):
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
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.lm_head(self.final_norm(x))

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over the MoE layers of the per-layer routed-expert load-balance
        aux term ``E·Σ_e f_e·p̄_e`` (no α) stashed on each ``MoEMLP`` by its most
        recent forward; a zero scalar if the model has no MoE layers. ``loss()``
        scales this by ``aux_coef``. The shared expert (no router) is excluded."""
        terms = [blk.ffn.aux_lbl for blk in self.blocks
                 if isinstance(blk.ffn, MoEMLP)]
        if not terms:
            return self.lm_head.weight.new_zeros(())
        return torch.stack(terms).sum()

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0) -> torch.Tensor:
        """Mean cross-entropy over all tokens (fp32) — matches the engine's
        per-round HeadLoss normalization. ``tokens``/``targets`` are ``(B, T)``
        int next-token ids. Pass ``aux_coef>0`` to add the routed-expert
        load-balancing auxiliary loss ``α·E·Σ_e f_e·p̄_e`` (``α = aux_coef``,
        ``p̄`` = mean normalized-sigmoid router prob) summed over the MoE layers
        — this matches the engine's balance loss; the shared expert is excluded.
        Default ``aux_coef=0`` → pure CE, bit-identical to the aux-free path."""
        logits = self.forward(tokens)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
        )
        if aux_coef > 0:
            ce = ce + aux_coef * self.load_balance_loss()
        return ce
