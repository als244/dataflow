"""Independent Qwen3.5-MoE — a plain PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the qwen3.5 MoE hybrid family, reimplemented
from scratch and FULLY SELF-CONTAINED: this file imports only ``torch`` —
nothing from ``dataflow`` and nothing from the sibling ``references`` modules
(the qwen3.5 hybrid attention is COPIED in here verbatim, not imported, so a
bug shared with ``reference_models/qwen35.py`` cannot hide). It is the most complex
member of the family: the dense qwen3.5 hybrid attention with every layer's
dense SwiGLU MLP replaced by a routed Mixture-of-Experts tail plus one
sigmoid-gated shared expert.

Each block interleaves two mixer kinds, then feeds a MoE tail:

  - **linear (Gated DeltaNet)** — RMSNorm → w_qkvz / w_ba projections →
    depthwise causal conv1d + silu over [q|k|v] → L2-normed q/k → the gated
    delta-rule recurrence (fp32 state, reset per row) → gated
    RMSNorm(silu(z)·rmsnorm·w) → output proj;
  - **full (gated attention)** — RMSNorm → wq=[Q|gate] / wk / wv → per-head
    RMSNorm qk-norm → PARTIAL RoPE (first rot_dim channels) → GQA causal
    attention → output gate σ(gate) → output proj;
  - **MoE tail** — RMSNorm(xo) = h; router GEMM → top-K experts with
    ``topk_then_softmax`` (pick the top-K logits, softmax over just those K;
    smallest-index tie-break) → masked expert loop (per-expert SwiGLU,
    fp32-accumulated routed sum) → ONE sigmoid-gated shared expert. The block
    output follows the pinned combine convention:

        out = xo + σ(w_shared_gate·h)·swiglu_shared(h) + routed_moe

    with ``routed_moe`` accumulated in fp32, the gated shared term rounded to
    the storage dtype and added to the residual, and the (base + routed) sum
    rounded once at the end.

``kind_of(i)`` = "full" when ``(i+1) % full_attention_interval == 0`` else
"lin". ``loss()`` returns mean CE; pass ``aux_coef>0`` to add the
routed-expert load-balancing auxiliary loss α·E·Σ_e f_e·p̄_e (α=aux_coef),
summed over MoE layers — the standard Switch/GShard term (matches the
engine); the shared expert is not part of it. Untied LM head (the 35B-A3B
config is untied).

Numeric conventions match the engine so the loss curves track within bf16
kernel-order noise: bf16 storage with fp32 reductions (RMSNorm, RoPE,
softmax, the delta-rule state, all routing math, and CE reduce in fp32 then
cast back), RMS eps 1e-5, L2 eps 1e-6, rope base 1e7. Weight orientation for
the parity bridge matches llama3/qwen35 (projection ``nn.Linear`` weight =
packed.T; conv = packed.unsqueeze(1); the stacked expert / 1-D params load
direct).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-5
L2NORM_EPS = 1e-6


@dataclass(frozen=True)
class Qwen35MoeConfig:
    n_layers: int
    d_model: int
    full_attention_interval: int
    # full-attention sub-block (gated, per-head qk-norm, partial rope)
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    # linear-attention sub-block (Gated DeltaNet)
    lin_k_heads: int
    lin_v_heads: int
    lin_k_head_dim: int
    lin_v_head_dim: int
    lin_conv_kernel: int
    # MoE FFN tail (every layer): routed experts + one sigmoid-gated shared expert
    n_experts: int
    top_k: int
    d_ff_expert: int
    n_shared_experts: int
    d_ff_shared: int
    vocab_size: int
    routing_mode: str = "topk_then_softmax"
    aux_coef: float = 0.001  # engine's LB training coefficient; loss() defaults to CE-only (aux_coef=0)
    rope_base: float = 10_000_000.0
    tied_embeddings: bool = False  # the 35B config is untied

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rot_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.lin_k_heads * self.lin_k_head_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def qkvz_dim(self) -> int:
        return 2 * self.key_dim + 2 * self.value_dim

    @property
    def ba_dim(self) -> int:
        return 2 * self.lin_v_heads

    def kind_of(self, layer: int) -> str:
        return "full" if (layer + 1) % self.full_attention_interval == 0 else "lin"


# --- primitives (self-contained copies; no cross-module imports) --------------


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


def l2norm(x: torch.Tensor) -> torch.Tensor:
    """Per-row (last-dim) L2 normalization (fla convention: SUM, eps 1e-6)."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + L2NORM_EPS)
    return (xf * rstd).to(x.dtype)


def partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 rot_dim: int) -> torch.Tensor:
    """Rotate only the first ``rot_dim`` channels of each head (rotate-half);
    the rest pass through. x: (B,T,H,head_dim); cos/sin: (T,rot_dim)."""
    xr = x[..., :rot_dim].float()
    rot = xr * cos[None, :, None, :] + _rotate_half(xr) * sin[None, :, None, :]
    return torch.cat([rot.to(x.dtype), x[..., rot_dim:]], dim=-1)


def swiglu(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) rounded to the storage dtype BEFORE the product
    (matches the engine's swiglu_fwd — silu in fp32, cast, then * value)."""
    return F.silu(x1.float()).to(x1.dtype) * x3


def delta_rule(q, k, v, beta, g, scale=None):
    """Sequential gated delta rule (fp32 state), batched over (B, heads):

        S = S * exp(g);  u = beta*(v - k·S);  S = S + k⊗u;  o = (scale·q)·S

    q,k: (B,T,HK,K) post-l2norm; v: (B,T,HV,V); beta,g: (B,T,HV). GVA:
    v-head i reads k-head i // (HV//HK). Each batch row is an independent
    sequence, so its state starts at zero (matches per-segment reset)."""
    B, T, HK, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    rep = HV // HK
    qf = q.float().repeat_interleave(rep, dim=2)
    kf = k.float().repeat_interleave(rep, dim=2)
    vf, betaf, gf = v.float(), beta.float(), g.float()
    if scale is None:
        scale = K ** -0.5
    state = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        state = state * gf[:, t].exp()[:, :, None, None]
        err = vf[:, t] - torch.einsum("bhk,bhkv->bhv", kf[:, t], state)
        upd = betaf[:, t][:, :, None] * err
        state = state + kf[:, t][:, :, :, None] * upd[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", qf[:, t] * scale, state))
    return torch.stack(outs, dim=1).to(v.dtype)


class GatedRMSNorm(nn.Module):
    """RMSNormGated: silu(z) * rmsnorm(o) * w over the last dim."""

    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        of = o.float()
        rstd = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
        y = F.silu(z.float()) * (of * rstd)
        return (y * self.weight.float()).to(o.dtype)


# --- mixers (self-contained copies of the qwen3.5 hybrid attention) -----------


def packed_positions(seq_lens, device) -> torch.Tensor:
    """Per-token rope positions for a PACKED round: every sequence
    restarts at 0. (varlen mode — see Model.forward(seq_lens=...))."""
    return torch.cat([torch.arange(n, device=device) for n in seq_lens])


def block_causal_mask(seq_lens, device) -> torch.Tensor:
    """(T, T) additive {0, -inf} fp32 mask for a packed round: causal
    WITHIN each sequence, -inf across sequences (block-diagonal varlen
    attention for the full-attention mixers; the DeltaNet mixers reset
    state per segment instead — see GatedDeltaNet.forward)."""
    t = int(sum(seq_lens))
    m = torch.full((t, t), float("-inf"), device=device)
    lo = 0
    for n in seq_lens:
        m[lo:lo + n, lo:lo + n] = torch.triu(
            torch.full((n, n), float("-inf"), device=device), diagonal=1)
        lo += n
    return m


def sequence_bounds(seq_lens) -> tuple[tuple[int, int], ...]:
    """Flat-token (lo, hi) per sequence for a packed round."""
    out, lo = [], 0
    for n in seq_lens:
        out.append((lo, lo + n))
        lo += n
    return tuple(out)


class GatedDeltaNet(nn.Module):
    """The linear (Gated DeltaNet) mixer, including its output projection."""

    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.cfg = cfg
        self.w_qkvz = nn.Linear(cfg.d_model, cfg.qkvz_dim, bias=False)
        self.w_ba = nn.Linear(cfg.d_model, cfg.ba_dim, bias=False)
        self.conv = nn.Conv1d(cfg.conv_dim, cfg.conv_dim, cfg.lin_conv_kernel,
                              groups=cfg.conv_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(cfg.lin_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.lin_v_heads))
        self.lin_norm = GatedRMSNorm(cfg.lin_v_head_dim)
        self.w_out = nn.Linear(cfg.value_dim, cfg.d_model, bias=False)

    def _conv_silu(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,conv_dim) -> depthwise causal conv1d (left pad W-1) + silu
        W = self.cfg.lin_conv_kernel
        xf = F.pad(x.float().transpose(1, 2), (W - 1, 0))     # (B, D, T+W-1)
        y = F.conv1d(xf, self.conv.weight.float(), groups=self.cfg.conv_dim)
        return F.silu(y.transpose(1, 2)).to(x.dtype)          # (B,T,D)

    def forward(self, h1: torch.Tensor, seq_bounds=None) -> torch.Tensor:
        if seq_bounds is not None:
            # packed varlen: the recurrence (conv left-pad + delta-rule
            # state) restarts per sequence — run each segment through the
            # single-sequence path and concatenate. Exact, not approximate:
            # both the conv pad and the state are zero-initialized.
            return torch.cat([self.forward(h1[:, lo:hi])
                              for lo, hi in seq_bounds], dim=1)
        c = self.cfg
        B, T, _ = h1.shape
        qkvz = self.w_qkvz(h1)
        ba = self.w_ba(h1)
        conv_in = qkvz[..., : c.conv_dim]
        z = qkvz[..., c.conv_dim:].view(B, T, c.lin_v_heads, c.lin_v_head_dim)
        b = ba[..., : c.lin_v_heads]
        a = ba[..., c.lin_v_heads:]
        post = self._conv_silu(conv_in)
        q = l2norm(post[..., : c.key_dim].view(B, T, c.lin_k_heads, c.lin_k_head_dim))
        k = l2norm(post[..., c.key_dim: 2 * c.key_dim].view(B, T, c.lin_k_heads, c.lin_k_head_dim))
        v = post[..., 2 * c.key_dim:].view(B, T, c.lin_v_heads, c.lin_v_head_dim)
        beta = torch.sigmoid(b.float()).to(h1.dtype)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        core = delta_rule(q, k, v, beta, g)
        o = self.lin_norm(core, z)
        return self.w_out(o.reshape(B, T, c.value_dim))


class GatedAttention(nn.Module):
    """The full (gated attention) mixer, including its output projection."""

    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, 2 * cfg.attn_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)
        self.wo = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

    def forward(self, h1, cos, sin, mask=None) -> torch.Tensor:
        c = self.cfg
        B, T, _ = h1.shape
        H, KV, hd = c.n_heads, c.n_kv_heads, c.head_dim
        qg = self.wq(h1)
        qm, gate = qg[..., : c.attn_dim], qg[..., c.attn_dim:]
        qn = self.q_norm(qm.view(B, T, H, hd))
        kn = self.k_norm(self.wk(h1).view(B, T, KV, hd))
        q = partial_rope(qn, cos, sin, c.rot_dim).transpose(1, 2)   # (B,H,T,hd)
        k = partial_rope(kn, cos, sin, c.rot_dim)
        v = self.wv(h1).view(B, T, KV, hd)
        rep = H // KV
        k = k.repeat_interleave(rep, dim=2).transpose(1, 2)
        v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:       # packed varlen: block-diagonal causality in the mask
            o = F.scaled_dot_product_attention(q, k, v,
                                               attn_mask=mask.to(q.dtype))
        o = o.transpose(1, 2).reshape(B, T, c.attn_dim)
        gated = o * torch.sigmoid(gate.float()).to(o.dtype)
        return self.wo(gated)


# --- MoE tail -----------------------------------------------------------------


class MoEMLP(nn.Module):
    """Routed top-K MoE (topk_then_softmax) + one sigmoid-gated shared expert.

    Golden weight-name map (orientation ``out = x @ w`` throughout):
      router      -> "w_router"        (d, E)
      w13         -> "w13_experts"     (E, d, 2F)   [x1 | x3] packed on 2F
      w2          -> "w2_experts"      (E, F, d)
      shared_gate -> "w_shared_gate"   (d, 1)
      shared_up   -> "w_s13"           (d, 2Fs)     [x1 | x3]
      shared_down -> "w_s2"            (Fs, d)

    Routing/aux math is fp32 from the bf16 router logits. loss() returns mean
    CE; pass aux_coef>0 to add the routed-expert load-balancing auxiliary loss
    α·E·Σ_e f_e·p̄_e (α=aux_coef), summed over MoE layers — the standard
    Switch/GShard term (matches the engine); the shared expert is not part of
    it. The forward stashes this layer's α-free L_layer in ``self.aux_lbl``.
    Selection is non-differentiable; the routing WEIGHTS stay differentiable
    through the softmax over the selected top-K logits, and the aux stays
    differentiable through p̄.
    """

    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.cfg = cfg
        d, E, F_ = cfg.d_model, cfg.n_experts, cfg.d_ff_expert
        fs = cfg.d_ff_shared
        self.router = nn.Linear(d, E, bias=False)
        self.w13 = nn.Parameter(torch.empty(E, d, 2 * F_))
        self.w2 = nn.Parameter(torch.empty(E, F_, d))
        nn.init.normal_(self.w13, std=d ** -0.5)
        nn.init.normal_(self.w2, std=F_ ** -0.5)
        if cfg.n_shared_experts:
            self.shared_gate = nn.Linear(d, cfg.n_shared_experts, bias=False)
            self.shared_up = nn.Linear(d, 2 * fs, bias=False)
            self.shared_down = nn.Linear(fs, d, bias=False)
        # round-global LBL state (see forward); ints detached, p_sum live
        self.step_counts: torch.Tensor | None = None
        self.round_p_sum: torch.Tensor | None = None
        self.round_tokens = 0

    def reset_round_lbl(self) -> None:
        self.step_counts = None
        self.round_p_sum = None
        self.round_tokens = 0

    def round_lbl(self) -> torch.Tensor:
        """ROUND-global L_layer = E * sum_e f_e * pbar_e from the pieces
        accumulated across the round's forwards — the engine's DEFAULT
        per-round LBL (round-global counts/probs, crossing sequence
        boundaries within the round; memory-efficient, ga-variant)."""
        t = self.round_tokens
        f = self.step_counts.float() / (t * self.cfg.top_k)
        return self.cfg.n_experts * (f * (self.round_p_sum / t)).sum()

    def _route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # topk_then_softmax: pick the top-K logits, softmax over just those K.
        # Smallest-index tie-break via a STABLE descending sort (torch.topk's
        # tie-break does not honor it). Weights sum to 1 (norm_topk_prob).
        lf = logits.float()
        vals, idx = torch.sort(lf, dim=-1, descending=True, stable=True)
        ids = idx[:, : self.cfg.top_k]
        weights = torch.softmax(vals[:, : self.cfg.top_k], dim=-1)
        return weights, ids

    def forward(self, h2: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        # h2: post-ffn-norm activation; resid: post-attention residual stream.
        c = self.cfg
        B, T, d = h2.shape
        F_ = c.d_ff_expert
        hf = h2.reshape(B * T, d)
        rf = resid.reshape(B * T, d)

        logits = self.router(hf)                             # bf16 GEMM (N, E)
        weights, ids = self._route(logits)                   # (N, K) fp32 / int

        # routed-expert load-balancing auxiliary loss (Switch/GShard), fp32
        # scalar with NO α: L_layer = E·Σ_e f_e·p̄_e. f_e = count_e/(T·K) from
        # the discrete top-K ids (detached counts); p̄_e = mean over tokens of
        # the FULL-E router softmax (gradient flows through p̄). The always-on
        # shared expert has no router and is excluded. Stashed for
        # Qwen35Moe.load_balance_loss(); the aux_coef=0 objective ignores it.
        p_full = torch.softmax(logits.float(), dim=-1)                 # (N, E)
        p_bar = p_full.mean(dim=0)                                     # (E,)
        counts_i = torch.bincount(ids.reshape(-1), minlength=c.n_experts)
        f = counts_i.float() / ids.numel()                   # ids.numel() = T·K
        self.aux_lbl = c.n_experts * (f * p_bar).sum()
        # ROUND-global LBL pieces (engine-default semantics): detached
        # counts + LIVE prob sums accumulated across the round's forwards;
        # the parity harness combines them via round_load_balance_loss()
        # and resets with reset_round_lbl()
        p_sum = p_full.sum(dim=0)                                      # LIVE
        self.step_counts = (counts_i if self.step_counts is None
                            else self.step_counts + counts_i)
        self.round_p_sum = (p_sum if self.round_p_sum is None
                            else self.round_p_sum + p_sum)
        self.round_tokens += p_full.shape[0]

        # masked expert loop: at most one top-K slot hits each expert per row
        routed = torch.zeros(B * T, d, dtype=torch.float32, device=hf.device)
        for e in range(c.n_experts):
            coef = (weights * (ids == e)).sum(-1)            # (N,) fp32
            h13 = hf @ self.w13[e]                           # (N, 2F)
            act = swiglu(h13[:, :F_], h13[:, F_:])
            routed = routed + coef[:, None] * (act @ self.w2[e]).float()

        base = rf
        if c.n_shared_experts:
            fs = c.d_ff_shared
            s13 = self.shared_up(hf)
            s_act = swiglu(s13[:, :fs], s13[:, fs:])
            sh = self.shared_down(s_act)                     # (N, d) storage dtype
            gate = torch.sigmoid(self.shared_gate(hf).float())   # (N, 1) fp32
            base = rf + (gate * sh.float()).to(rf.dtype)

        out = (base.float() + routed).to(h2.dtype)
        return out.reshape(B, T, d)


# --- block / model ------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig, layer: int):
        super().__init__()
        self.kind = cfg.kind_of(layer)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.mixer = GatedAttention(cfg) if self.kind == "full" else GatedDeltaNet(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = MoEMLP(cfg)

    def forward(self, x, cos, sin, mask=None, seq_bounds=None):
        h1 = self.attn_norm(x)
        mix = (self.mixer(h1, cos, sin, mask) if self.kind == "full"
               else self.mixer(h1, seq_bounds))
        xo = x + mix
        return self.moe(self.ffn_norm(xo), xo)   # residual-included output


class Qwen35Moe(nn.Module):
    """Untied-embedding qwen3.5-MoE. ``forward`` takes ``(B, T)`` int tokens
    where each row is an independent causal sequence (uniform packing)."""

    # load-balance form the training-parity harness can rely on:
    # "forward_global" (see gradcheck.reference_model_step)
    AUX_FORM = "forward_global"
    SUPPORTS_PACKED = True

    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.embed.weight
        # recompute each block in the backward (activation checkpointing) to
        # trade compute for memory — only needed for the largest model on a
        # single card; off by default.
        self.grad_checkpoint = False

    def forward(self, tokens: torch.Tensor,
                seq_lens: tuple[int, ...] | None = None) -> torch.Tensor:
        B, T = tokens.shape
        x = self.embed(tokens)
        if seq_lens is None:
            cos, sin = rope_tables(T, self.cfg.rot_dim, self.cfg.rope_base,
                                   x.device)
            mask = None
            bounds = None
        else:
            if B != 1 or T != int(sum(seq_lens)):
                raise ValueError(f"packed mode expects (1, sum(seq_lens)) "
                                 f"tokens; got {tuple(tokens.shape)} for "
                                 f"{seq_lens}")
            cos, sin = rope_tables(max(seq_lens), self.cfg.rot_dim,
                                   self.cfg.rope_base, x.device)
            pos = packed_positions(seq_lens, x.device)
            cos, sin = cos[pos], sin[pos]
            mask = block_causal_mask(seq_lens, x.device)
            bounds = sequence_bounds(seq_lens)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, mask, bounds, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, bounds)
        return self.lm_head(self.final_norm(x))


    def reset_round_lbl(self) -> None:
        """Clear every MoE layer's accumulated round-LBL pieces (call
        between rounds / steps in multi-round harnesses)."""
        for m in self.modules():
            if hasattr(m, "round_p_sum"):
                m.reset_round_lbl()

    def round_load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the ROUND-global load-balance term
        (round_lbl) — the engine-default per-round LBL. The harness
        applies aux_coef and adds it ONCE per round after the round's
        forwards; contrast load_balance_loss() (per-forward form)."""
        total = torch.zeros((), dtype=torch.float32,
                            device=self.embed.weight.device)
        for m in self.modules():
            if getattr(m, "round_p_sum", None) is not None:
                total = total + m.round_lbl()
        return total

    def load_balance_loss(self) -> torch.Tensor:
        """Sum over MoE layers of the routed-expert load-balancing auxiliary
        loss L_layer = E·Σ_e f_e·p̄_e (α-free), as stashed by the most recent
        ``forward``. Returns a fp32 0.0 scalar when there are no MoE layers."""
        total = torch.zeros((), dtype=torch.float32,
                            device=self.embed.weight.device)
        for blk in self.blocks:
            aux = getattr(blk.moe, "aux_lbl", None)
            if aux is not None:
                total = total + aux
        return total

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             aux_coef: float = 0.0,
             seq_lens: tuple[int, ...] | None = None) -> torch.Tensor:
        """loss() returns mean CE; pass aux_coef>0 to add the routed-expert
        load-balancing auxiliary loss α·E·Σ_e f_e·p̄_e (α=aux_coef), summed
        over MoE layers — the standard Switch/GShard term (matches the
        engine); the shared expert is not part of it. Mean cross-entropy over
        all tokens (fp32) matches the engine's per-round HeadLoss
        normalization. ``tokens``/``targets`` are ``(B, T)`` int next-token
        ids; the default (aux_coef=0) is CE-only."""
        logits = self.forward(tokens, seq_lens=seq_lens)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1).long())
        if aux_coef > 0:
            return ce + aux_coef * self.load_balance_loss()
        return ce
