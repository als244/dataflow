"""Independent Qwen3.5-dense — a plain PyTorch ``nn.Module`` + autograd.

The correctness GROUND TRUTH for the qwen3.5 hybrid family, reimplemented
from scratch (imports only ``torch`` + shared primitives from ``.llama3``;
nothing from ``dataflow``). Qwen3.5-dense interleaves two mixer kinds under
a shared SwiGLU MLP:

  - **linear (Gated DeltaNet)** — RMSNorm → w_qkvz / w_ba projections →
    depthwise causal conv1d + silu over [q|k|v] → L2-normed q/k → the gated
    delta-rule recurrence → gated RMSNorm(silu(z)·rmsnorm·w) → output proj;
  - **full (gated attention)** — RMSNorm → wq=[Q|gate] / wk / wv → per-head
    RMSNorm qk-norm → PARTIAL RoPE (first rot_dim channels) → GQA causal
    attention → output gate σ(gate) → output proj.

``kind_of(i)`` = "full" when ``(i+1) % full_attention_interval == 0`` else
"lin". Numeric conventions match the engine's reference ops: bf16 storage
with fp32 reductions, RMS eps 1e-5, L2 eps 1e-6, fp32 delta-rule state,
rope base 1e7. Weight orientation for the bridge is the same as llama3
(``nn.Linear`` weight = packed.T; conv = packed.unsqueeze(1); 1-D params
direct).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama3 import RMS_EPS, RMSNorm, _rotate_half, rope_tables

L2NORM_EPS = 1e-6


@dataclass(frozen=True)
class Qwen35Config:
    n_layers: int
    d_model: int
    full_attention_interval: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    lin_k_heads: int
    lin_v_heads: int
    lin_k_head_dim: int
    lin_v_head_dim: int
    lin_conv_kernel: int
    d_ff: int
    vocab_size: int
    rope_base: float = 10_000_000.0
    tied_embeddings: bool = False

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


def seq_bounds_of(seq_lens) -> tuple[tuple[int, int], ...]:
    """Flat-token (lo, hi) per sequence for a packed round."""
    out, lo = [], 0
    for n in seq_lens:
        out.append((lo, lo + n))
        lo += n
    return tuple(out)


class GatedDeltaNet(nn.Module):
    """The linear (Gated DeltaNet) mixer, including its output projection."""

    def __init__(self, cfg: Qwen35Config):
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

    def __init__(self, cfg: Qwen35Config):
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


class MLP(nn.Module):
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.w1(x), self.w3(x)
        return self.w2(F.silu(gate.float()).to(gate.dtype) * up)


class Block(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer: int):
        super().__init__()
        self.kind = cfg.kind_of(layer)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.mixer = GatedAttention(cfg) if self.kind == "full" else GatedDeltaNet(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask=None, seq_bounds=None):
        h1 = self.attn_norm(x)
        mix = (self.mixer(h1, cos, sin, mask) if self.kind == "full"
               else self.mixer(h1, seq_bounds))
        xo = x + mix
        return xo + self.mlp(self.ffn_norm(xo))


class Qwen35(nn.Module):

    SUPPORTS_PACKED = True
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.embed.weight
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
            bounds = seq_bounds_of(seq_lens)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, mask, bounds, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, bounds)
        return self.lm_head(self.final_norm(x))

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, *,
             seq_lens: tuple[int, ...] | None = None) -> torch.Tensor:
        logits = self.forward(tokens, seq_lens=seq_lens)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               targets.reshape(-1).long())
