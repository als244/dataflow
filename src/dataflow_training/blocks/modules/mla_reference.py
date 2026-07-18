"""Multi-head Latent Attention (MLA, DeepSeek-V3) — REFERENCE forms only.

NOT a task/executable: this is the pure-autograd correctness anchor
(parallel to tasks/moe/reference.py) that goldens and ladders compose.
The runtime executables live in dsv3_blocks.py as staged BlockFwd/
BlockBwd classes like every other family.

Semantics pinned here (no flextrain reference exists for DeepSeek; the
anchor is HF ``modeling_deepseek`` + these autograd forms, tested against
hand-written backwards in the block executables):

- Two low-rank stacks with RMSNorm mid-stack:
    q:  x @ w_q_a (d -> q_lora) -> rmsnorm -> @ w_q_b (-> h*(nope+rope))
    kv: x @ w_kv_a (d -> kv_lora + rope); latent = rmsnorm(first kv_lora),
        k_rope = LAST rope dims (ONE 64-dim vector per token, shared by
        every head — "decoupled rope"); latent @ w_kv_b (-> h*(nope+v)).
- Rope (standard rotate-half, ``ops.rope_fwd``) applies ONLY to the rope
  dims: per-head on q's rope slice, single-"head" on the shared k_rope.
- Per-head attention dims: qk = nope + rope (192 at 671B), v = nope (128).
  Flash/SDPA take one shared head_dim, so V IS ZERO-PADDED to qk dims —
  EXACT: softmax(QK^T)@[V|0] = [softmax(QK^T)@V | 0], and the padded
  output/gradient columns are identically zero (pinned by test). Softmax
  scale = qk_dim^-0.5 — SDPA's default at head_dim=qk, which is V3's
  native-training scale (yarn mscale is an inference-time long-context
  concern; documented, not modeled).
- Saved-ctx design (the MLA win, used by the block executables): the
  compressed latents (q_lora optional, kv latent + k_rope MANDATORY) are
  what's worth saving — backward re-expands through w_q_b/w_kv_b, so
  attention context per layer is ~kv_lora+rope wide instead of h*(qk+v).
"""
from __future__ import annotations

import torch

from .. import ops


def mla_head_dims(dims) -> tuple[int, int]:
    """(qk_dim, v_dim) per head."""
    return dims.qk_nope_dim + dims.qk_rope_dim, dims.v_head_dim


def mla_qkv_reference(
    h1: torch.Tensor, w: dict[str, torch.Tensor], dims, segments=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The MLA expansion shared by dense (dsv3) and DSA (dsv32) forms:
    returns (q_lora_n (t, q_lora), q_full (t,h,qk), k_full (t,h,qk),
    v_pad (t,h,qk)) — padded-v convention. ``segments`` supplies the
    per-sequence rope positions (``segments.positions``); None derives the
    round segmentation from ``dims``."""
    d = dims
    t = h1.shape[0]
    h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
    qk = nope + rope
    seg = segments if segments is not None else ops.Segments.of_dims(dims).on(h1.device)
    pos = seg.positions

    # q stack
    q_lora = ops.rmsnorm_reference(h1 @ w["w_q_a"], w["q_a_norm_w"])
    q = (q_lora @ w["w_q_b"]).view(t, h, qk)
    q_rope = ops.rope_fwd(
        q[..., nope:].reshape(t, h * rope).contiguous(), pos, h, rope, d.rope_base,
    ).view(t, h, rope)
    q_full = torch.cat([q[..., :nope], q_rope], dim=-1)

    # kv stack: latent + shared decoupled rope
    kv_a = h1 @ w["w_kv_a"]
    latent = ops.rmsnorm_reference(kv_a[:, : d.kv_lora_rank], w["kv_a_norm_w"])
    k_rope = ops.rope_fwd(
        kv_a[:, d.kv_lora_rank:].contiguous(), pos, 1, rope, d.rope_base,
    )
    kvb = (latent @ w["w_kv_b"]).view(t, h, nope + v)
    k_full = torch.cat(
        [kvb[..., :nope], k_rope.view(t, 1, rope).expand(t, h, rope)], dim=-1,
    )
    v_pad = torch.cat(
        [kvb[..., nope:], torch.zeros(t, h, qk - v, dtype=h1.dtype, device=h1.device)],
        dim=-1,
    )
    return q_lora, q_full, k_full, v_pad


def mla_attention_reference(
    h1: torch.Tensor, w: dict[str, torch.Tensor], dims, segments=None,
) -> torch.Tensor:
    """Post-attn-norm input h1 (t, d) -> attention output (t, n_heads*v)
    (caller applies wo + residual). Pure autograd; mirrors the runtime
    stage decomposition 1:1 so ladders compare piecewise. ``segments`` (the
    round's ``Segments``; None derives it from ``dims``) supplies both the
    rope positions and the block-diagonal attention structure."""
    d = dims
    t = h1.shape[0]
    h, v = d.n_heads, d.v_head_dim
    qk = d.qk_nope_dim + d.qk_rope_dim
    seg = segments if segments is not None else ops.Segments.of_dims(dims).on(h1.device)
    _, q_full, k_full, v_pad = mla_qkv_reference(h1, w, dims, seg)
    attn = ops.attention_reference(
        q_full.reshape(t, h * qk), k_full.reshape(t, h * qk),
        v_pad.reshape(t, h * qk), h, h, qk, seg,
    )
    return attn.view(t, h, qk)[..., :v].reshape(t, h * v)


def mla_block_reference(
    x: torch.Tensor, w: dict[str, torch.Tensor], dims, segments=None,
) -> torch.Tensor:
    """Full attention half of a DSv3 block: norm -> MLA -> wo -> residual."""
    h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
    attn = mla_attention_reference(h1, w, dims, segments)
    return x + attn @ w["wo"]
