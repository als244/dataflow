"""Block-level pure-autograd reference forwards for the generic
block-backward gate (gradcheck.check_block_backward).

Composed EXCLUSIVELY from the pinned reference op library
(tasks.ops.*_reference) — the same layer-3 anchors the kernel pins
use — over packed-leaf dicts at engine field names. One function per
homogeneous family that runs the generic ladder; heterogeneous
families gate their per-kind blocks in their own test modules.
"""
from __future__ import annotations

import torch

from dataflow_training.blocks import ops


def llama3_block_forward(dims, x: torch.Tensor, w: dict,
                         seg) -> torch.Tensor:
    d = dims
    h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
    pos = seg.positions
    q = ops.rope_fwd(h1 @ w["wq"], pos, d.n_heads, d.head_dim, d.rope_base)
    k = ops.rope_fwd(h1 @ w["wk"], pos, d.n_kv_heads, d.head_dim,
                     d.rope_base)
    v = h1 @ w["wv"]
    attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads,
                                   d.head_dim, seg)
    h_mid = x + attn @ w["wo"]
    h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
    x1 = h2 @ w["w1"]
    x3 = h2 @ w["w3"]
    return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]


def qwen3_block_forward(dims, x: torch.Tensor, w: dict,
                        seg) -> torch.Tensor:
    d = dims
    t, h, kvh, hd = d.tokens, d.n_heads, d.n_kv_heads, d.head_dim
    h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
    qm = (h1 @ w["wq"]).view(t, h, hd)
    km = (h1 @ w["wk"]).view(t, kvh, hd)
    qn = ops.rmsnorm_reference(qm, w["q_norm_w"]).view(t, d.q_dim)
    kn = ops.rmsnorm_reference(km, w["k_norm_w"]).view(t, d.kv_dim)
    pos = seg.positions
    q = ops.rope_fwd(qn, pos, h, hd, d.rope_base)
    k = ops.rope_fwd(kn, pos, kvh, hd, d.rope_base)
    v = h1 @ w["wv"]
    attn = ops.attention_reference(q, k, v, h, kvh, hd, seg)
    h_mid = x + attn @ w["wo"]
    h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
    x1 = h2 @ w["w1"]
    x3 = h2 @ w["w3"]
    return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]


BLOCK_FORWARDS = {
    "llama3": llama3_block_forward,
    "qwen3": qwen3_block_forward,
}
