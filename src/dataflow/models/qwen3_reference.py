"""Golden Qwen3-dense reference: plain eager torch + autograd, hand-written.

Same contract as GoldenLlama3 (which it reuses for everything family-neutral:
packed-leaf handling, loss composition, the exact AdamW replica): forward
math composed from the ops' *reference* forms, packed layouts shared with
the runtime. The block differs by qk-norm — per-head RMSNorm on q/k between
projection and rope, one shared (head_dim,) weight each.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.tasks import ops
from dataflow.tasks.layouts import PackedLayout, Qwen3Dims, qwen3_weight_layout
from dataflow.models.llama3_reference import GoldenLlama3


@dataclass
class GoldenQwen3(GoldenLlama3):
    dims: Qwen3Dims  # re-typed, position and (lack of) default inherited

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3_weight_layout(self.dims, layer=layer)

    def block_forward(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.dims
        t, h, kvh, hd = d.tokens, d.n_heads, d.n_kv_heads, d.head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qm = (h1 @ w["wq"]).view(t, h, hd)
        km = (h1 @ w["wk"]).view(t, kvh, hd)
        qn = ops.rmsnorm_reference(qm, w["q_norm_w"]).view(t, d.q_dim)
        kn = ops.rmsnorm_reference(km, w["k_norm_w"]).view(t, d.kv_dim)
        pos = ops.positions_for(d.seq_spec, x.shape[0], x.device)
        q = ops.rope_fwd(qn, pos, h, hd, d.rope_base)
        k = ops.rope_fwd(kn, pos, kvh, hd, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, h, kvh, hd, d.seq_spec)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]
