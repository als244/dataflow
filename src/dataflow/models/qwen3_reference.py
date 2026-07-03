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
from dataflow.tasks.layouts import Qwen3Dims, qwen3_weight_layout
from dataflow.models.llama3_reference import GoldenLlama3


@dataclass
class GoldenQwen3(GoldenLlama3):
    dims: Qwen3Dims  # re-typed, position and (lack of) default inherited

    def _block_views(self, flat_bf16: torch.Tensor) -> dict[str, torch.Tensor]:
        wl = qwen3_weight_layout(self.dims)
        out: dict[str, torch.Tensor] = {}
        for f in wl.fields:
            n = 1
            for dim in f.shape:
                n *= int(dim)
            start = f.offset_bytes // 2
            out[f.name] = flat_bf16[start : start + n].view(f.shape)
        return out

    def block_forward(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.dims
        t, h, kvh, hd = d.tokens, d.n_heads, d.n_kv_heads, d.head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qm = (h1 @ w["wq"]).view(t, h, hd)
        km = (h1 @ w["wk"]).view(t, kvh, hd)
        qn = ops.rmsnorm_reference(qm, w["q_norm_w"]).view(t, d.q_dim)
        kn = ops.rmsnorm_reference(km, w["k_norm_w"]).view(t, d.kv_dim)
        q = ops.rope_fwd(qn, d.seq_len, h, hd, d.rope_base)
        k = ops.rope_fwd(kn, d.seq_len, kvh, hd, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, h, kvh, hd, d.seq_len)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]
