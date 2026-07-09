"""Golden Qwen3-MoE reference: plain eager torch + autograd, hand-written.

Same contract as GoldenOlmoe (whose CE+aux objective / CE-only reported
loss and AdamW replica are inherited unchanged). The block differs only in
attention: qwen3's PER-HEAD qk-norm (one shared ``(head_dim,)`` weight for
q and one for k, applied over head_dim-wide rows) with GQA and rope 1e6.
The MoE tail is composed from ``dataflow.tasks.modules.moe.reference`` with
``topk_then_softmax`` (norm_topk_prob=true) and NO shared expert.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.models.olmoe_reference import GoldenOlmoe
from dataflow.tasks import ops
from dataflow.tasks.layouts import PackedLayout, Qwen3MoeDims, qwen3moe_weight_layout
from dataflow.tasks.modules.moe.reference import moe_mlp_reference


@dataclass
class GoldenQwen3Moe(GoldenOlmoe):
    dims: Qwen3MoeDims  # re-typed; position and (lack of) default inherited

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3moe_weight_layout(self.dims, layer=layer)

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        t, h, kvh, hd = x.shape[0], d.n_heads, d.n_kv_heads, d.head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qm = h1 @ w["wq"]
        km = h1 @ w["wk"]
        # PER-HEAD qk-norm: rmsnorm over head_dim-wide rows
        qn = ops.rmsnorm_reference(
            qm.view(t * h, hd), w["q_norm_w"]).view(t, d.q_dim)
        kn = ops.rmsnorm_reference(
            km.view(t * kvh, hd), w["k_norm_w"]).view(t, d.kv_dim)
        pos = seg.positions
        q = ops.rope_fwd(qn, pos, h, hd, d.rope_base)
        k = ops.rope_fwd(kn, pos, kvh, hd, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, h, kvh, hd, seg)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        return moe_mlp_reference(h2, w, d.moe, h_mid, route_ids=route_ids)
