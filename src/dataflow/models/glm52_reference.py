"""Golden GLM-5.2 (IndexShare) reference: plain eager torch + autograd.

Extends GoldenDsv3 (mixed depth, noaux bias rule, AdamW replica) with
DSA + cross-layer index reuse: "full" layers run the lightning indexer
and cache their (scores, mask) as the running group state; "shared"
layers reuse the nearest preceding full layer's selection and carry NO
indexer weights. Every group member — the leader included — contributes
(1/N) * KL(p_member || sigma_leader) to the aux (the paper's L^I_multi;
Proposition 1: the gradient equals aligning sigma to the CENTROID of
the members' targets). Autograd accumulates the member terms on the
leader's shared scores tensor, so the leader's indexer weights receive
exactly sigma - mean(p) — the same gradient the runtime assembles
through dM.

Layer identity comes from a call counter reset in loss_terms (the base
iterates blocks without indices — same order-dependent-state style as
the bias counts capture).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.models.dsv3_reference import GoldenDsv3
from dataflow.tasks import ops
from dataflow.tasks.dsa_reference import (
    dsa_index_scores_reference,
    dsa_indexer_kl_reference,
    dsa_mask_from_idx,
    dsa_sparse_attention_reference,
    dsa_topk_reference,
)
from dataflow.tasks.layouts import (
    Glm52Dims,
    PackedLayout,
    dsv3_moe_weight_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
)
from dataflow.tasks.mla_reference import mla_qkv_reference
from dataflow.tasks.moe.reference import moe_mlp_reference, moe_topk_reference


@dataclass
class GoldenGlm52(GoldenDsv3):
    dims: Glm52Dims  # re-typed

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        if layer is None:
            return dsv32_moe_weight_layout(self.dims, layer=layer)
        kind = self.dims.kind_of(layer)
        if kind == "gdl":
            return dsv32_dense_weight_layout(self.dims, layer=layer)
        if kind == "gml":
            return dsv32_moe_weight_layout(self.dims, layer=layer)
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    def loss_terms(self, tokens, targets):
        self._layer_ptr = 0
        self._group_scores = None   # leader's live scores (autograd node)
        self._group_mask = None
        return super().loss_terms(tokens, targets)

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        i = self._layer_ptr
        self._layer_ptr += 1
        t = x.shape[0]
        h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, w, d)

        if d.role_of(i) == "full":
            scores = dsa_index_scores_reference(h1.detach(), q_lora.detach(), w, d)
            sel = dsa_topk_reference(scores.detach(), d.index_topk)
            mask = dsa_mask_from_idx(sel, d, t)
            self._group_scores, self._group_mask = scores, mask
        scores, mask = self._group_scores, self._group_mask

        qf = q_full.reshape(t, h * qk)
        kf = k_full.reshape(t, h * qk)
        vp = v_pad.reshape(t, h * qk)
        attn = dsa_sparse_attention_reference(qf, kf, vp, mask, d)
        attn = attn.view(t, h, qk)[..., :v].reshape(t, h * v)
        h_mid = x + attn @ w["wo"]

        # this member's target on the shared live set (detached)
        with torch.no_grad():
            p = torch.zeros(t, t, device=x.device)
            scale = qk ** -0.5
            q3 = q_full.detach().float()
            k3 = k_full.detach().float()
            lo = 0
            for L in ops.seq_lens_of(d.seq_spec, t):
                hi = lo + L
                for hh in range(h):
                    lg = (q3[lo:hi, hh] @ k3[lo:hi, hh].T) * scale
                    p[lo:hi, lo:hi] += torch.softmax(
                        lg + mask[lo:hi, lo:hi], dim=-1)
                lo = hi
        n = len(d.group_members(d.leader_of(i)))
        kl = dsa_indexer_kl_reference(scores, mask, p) / n

        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        if "w13_experts" not in w:
            s = ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"])
            return h_mid + s @ w["w2"], kl
        lens = self._seq_lens()
        y, aux = moe_mlp_reference(h2, w, d.moe, h_mid,
                                   route_ids=route_ids, seq_lens=lens)
        if route_ids is None:
            with torch.no_grad():
                logits = h2 @ w["w_router"]
                _, ids = moe_topk_reference(
                    logits, d.moe.top_k, d.moe.routing_mode,
                    bias=w["w_router_bias"].float(),
                    n_group=d.moe.n_group, topk_group=d.moe.topk_group,
                    routed_scaling=d.moe.routed_scaling,
                )
                cnt = torch.bincount(ids.reshape(-1),
                                     minlength=d.moe.n_experts).float()
            if not hasattr(self, "_pending_counts"):
                self._pending_counts = []
            self._pending_counts.append(cnt)
        return y, aux + kl
