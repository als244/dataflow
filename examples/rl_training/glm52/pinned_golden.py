"""Capture/pinned variants of the glm52 golden for the RL example.

- CAPTURE mode (fake_inference): runs the plain golden forward while
  recording, per layer, the block INPUT (the activation checkpoint), the
  leader's top-k selection, and the MoE routing decisions — the exact
  payloads a real inference engine would save.
- PINNED mode (reference_trainer): consumes those saved selections and
  routings verbatim (mask from saved indices, route_ids pinned), so the
  autograd backward sees the inference engine's exact choices — the same
  contract the runtime enforces through M objects + train_indexer=False.

The forward body mirrors ``models/glm52_reference.py`` (kept in sync by
the example's parity gate itself: any drift fails the gate).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.models.glm52_reference import GoldenGlm52
from dataflow.tasks import ops
from dataflow.tasks.dsa_reference import (
    dsa_index_scores_reference,
    dsa_mask_from_idx,
    dsa_sparse_attention_reference,
    dsa_topk_reference,
)
from dataflow.tasks.mla_reference import mla_qkv_reference
from dataflow.tasks.moe.reference import moe_mlp_reference, moe_topk_reference


@dataclass
class RlGlm52(GoldenGlm52):
    """capture=True records; saved=<artifacts> pins."""

    capture: bool = False
    saved: dict | None = None

    def reset_capture(self):
        self.captured = {"x": [], "sel": {}, "route_ids": {}, "route_w": {}}

    def block_forward(self, x, w, route_ids=None):
        d = self.dims
        i = self._layer_ptr
        self._layer_ptr += 1
        t = x.shape[0]
        h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim

        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, w, d)

        if self.saved is not None:
            if d.role_of(i) == "full":
                sel = self.saved["sel"][i].to(x.device)
                self._group_mask = dsa_mask_from_idx(sel, d, t)
                self._group_scores = None  # frozen: no indexer autograd
            mask = self._group_mask
        else:
            if d.role_of(i) == "full":
                scores = dsa_index_scores_reference(
                    h1.detach(), q_lora.detach(), w, d)
                if not getattr(d, "train_indexer", True):
                    scores = scores.detach()
                sel = dsa_topk_reference(scores.detach(), d.index_topk)
                if self.capture:
                    self.captured["sel"][i] = sel.detach().to(torch.int32).cpu()
                mask = dsa_mask_from_idx(sel, d, t)
                self._group_scores, self._group_mask = scores, mask
            mask = self._group_mask

        qf = q_full.reshape(t, h * qk)
        kf = k_full.reshape(t, h * qk)
        vp = v_pad.reshape(t, h * qk)
        attn = dsa_sparse_attention_reference(qf, kf, vp, mask, d)
        attn = attn.view(t, h, qk)[..., :v].reshape(t, h * v)
        h_mid = x + attn @ w["wo"]

        # frozen indexer: no KL term anywhere (matches the runtime's
        # train_indexer=False contract)
        kl = torch.zeros((), device=x.device)

        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        if "w13_experts" not in w:
            s = ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"])
            return h_mid + s @ w["w2"], kl

        ids = route_ids
        if self.saved is not None:
            ids = self.saved["route_ids"][i].to(x.device)
        if ids is None:
            with torch.no_grad():
                logits = h2 @ w["w_router"]
                rw, ids = moe_topk_reference(
                    logits, d.moe.top_k, d.moe.routing_mode,
                    bias=w["w_router_bias"].float(),
                    n_group=d.moe.n_group, topk_group=d.moe.topk_group,
                    routed_scaling=d.moe.routed_scaling,
                )
            if self.capture:
                self.captured["route_ids"][i] = ids.detach().to(torch.int32).cpu()
                self.captured["route_w"][i] = rw.detach().to(torch.bfloat16).cpu()
        lens = self._seq_lens()
        y, aux = moe_mlp_reference(h2, w, d.moe, h_mid,
                                   route_ids=ids, seq_lens=lens)
        with torch.no_grad():
            cnt = torch.bincount(ids.reshape(-1),
                                 minlength=d.moe.n_experts).float()
        if not hasattr(self, "_pending_counts"):
            self._pending_counts = []
        self._pending_counts.append(cnt)
        return y, aux + kl
