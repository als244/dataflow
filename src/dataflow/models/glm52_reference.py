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

TWO TRAINING MODES, one golden: SPARSE (leader top-k shared by the
group; members' gathered rows average into L^I_multi) and DENSE
WARM-UP (causal masks, full-prefix rows, mains frozen via
dims.opt_policy). Mode branches live in dsa_selection_mask_reference
and the optimizer policy — block_forward reads identically in both.

"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.models.dsv3_reference import GoldenDsv3
from dataflow.tasks import ops
from dataflow.tasks.modules.dsa_reference import (
    dsa_attention_rows_reference,
    dsa_topk_reference,
    dsa_mask_from_idx,
    dsa_selection_mask_reference,
    dsa_index_scores_reference,
    dsa_indexer_kl_reference,
    dsa_sparse_attention_reference,
)
from dataflow.tasks.layouts import (
    Glm52Dims,
    PackedLayout,
    dsv3_moe_weight_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
)
from dataflow.tasks.modules.mla_reference import mla_qkv_reference
from dataflow.tasks.modules.moe.reference import moe_mlp_reference, moe_topk_reference


@dataclass
class GoldenGlm52(GoldenDsv3):
    dims: Glm52Dims  # re-typed

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        if layer is None:
            return dsv32_moe_weight_layout(self.dims, layer=layer)
        kind = self.dims.kinds[layer]
        if kind == "gdl":
            return dsv32_dense_weight_layout(self.dims, layer=layer)
        if kind == "gml":
            return dsv32_moe_weight_layout(self.dims, layer=layer)
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    def loss_terms(self, tokens, targets, segments=None):
        self._layer_ptr = 0
        self._group_scores = None   # leader's live scores (autograd node)
        self._group_mask = None
        return super().loss_terms(tokens, targets, segments)

    def train_step(self, tokens, targets, segments=None) -> float:
        if getattr(self.dims, "sparse_mode", True):
            return super().train_step(tokens, targets, segments)
        # dense warm-up: the OBJECTIVE is L^I_multi alone — no head, no
        # CE, no dy chain (matching the specialized program). Trained
        # and reported as the per-group CENTROID KL: its sigma-gradient
        # equals the member-sum form, and its VALUE is exactly the
        # engine's loss accumulator. ``targets`` is unused.
        self._pending_counts = []
        for p_ in self.parameters():
            p_.grad = None
        kl_total = self.warmup_kl(tokens, segments)
        kl_total.backward()
        self.step_count += 1
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", leaves)
        return float(kl_total.detach())

    def warmup_kl(self, tokens, segments=None) -> "torch.Tensor":
        """Forward through the blocks (no head), then one KL per GROUP
        against the member-averaged full-prefix target."""
        self._wu_groups = []
        self._layer_ptr = 0
        x = self.w_embed["w"][tokens.long()]
        seg = self._segments(segments, x.device)
        for w in self.w_blocks:
            x, _ = self.block_forward(x, w, segments=seg)
        total = None
        for g in self._wu_groups:
            centroid = g["psum"] / g["n"]
            kl = dsa_indexer_kl_reference(g["scores"], g["mask"], centroid)
            total = kl if total is None else total + kl
        return total

    def _leader_selection(self, h1, q_lora, w, t: int, device, segments=None):
        """Group leader's shared pair (scores, mask). Indexer inputs are
        DETACHED (the paper's seam: CE never reaches the indexer); a
        frozen indexer detaches the scores themselves. The mask states
        the TWO TRAINING MODES in one place: sparse top-k live set vs
        dense warm-up causal (dsa_selection_mask_reference)."""
        d = self.dims
        scores = dsa_index_scores_reference(h1.detach(), q_lora.detach(), w, d, segments)
        if not getattr(d, "train_indexer", True):
            scores = scores.detach()
        return scores, dsa_selection_mask_reference(scores, d, t, device, segments)

    def _member_kl(self, i: int, scores, mask, q_full, k_full, t: int, segments=None):
        """SPARSE mode: this member's contribution to the group
        objective — its own attention rows on the shared live set,
        weighted 1/N; summed over members this IS L^I_multi.

        DENSE WARM-UP: members only deposit their full-prefix rows into
        the group centroid (finalized once per group in warmup_kl —
        gradient sigma - centroid is IDENTICAL to the member-sum form,
        and the value matches the engine's loss accumulator)."""
        d = self.dims
        if not getattr(d, "train_indexer", True):
            return torch.zeros((), device=q_full.device)
        p = dsa_attention_rows_reference(q_full, k_full, mask, d, t, segments)
        if not getattr(d, "sparse_mode", True):
            g = self._wu_groups[-1]
            g["psum"] = p if g["psum"] is None else g["psum"] + p
            g["n"] += 1
            return torch.zeros((), device=q_full.device)
        n = len(d.group_members(d.leader_of(i)))
        return dsa_indexer_kl_reference(scores, mask, p) / n

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        i = self._layer_ptr
        self._layer_ptr += 1
        t = x.shape[0]
        h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, w, d, seg)

        # IndexShare: the group LEADER computes scores + the mode's mask
        # once and caches them; every member (leader included) reads the
        # shared pair below
        if d.role_of(i) == "full":
            self._group_scores, self._group_mask = \
                self._leader_selection(h1, q_lora, w, t, x.device, seg)
            if not getattr(d, "sparse_mode", True):
                if not hasattr(self, "_wu_groups"):
                    self._wu_groups = []
                self._wu_groups.append({"scores": self._group_scores,
                                        "mask": self._group_mask,
                                        "psum": None, "n": 0})
        scores, mask = self._group_scores, self._group_mask

        qf = q_full.reshape(t, h * qk)
        kf = k_full.reshape(t, h * qk)
        vp = v_pad.reshape(t, h * qk)
        attn = dsa_sparse_attention_reference(qf, kf, vp, mask, d, seg)
        attn = attn.view(t, h, qk)[..., :v].reshape(t, h * v)
        h_mid = x + attn @ w["wo"]

        kl = self._member_kl(i, scores, mask, q_full, k_full, t, seg)

        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        if "w13_experts" not in w:
            s = ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"])
            return h_mid + s @ w["w2"], kl
        lens = self._seq_lens()
        y, aux = moe_mlp_reference(h2, w, d.moe, h_mid,
                                   route_ids=route_ids, seq_lens=lens)
        if route_ids is None:
            self._note_router_counts(h2, w)   # inherited (GoldenDsv3)
        if not getattr(d, "sparse_mode", True):
            return y, kl          # warm-up objective excludes the aux term
        return y, aux + kl
