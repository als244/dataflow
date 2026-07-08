"""Golden DeepSeek-V3.2 reference: plain eager torch + autograd.

Extends GoldenDsv3 (mixed depth, noaux bias rule, AdamW replica) with
DSA in every layer's attention: lightning-indexer scores on DETACHED
inputs, top-k selection, mask-form sparse core, and the indexer KL term
added to the per-block aux (objective = CE + sum(moe aux + L_I); CE-only
reported — the runtime injects the same gradients analytically).

Training-schedule note: the indexer trains at the shared AdamW lr (the
paper uses a separate lr) — the runtime does the same, so parity gates
compare like against like.

TWO TRAINING MODES, one golden: SPARSE (top-k of the indexer's
scores selects the attention live set; per-layer KL on gathered
targets) and DENSE WARM-UP (sparse_mode=False: causal attention,
full-prefix KL targets, mains frozen via dims.opt_policy — only the
indexer steps). The mode branches live in
dsa_selection_mask_reference and the optimizer policy, not in this
file's control flow.

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
    Dsv32Dims,
    PackedLayout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
)
from dataflow.tasks.modules.mla_reference import mla_qkv_reference
from dataflow.tasks.modules.moe.reference import moe_mlp_reference, moe_topk_reference


_IDX_FIELDS = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")


@dataclass
class GoldenDsv32(GoldenDsv3):
    dims: Dsv32Dims  # re-typed

    def train_step(self, tokens, targets) -> float:
        if getattr(self.dims, "sparse_mode", True):
            return super().train_step(tokens, targets)
        # dense warm-up: the OBJECTIVE is the indexer KL alone — the
        # specialized program has no head, no CE, no dy chain. The
        # reported loss is the summed per-layer KL(p || sigma), exactly
        # the engine's loss_{s}_{r} accumulator. ``targets`` is unused.
        self._pending_counts = []
        for p_ in self.parameters():
            p_.grad = None
        kl_total = self.warmup_kl(tokens)
        kl_total.backward()
        self.step_count += 1
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", leaves)
        return float(kl_total.detach())

    def warmup_kl(self, tokens) -> "torch.Tensor":
        """Forward through the blocks (no head) summing each layer's
        KL — for dsv32 the training objective and the reported value
        coincide (every layer is its own group)."""
        x = self.w_embed["w"][tokens.long()]
        total = None
        for w in self.w_blocks:
            x, kl = self.block_forward(x, w)
            total = kl if total is None else total + kl
        return total

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        if layer is not None and self.dims.kind_of(layer) == "dense":
            return dsv32_dense_weight_layout(self.dims, layer=layer)
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        t = x.shape[0]
        h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, w, d)

        # indexer scores (input DETACHED — the paper's seam: CE never
        # reaches the indexer) + the mode's mask (sparse top-k live set
        # vs dense warm-up causal — dsa_selection_mask_reference)
        scores = dsa_index_scores_reference(h1.detach(), q_lora.detach(), w, d)
        mask = dsa_selection_mask_reference(scores, d, t, x.device)
        train_idx = getattr(d, "train_indexer", True)

        qf = q_full.reshape(t, h * qk)
        kf = k_full.reshape(t, h * qk)
        vp = v_pad.reshape(t, h * qk)
        attn = dsa_sparse_attention_reference(qf, kf, vp, mask, d)
        attn = attn.view(t, h, qk)[..., :v].reshape(t, h * v)
        h_mid = x + attn @ w["wo"]

        # per-layer indexer objective: KL(p || sigma) with p = this
        # layer's own attention rows on the mask's live set
        if train_idx:
            p = dsa_attention_rows_reference(q_full, k_full, mask, d, t)
            kl = dsa_indexer_kl_reference(scores, mask, p)
        else:
            kl = torch.zeros((), dtype=torch.float32, device=x.device)

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
        if not getattr(d, "sparse_mode", True):
            return y, kl          # warm-up objective excludes the aux term
        return y, aux + kl
