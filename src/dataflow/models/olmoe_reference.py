"""Golden OLMoE reference: plain eager torch + autograd, hand-written.

Same contract as GoldenLlama3 (reused for packed-leaf handling and the
exact AdamW replica). Block = qwen3-shaped attention with FULL-ROW qk-norm
(one RMSNorm over the whole q/k rows) + the MoE SwiGLU tail composed from
``dataflow.tasks.modules.moe.reference`` (routing modes, smallest-index tie-break,
fp32 combine, shared-expert and aux-loss semantics all pinned there).

Aux load-balance loss: the AUTOGRAD OBJECTIVE is CE + sum of per-layer aux
terms (f detached) — reproducing the runtime's gradient-injected analytic
form exactly — while the RETURNED/reported loss stays CE-only (the runtime
``loss_*`` object is CE; parity gates compare that).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.tasks import ops
from dataflow.tasks.layouts import OlmoeDims, PackedLayout, olmoe_weight_layout
from dataflow.tasks.modules.moe.reference import moe_mlp_reference
from dataflow.models.llama3_reference import GoldenLlama3


@dataclass
class GoldenOlmoe(GoldenLlama3):
    dims: OlmoeDims  # re-typed; position and (lack of) default inherited

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        return olmoe_weight_layout(self.dims, layer=layer)

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # route_ids pins the discrete selection (block-ladder use only;
        # see moe_mlp_reference — E2E paths leave it None)
        d = self.dims
        seg = self._segments(segments, x.device)
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qn = ops.rmsnorm_reference(h1 @ w["wq"], w["q_norm_w"])   # full-row
        kn = ops.rmsnorm_reference(h1 @ w["wk"], w["k_norm_w"])   # full-row
        pos = seg.positions
        q = ops.rope_fwd(qn, pos, d.n_heads, d.head_dim, d.rope_base)
        k = ops.rope_fwd(kn, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(
            q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, seg
        )
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        return moe_mlp_reference(h2, w, d.moe, h_mid, route_ids=route_ids)

    def loss_terms(
        self, tokens: torch.Tensor, targets: torch.Tensor, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(CE, total aux) — CE is the reported loss; CE + aux is the
        autograd objective."""
        seg = self._segments(segments, self.w_embed["w"].device)
        x = self.w_embed["w"][tokens.long()]
        aux_total = torch.zeros((), dtype=torch.float32, device=x.device)
        for w in self.w_blocks:
            x, aux = self.block_forward(x, w, segments=seg)
            aux_total = aux_total + aux
        logits = ops.rmsnorm_reference(x, self.w_head["final_norm_w"]) @ self.w_head["w"].T
        return ops.ce_loss_reference(logits, targets), aux_total

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> torch.Tensor:
        return self.loss_terms(tokens, targets, segments)[0]

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> float:
        for p in self.parameters():
            p.grad = None
        ce, aux_total = self.loss_terms(tokens, targets, segments)
        (ce + aux_total).backward()
        self.step_count += 1
        self._opt_obj("embed", self.w_embed)
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", leaves)
        self._opt_obj("head", self.w_head)
        return float(ce.detach())
