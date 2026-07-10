"""Golden DeepSeek-V3 reference: plain eager torch + autograd, hand-written.

Extends GoldenOlmoe (CE + aux objective, CE-only reported loss, AdamW
replica) with: MLA attention (tasks/mla_reference.py forms), MIXED depth
(first_k_dense dense-SwiGLU layers, MoE rest — kind inferred per layer),
sigmoid_noaux_tc routing with the UNGATED shared expert, V3's sequence-
wise aux, and the balance-bias step rule applied EXACTLY like the
runtime: counts captured from the forward's discrete assignments, bias
updated after AdamW with b += speed * sign(mean - count) — the bias
itself never sees AdamW math or autograd gradients.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.models.olmoe_reference import GoldenOlmoe
from dataflow.tasks import ops
from dataflow.tasks.layouts import (
    Dsv3Dims,
    PackedLayout,
    dsv3_dense_weight_layout,
    dsv3_moe_weight_layout,
)
from dataflow.tasks.modules.mla_reference import mla_attention_reference
from dataflow.tasks.modules.moe.reference import moe_mlp_reference


@dataclass
class GoldenDsv3(GoldenOlmoe):
    dims: Dsv3Dims  # re-typed; position and (lack of) default inherited

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        if layer is not None and self.dims.kinds[layer] == "dense":
            return dsv3_dense_weight_layout(self.dims, layer=layer)
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    def _seq_lens(self) -> tuple[int, ...]:
        d = self.dims
        if d.seq_lens is not None:
            return tuple(d.seq_lens)
        return (d.seq_len,) * (d.tokens // d.seq_len)

    def _note_router_counts(self, h2: torch.Tensor, w) -> None:
        """Record this layer's per-expert assignment counts (detached) for
        the noaux router-bias speed rule applied at optimizer time.
        Inherited by every noaux-family golden (dsv32, glm52)."""
        from dataflow.tasks.modules.moe.reference import router_counts_reference

        if not hasattr(self, "_pending_counts"):
            self._pending_counts = []
        self._pending_counts.append(
            router_counts_reference(h2, w, self.dims.moe))

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        attn = mla_attention_reference(h1, w, d, seg)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        if "w13_experts" not in w:                      # dense kind
            s = ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"])
            y = h_mid + s @ w["w2"]
            return y, torch.zeros((), dtype=torch.float32, device=x.device)
        y, aux = moe_mlp_reference(
            h2, w, d.moe, h_mid, route_ids=route_ids, seq_lens=self._seq_lens(),
        )
        # capture the step's discrete assignment counts for the bias rule
        # (mirrors the runtime's counts-through-the-dW-slot aggregation)
        if route_ids is None:
            self._note_router_counts(h2, w)
        return y, aux

    def _opt_obj(self, obj: str, leaves) -> None:
        if "w_router_bias" in leaves:
            rest = {k: v for k, v in leaves.items() if k != "w_router_bias"}
            super()._opt_obj(obj, rest)
        else:
            super()._opt_obj(obj, leaves)

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> float:
        self._pending_counts = []
        for p in self.parameters():
            p.grad = None
        ce, aux_total = self.loss_terms(tokens, targets, segments)
        (ce + aux_total).backward()
        self.step_count += 1
        self._opt_obj("embed", self.w_embed)
        speed = self.dims.moe.bias_update_speed
        counts_iter = iter(self._pending_counts)
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", leaves)
            if "w_router_bias" in leaves and speed:
                c = next(counts_iter)
                b = leaves["w_router_bias"]
                b.data.add_(torch.sign(c.mean() - c).to(b.dtype), alpha=speed)
        self._opt_obj("head", self.w_head)
        return float(ce.detach())
