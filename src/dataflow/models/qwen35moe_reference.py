"""Golden Qwen3.5-MoE reference: the dense hybrid golden with the MoE tail.

Subclasses GoldenQwen35 (leaf handling, exact AdamW, per-kind attention
math) and swaps the dense SwiGLU tail for
``dataflow.tasks.modules.moe.reference.moe_mlp_reference`` — routed top-k
(topk_then_softmax at 35B-A3B) + ONE sigmoid-gated shared expert + the
gradient-injected load-balance aux. The autograd objective is CE + the
per-layer aux terms (f detached); the RETURNED/reported loss stays
CE-only. Untied embeddings only (the 35B config).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.tasks import ops
from dataflow.tasks.layouts import (
    PackedLayout,
    Qwen35MoeDims,
    qwen35moe_attn_weight_layout,
    qwen35moe_lin_weight_layout,
)
from dataflow.tasks.modules.moe.reference import moe_mlp_reference
from dataflow.models.llama3_reference import Leaves
from dataflow.models.qwen35_reference import GoldenQwen35


@dataclass
class GoldenQwen35Moe(GoldenQwen35):
    dims: Qwen35MoeDims  # re-typed; position and (lack of) default inherited

    def layer_layout(self, i: int) -> PackedLayout:
        d = self.dims
        build = (
            qwen35moe_attn_weight_layout if d.kinds[i] == "full"
            else qwen35moe_lin_weight_layout
        )
        return build(d, layer=i)

    # --- per-kind forwards: dense attention parts + the MoE tail ---------------
    # route_ids pins the discrete selection (block-ladder use only; see
    # moe_mlp_reference — E2E paths leave it None)

    def _moe_tail(self, xo, w, route_ids):
        h2 = ops.rmsnorm_reference(xo, w["ffn_norm_w"])
        return moe_mlp_reference(h2, w, self.dims.moe, xo, route_ids=route_ids)

    def lin_block_forward(
        self, x: torch.Tensor, w: Leaves, route_ids: torch.Tensor | None = None,
        segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        t = x.shape[0]
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qkvz = h1 @ w["w_qkvz"]
        ba = h1 @ w["w_ba"]
        conv_in = qkvz[:, : d.conv_dim]
        z = qkvz[:, d.conv_dim :].view(t, d.lin_v_heads, d.lin_v_head_dim)
        b = ba[:, : d.lin_v_heads]
        a = ba[:, d.lin_v_heads :]
        post = ops.causal_conv1d_silu_reference(conv_in, w["w_conv"], segments=seg)
        q = ops.l2norm_reference(post[:, : d.key_dim].reshape(t, d.lin_k_heads, d.lin_k_head_dim))
        k = ops.l2norm_reference(
            post[:, d.key_dim : 2 * d.key_dim].reshape(t, d.lin_k_heads, d.lin_k_head_dim)
        )
        v = post[:, 2 * d.key_dim :].reshape(t, d.lin_v_heads, d.lin_v_head_dim)
        beta = torch.sigmoid(b.float()).to(x.dtype)
        g = ops.gated_delta_gate_reference(a, w["A_log"], w["dt_bias"])
        core = ops.gated_delta_rule_reference(q, k, v, beta, g, segments=seg)
        o_normed = ops.gated_rmsnorm_reference(core, z, w["lin_norm_w"])
        xo = x + o_normed.reshape(t, d.value_dim) @ w["w_out"]
        return self._moe_tail(xo, w, route_ids)

    def full_block_forward(
        self, x: torch.Tensor, w: Leaves, route_ids: torch.Tensor | None = None,
        segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        t = x.shape[0]
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qg = h1 @ w["wq"]                       # (t, 2*attn_dim): [Q_all | gate_all]
        qm, gate = qg[:, : d.attn_dim], qg[:, d.attn_dim :]
        km = h1 @ w["wk"]
        v = h1 @ w["wv"]
        qn = ops.rmsnorm_reference(
            qm.view(t, d.n_heads, d.head_dim), w["q_norm_w"]
        ).view(t, d.attn_dim)
        kn = ops.rmsnorm_reference(
            km.view(t, d.n_kv_heads, d.head_dim), w["k_norm_w"]
        ).view(t, d.kv_dim)
        q = ops.partial_rope_reference(qn, seg, d.n_heads, d.head_dim, d.rot_dim, d.rope_base)
        k = ops.partial_rope_reference(kn, seg, d.n_kv_heads, d.head_dim, d.rot_dim, d.rope_base)
        attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, seg)
        gated = attn * torch.sigmoid(gate.float()).to(attn.dtype)
        xo = x + gated @ w["wo"]
        return self._moe_tail(xo, w, route_ids)

    # --- loss / training (CE reported; CE + aux differentiated) ----------------

    def loss_terms(
        self, tokens: torch.Tensor, targets: torch.Tensor, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        assert not self.tied, "qwen35moe is untied (the 35B config)"
        x = self.w_embed["w"][tokens.long()]
        seg = self._segments(segments, x.device)
        aux_total = torch.zeros((), dtype=torch.float32, device=x.device)
        for i in range(d.n_layers):
            w = self.w_blocks[i]
            x, aux = (
                self.full_block_forward(x, w, segments=seg) if d.kinds[i] == "full"
                else self.lin_block_forward(x, w, segments=seg)
            )
            aux_total = aux_total + aux
        hv = self.w_head
        logits = ops.rmsnorm_reference(x, hv["final_norm_w"]) @ hv["w"].T
        return ops.ce_loss_reference(logits, targets), aux_total

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> torch.Tensor:
        return self.loss_terms(tokens, targets, segments)[0]

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> float:
        for p in self.parameters():
            p.grad = None
        ce, aux_total = self.loss_terms(tokens, targets, segments)
        (ce + aux_total).backward()
        self.step_count += 1
        self._opt_obj("embed", "embed", self.w_embed)
        self._opt_obj("head", "head", self.w_head)
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", None, leaves)
        return float(ce.detach())
