"""Golden Qwen3.5-dense reference: plain eager torch + autograd.

Standalone (the hybrid + tied-embedding structure doesn't fit the llama
golden's three-leaf shape): ONE packed embed/head leaf ``[table |
final_norm_w]`` plus one packed leaf per layer, per-kind block forwards
composed EXCLUSIVELY from the pinned reference ops (tasks/ops.py — the same
functions the kernel-contract tests validate against fla), and the exact
AdamW replica. The sequential delta-rule recurrence makes this golden slow
and obviously correct; use tiny configs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from dataflow.tasks import ops
from dataflow.tasks.layouts import (
    Qwen35Dims,
    head_weight_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_weight_layout,
)
from dataflow.tasks.llama3_blocks import AdamWHyper


def _views(layout, flat_bf16: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in layout.fields:
        n = int(math.prod(f.shape))
        start = f.offset_bytes // 2
        out[f.name] = flat_bf16[start : start + n].view(f.shape)
    return out


@dataclass
class GoldenQwen35:
    dims: Qwen35Dims
    hyper: AdamWHyper = field(default_factory=AdamWHyper)
    w_embed: torch.Tensor = None  # packed flat bf16 [table | final_norm_w]
    w_blocks: list[torch.Tensor] = field(default_factory=list)
    step_count: int = 0
    _adam_m: dict[str, torch.Tensor] = field(default_factory=dict)
    _adam_v: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_packed_bytes(
        cls, dims: Qwen35Dims, n_layers: int,
        w_embed_bytes: torch.Tensor, w_block_bytes: list[torch.Tensor],
        hyper: AdamWHyper = AdamWHyper(),
    ) -> "GoldenQwen35":
        assert n_layers == dims.n_layers == len(w_block_bytes)
        model = cls(dims=dims, hyper=hyper)
        model.w_embed = w_embed_bytes.clone().view(torch.bfloat16).cuda().requires_grad_()
        model.w_blocks = [
            b.clone().view(torch.bfloat16).cuda().requires_grad_() for b in w_block_bytes
        ]
        return model

    def _layer_views(self, i: int) -> dict[str, torch.Tensor]:
        d = self.dims
        layout = (
            qwen35_attn_weight_layout(d) if d.kind_of(i) == "full"
            else qwen35_lin_weight_layout(d)
        )
        return _views(layout, self.w_blocks[i])

    # --- per-kind block forwards (pinned reference ops only) -------------------

    def lin_block_forward(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.dims
        t = x.shape[0]
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        qkvz = h1 @ w["w_qkvz"]
        ba = h1 @ w["w_ba"]
        conv_in = qkvz[:, : d.conv_dim]
        z = qkvz[:, d.conv_dim :].view(t, d.num_v_heads, d.head_v_dim)
        b = ba[:, : d.num_v_heads]
        a = ba[:, d.num_v_heads :]
        post = ops.causal_conv1d_silu_reference(conv_in, w["w_conv"])
        q = ops.l2norm_reference(post[:, : d.key_dim].reshape(t, d.num_k_heads, d.head_k_dim))
        k = ops.l2norm_reference(
            post[:, d.key_dim : 2 * d.key_dim].reshape(t, d.num_k_heads, d.head_k_dim)
        )
        v = post[:, 2 * d.key_dim :].reshape(t, d.num_v_heads, d.head_v_dim)
        beta = torch.sigmoid(b.float()).to(x.dtype)
        g = ops.gated_delta_gate_reference(a, w["A_log"], w["dt_bias"])
        core = ops.gated_delta_rule_reference(q, k, v, beta, g)
        o_normed = ops.gated_rmsnorm_reference(core, z, w["lin_norm_w"])
        xo = x + o_normed.reshape(t, d.value_dim) @ w["w_out"]
        h2 = ops.rmsnorm_reference(xo, w["ffn_norm_w"])
        return xo + ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"]) @ w["w2"]

    def full_block_forward(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.dims
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
        q = ops.partial_rope_reference(qn, d.seq_len, d.n_heads, d.head_dim, d.rot_dim, d.rope_base)
        k = ops.partial_rope_reference(kn, d.seq_len, d.n_kv_heads, d.head_dim, d.rot_dim, d.rope_base)
        attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, d.seq_len)
        gated = attn * torch.sigmoid(gate.float()).to(attn.dtype)
        xo = x + gated @ w["wo"]
        h2 = ops.rmsnorm_reference(xo, w["ffn_norm_w"])
        return xo + ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"]) @ w["w2"]

    # --- loss / training --------------------------------------------------------

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dims
        hv = _views(head_weight_layout(d), self.w_embed)
        x = hv["w"][tokens.long()]
        for i in range(d.n_layers):
            w = self._layer_views(i)
            x = (
                self.full_block_forward(x, w) if d.kind_of(i) == "full"
                else self.lin_block_forward(x, w)
            )
        logits = ops.rmsnorm_reference(x, hv["final_norm_w"]) @ hv["w"].T
        return ops.ce_loss_reference(logits, targets)

    def _adamw(self, name: str, w: torch.Tensor, g: torch.Tensor) -> None:
        hp = self.hyper
        if name not in self._adam_m:
            self._adam_m[name] = torch.zeros_like(w, dtype=torch.bfloat16)
            self._adam_v[name] = torch.zeros_like(w, dtype=torch.bfloat16)
        m, v = self._adam_m[name], self._adam_v[name]
        with torch.no_grad():
            ops.adamw_step(
                w.data.view(-1), g.view(-1), m.view(-1), v.view(-1),
                lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                weight_decay=hp.weight_decay, step=self.step_count,
            )

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        for p in self.parameters():
            p.grad = None
        loss = self.loss(tokens, targets)
        loss.backward()
        self.step_count += 1
        self._adamw("embed", self.w_embed, self.w_embed.grad)
        for i, flat in enumerate(self.w_blocks):
            self._adamw(f"block_{i}", flat, flat.grad)
        return float(loss.detach())

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_embed, *self.w_blocks]

    def grads_packed(self) -> dict[str, torch.Tensor]:
        out = {"dW_embed": self.w_embed.grad.reshape(-1)}
        for i, flat in enumerate(self.w_blocks):
            out[f"dW_{i}"] = flat.grad.reshape(-1)
        return out
