"""Golden Qwen3.5-dense reference: plain eager torch + autograd.

Standalone (the hybrid structure doesn't fit the llama golden): per-kind
block forwards composed EXCLUSIVELY from the pinned reference ops
(tasks/ops.py — the same functions the kernel-contract tests validate
against fla) and the exact AdamW replica. Embeddings follow the config:
UNTIED (the 9B) = bare-table w_embed + packed ``[table | final_norm_w]``
w_head leaves; TIED (2B-style, ``w_head=None``) = the head layout rides
the single w_embed object (policy-addressed as head.*). Parameters are
PER-FIELD leaves at the dims' dtype-policy storage dtypes. The sequential
delta-rule recurrence makes this golden slow and obviously correct; use
tiny configs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataflow.tasks import ops
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
from dataflow.tasks.layouts import (
    PackedLayout,
    Qwen35Dims,
    embed_weight_layout,
    head_weight_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_weight_layout,
)
from dataflow.tasks.base_blocks import AdamWHyper
from dataflow.models.llama3_reference import Leaves, unpack_leaves


@dataclass
class GoldenQwen35:
    dims: Qwen35Dims
    hyper: AdamWHyper = field(default_factory=AdamWHyper)
    # per-field typed leaves; tied: w_embed carries [w | final_norm_w]
    w_embed: Leaves = None  # type: ignore[assignment]
    w_head: Leaves = None   # untied only
    w_blocks: list[Leaves] = field(default_factory=list)
    step_count: int = 0
    _opt_state: dict[str, dict] = field(default_factory=dict)

    @property
    def tied(self) -> bool:
        return self.w_head is None

    def layer_layout(self, i: int) -> PackedLayout:
        d = self.dims
        build = (
            qwen35_attn_weight_layout if d.kind_of(i) == "full"
            else qwen35_lin_weight_layout
        )
        return build(d, layer=i)

    def embed_layout(self) -> PackedLayout:
        # tied packs the head layout into W_embed
        return head_weight_layout(self.dims) if self.tied else embed_weight_layout(self.dims)

    @classmethod
    def from_packed_bytes(
        cls, dims: Qwen35Dims, n_layers: int,
        w_embed_bytes: torch.Tensor, w_block_bytes: list[torch.Tensor],
        w_head_bytes: torch.Tensor | None = None,
        hyper: AdamWHyper = AdamWHyper(),
    ) -> "GoldenQwen35":
        assert n_layers == dims.n_layers == len(w_block_bytes)
        model = cls(dims=dims, hyper=hyper)
        if w_head_bytes is not None:
            model.w_head = unpack_leaves(head_weight_layout(dims), w_head_bytes)
        model.w_embed = unpack_leaves(model.embed_layout(), w_embed_bytes)
        model.w_blocks = [
            unpack_leaves(model.layer_layout(i), b) for i, b in enumerate(w_block_bytes)
        ]
        return model

    def final_leaves(self, object_id: str) -> tuple[PackedLayout, Leaves]:
        if object_id == "W_embed":
            return self.embed_layout(), self.w_embed
        if object_id == "W_head":
            return head_weight_layout(self.dims), self.w_head
        i = int(object_id.split("_")[1])
        return self.layer_layout(i), self.w_blocks[i]

    # --- per-kind block forwards (pinned reference ops only) -------------------

    def _segments(self, segments, device) -> "ops.Segments":
        """This model's round segmentation (materialized on ``device``);
        explicit ``segments`` win (the direct gates share the engine's), else
        derive from dims — one materialization per forward, read as fields."""
        if segments is not None:
            return segments
        return ops.Segments.of_dims(self.dims).on(device)

    def lin_block_forward(self, x: torch.Tensor, w: Leaves, segments=None) -> torch.Tensor:
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
        h2 = ops.rmsnorm_reference(xo, w["ffn_norm_w"])
        return xo + ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"]) @ w["w2"]

    def full_block_forward(self, x: torch.Tensor, w: Leaves, segments=None) -> torch.Tensor:
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
        h2 = ops.rmsnorm_reference(xo, w["ffn_norm_w"])
        return xo + ops.swiglu_fwd(h2 @ w["w1"], h2 @ w["w3"]) @ w["w2"]

    # --- loss / training --------------------------------------------------------

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> torch.Tensor:
        d = self.dims
        hv = self.w_embed if self.tied else self.w_head
        table = hv["w"] if self.tied else self.w_embed["w"]
        x = table[tokens.long()]
        seg = self._segments(segments, x.device)
        for i in range(d.n_layers):
            w = self.w_blocks[i]
            x = (
                self.full_block_forward(x, w, seg) if d.kind_of(i) == "full"
                else self.lin_block_forward(x, w, seg)
            )
        logits = ops.rmsnorm_reference(x, hv["final_norm_w"]) @ hv["w"].T
        return ops.ce_loss_reference(logits, targets)

    def _field_dtypes(self, ns: str | None, name: str, layer: int | None = None):
        return self.dims.dtypes.for_field(f"{ns}.{name}" if ns else name, layer)

    def _opt_obj(self, obj: str, ns: str | None, leaves: Leaves) -> None:
        from dataflow.tasks.optim import reference_field_step

        layer = int(obj.split("_")[1]) if obj.startswith("block_") else None
        for name, w in leaves.items():
            dts = self._field_dtypes(obj, ns, name)
            key = f"{obj}.{name}"
            reference_field_step(
                self.dims, self.hyper, ns=ns, layer=layer, name=name, w=w,
                state=self._opt_state.setdefault(key, {}),
                step=self.step_count,
                grad_dtype=TORCH_DTYPE_BY_NAME[dts.grad],
                opt_dtype=TORCH_DTYPE_BY_NAME[dts.opt],
            )
    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> float:
        for p in self.parameters():
            p.grad = None
        loss = self.loss(tokens, targets, segments)
        loss.backward()
        self.step_count += 1
        # tied W_embed IS the head layout — policy-addressed as head.*
        self._opt_obj("embed", "head" if self.tied else "embed", self.w_embed)
        if self.w_head is not None:
            self._opt_obj("head", "head", self.w_head)
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", None, leaves)
        return float(loss.detach())

    def parameters(self) -> list[torch.Tensor]:
        out = list(self.w_embed.values())
        if self.w_head is not None:
            out.extend(self.w_head.values())
        for leaves in self.w_blocks:
            out.extend(leaves.values())
        return out
