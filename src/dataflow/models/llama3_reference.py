"""Golden llama3 reference: plain eager torch + autograd, hand-written.

Every correctness gate compares the runtime against this model. It shares
the packed-weight layouts (so state is directly comparable field-by-field)
but is otherwise independent of the tasks layer's launch code: forward math
is composed from the ops' *reference* forms and autograd derives backward —
catching errors in our hand-written backward compositions.

Parameters are PER-FIELD leaves at each field's storage dtype (the dims'
dtype policy — docs/notes/dtype-policy-design.md); the AdamW update
replicates ops.adamw_step exactly, including the storage-dtype round-trips
of the moments (opt dtype) and the gradient (grad dtype), so post-step
state is structurally comparable to the runtime's per-field optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataflow.tasks import ops
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
from dataflow.tasks.layouts import (
    LlamaDims,
    PackedLayout,
    embed_weight_layout,
    head_weight_layout,
    weight_layout,
)
from dataflow.tasks.llama3_blocks import AdamWHyper

Leaves = dict[str, torch.Tensor]


def unpack_leaves(layout: PackedLayout, raw_bytes: torch.Tensor) -> Leaves:
    """Typed CUDA leaf tensors (requires_grad) from a packed uint8 copy."""
    out: Leaves = {}
    for f in layout.fields:
        sl = raw_bytes[f.offset_bytes : f.offset_bytes + f.nbytes]
        t = sl.clone().view(TORCH_DTYPE_BY_NAME[f.dtype]).view(f.shape)
        out[f.name] = t.cuda().requires_grad_()
    return out


@dataclass
class GoldenLlama3:
    dims: LlamaDims
    n_layers: int
    hyper: AdamWHyper = field(default_factory=AdamWHyper)
    # per-field typed leaf tensors, keyed by layout field name
    w_embed: Leaves = None  # type: ignore[assignment]
    w_blocks: list[Leaves] = field(default_factory=list)
    w_head: Leaves = None  # type: ignore[assignment]
    step_count: int = 0
    _adam_m: dict[str, torch.Tensor] = field(default_factory=dict)
    _adam_v: dict[str, torch.Tensor] = field(default_factory=dict)

    # --- family layout hooks (overridden by subclasses) -----------------------

    def block_layout(self) -> PackedLayout:
        return weight_layout(self.dims)

    def embed_layout(self) -> PackedLayout:
        return embed_weight_layout(self.dims)

    def head_layout(self) -> PackedLayout:
        return head_weight_layout(self.dims)

    @classmethod
    def from_packed_bytes(
        cls, dims: LlamaDims, n_layers: int,
        w_embed_bytes: torch.Tensor, w_block_bytes: list[torch.Tensor], w_head_bytes: torch.Tensor,
        hyper: AdamWHyper = AdamWHyper(),
    ) -> "GoldenLlama3":
        """Build from uint8 copies of the runtime's packed weight objects."""
        model = cls(dims=dims, n_layers=n_layers, hyper=hyper)
        model.w_embed = unpack_leaves(model.embed_layout(), w_embed_bytes)
        model.w_head = unpack_leaves(model.head_layout(), w_head_bytes)
        bl = model.block_layout()
        model.w_blocks = [unpack_leaves(bl, b) for b in w_block_bytes]
        return model

    def final_leaves(self, object_id: str) -> tuple[PackedLayout, Leaves]:
        """(layout, leaves) for a runtime weight object id — the gate-side
        comparison unit (per-field, dtype-true, padding never compared)."""
        if object_id == "W_embed":
            return self.embed_layout(), self.w_embed
        if object_id == "W_head":
            return self.head_layout(), self.w_head
        return self.block_layout(), self.w_blocks[int(object_id.split("_")[1])]

    # --- forward --------------------------------------------------------------

    def block_forward(self, x: torch.Tensor, w: Leaves) -> torch.Tensor:
        d = self.dims
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        pos = ops.positions_for(d.seq_spec, x.shape[0], x.device)
        q = ops.rope_fwd(h1 @ w["wq"], pos, d.n_heads, d.head_dim, d.rope_base)
        k = ops.rope_fwd(h1 @ w["wk"], pos, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, d.seq_spec)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.w_embed["w"][tokens.long()]
        for w in self.w_blocks:
            x = self.block_forward(x, w)
        logits = ops.rmsnorm_reference(x, self.w_head["final_norm_w"]) @ self.w_head["w"].T
        return ops.ce_loss_reference(logits, targets)

    # --- training -------------------------------------------------------------

    def _field_dtypes(self, obj: str, name: str):
        ns = {"embed": "embed", "head": "head"}.get(obj)
        return self.dims.dtypes.for_field(f"{ns}.{name}" if ns else name)

    def _adamw_obj(self, obj: str, leaves: Leaves) -> None:
        """Per-field AdamW mirroring the runtime executor: the gradient
        rounds through its grad STORAGE dtype first, moments live at the
        opt dtype, fp32 math in ops.adamw_step."""
        hp = self.hyper
        for name, w in leaves.items():
            dts = self._field_dtypes(obj, name)
            key = f"{obj}.{name}"
            if key not in self._adam_m:
                odt = TORCH_DTYPE_BY_NAME[dts.opt]
                self._adam_m[key] = torch.zeros_like(w, dtype=odt)
                self._adam_v[key] = torch.zeros_like(w, dtype=odt)
            g = w.grad.to(TORCH_DTYPE_BY_NAME[dts.grad])
            with torch.no_grad():
                ops.adamw_step(
                    w.data.view(-1), g.view(-1),
                    self._adam_m[key].view(-1), self._adam_v[key].view(-1),
                    lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                    weight_decay=hp.weight_decay, step=self.step_count,
                )

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        for p in self.parameters():
            p.grad = None
        loss = self.loss(tokens, targets)
        loss.backward()
        self.step_count += 1
        self._adamw_obj("embed", self.w_embed)
        for i, leaves in enumerate(self.w_blocks):
            self._adamw_obj(f"block_{i}", leaves)
        self._adamw_obj("head", self.w_head)
        return float(loss.detach())

    def parameters(self) -> list[torch.Tensor]:
        out = list(self.w_embed.values())
        for leaves in self.w_blocks:
            out.extend(leaves.values())
        out.extend(self.w_head.values())
        return out
