"""Golden llama3 reference: plain eager torch + autograd, hand-written.

Every correctness gate compares the runtime against this model. It shares
the packed-weight layouts (so state is directly comparable buffer-to-buffer)
but is otherwise independent of the tasks layer's launch code: forward math
is composed from the ops' *reference* forms and autograd derives backward —
catching errors in our hand-written backward compositions.

The AdamW update replicates ops.adamw_step exactly, including the bf16
state round-trip, so post-step parameters are structurally comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataflow.tasks import ops
from dataflow.tasks.layouts import LlamaDims, PackedLayout, weight_layout
from dataflow.tasks.llama3_blocks import AdamWHyper


@dataclass
class GoldenLlama3:
    dims: LlamaDims
    n_layers: int
    hyper: AdamWHyper = field(default_factory=AdamWHyper)
    # flat bf16 parameter tensors (leaves), unpacked lazily per forward
    w_embed: torch.Tensor = None  # type: ignore[assignment]
    w_blocks: list[torch.Tensor] = field(default_factory=list)
    w_head: torch.Tensor = None  # type: ignore[assignment]
    step_count: int = 0
    _adam_m: dict[str, torch.Tensor] = field(default_factory=dict)
    _adam_v: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_packed_bytes(
        cls, dims: LlamaDims, n_layers: int,
        w_embed_bytes: torch.Tensor, w_block_bytes: list[torch.Tensor], w_head_bytes: torch.Tensor,
        hyper: AdamWHyper = AdamWHyper(),
    ) -> "GoldenLlama3":
        """Build from uint8 copies of the runtime's packed weight objects.

        Block leaves stay PACKED as flat bf16 tensors (the weight layout is
        all-bf16 with 256-byte-aligned offsets, so every field lands on an
        integral bf16 element offset); autograd then produces a packed grad
        directly comparable to the runtime's dW objects.
        """
        model = cls(dims=dims, n_layers=n_layers, hyper=hyper)
        model.w_embed = (
            w_embed_bytes.clone().view(torch.bfloat16).view(dims.vocab_size, dims.d_model)
            .cuda().requires_grad_()
        )
        model.w_head = (
            w_head_bytes.clone().view(torch.bfloat16).view(dims.vocab_size, dims.d_model)
            .cuda().requires_grad_()
        )
        model.w_blocks = [
            b.clone().view(torch.bfloat16).cuda().requires_grad_() for b in w_block_bytes
        ]
        return model

    def _block_views(self, flat_bf16: torch.Tensor) -> dict[str, torch.Tensor]:
        wl = weight_layout(self.dims)
        out: dict[str, torch.Tensor] = {}
        for f in wl.fields:
            n = 1
            for dim in f.shape:
                n *= int(dim)
            start = f.offset_bytes // 2
            out[f.name] = flat_bf16[start : start + n].view(f.shape)
        return out

    # --- forward --------------------------------------------------------------

    def block_forward(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.dims
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q = ops.rope_fwd(h1 @ w["wq"], d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        k = ops.rope_fwd(h1 @ w["wk"], d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.w_embed[tokens.long()]
        for flat in self.w_blocks:
            x = self.block_forward(x, self._block_views(flat))
        logits = ops.rmsnorm_noweight_reference(x) @ self.w_head.T
        return ops.ce_loss_reference(logits, targets)

    # --- training -------------------------------------------------------------

    def _adamw(self, name: str, w: torch.Tensor, g: torch.Tensor) -> None:
        hp = self.hyper
        if name not in self._adam_m:
            self._adam_m[name] = torch.zeros_like(w, dtype=torch.bfloat16)
            self._adam_v[name] = torch.zeros_like(w, dtype=torch.bfloat16)
        m, v = self._adam_m[name], self._adam_v[name]
        with torch.no_grad():
            ops.adamw_step(
                w.data.view(-1) if w.dtype == torch.bfloat16 else w.data,
                g.view(-1) if g.dtype == torch.bfloat16 else g,
                m.view(-1), v.view(-1),
                lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                weight_decay=hp.weight_decay, step=self.step_count,
            )

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        for p in self.parameters():
            p.grad = None
        loss = self.loss(tokens, targets)
        loss.backward()
        self.step_count += 1
        self._adamw("embed", self.w_embed, self.w_embed.grad.to(torch.bfloat16))
        for i, flat in enumerate(self.w_blocks):
            self._adamw(f"block_{i}", flat, flat.grad)
        self._adamw("head", self.w_head, self.w_head.grad.to(torch.bfloat16))
        return float(loss.detach())

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_embed, *self.w_blocks, self.w_head]

    def grads_packed(self) -> dict[str, torch.Tensor]:
        """Packed flat bf16 grads comparable to the runtime's dW objects."""
        out = {"dW_embed": self.w_embed.grad.to(torch.bfloat16).reshape(-1)}
        for i, flat in enumerate(self.w_blocks):
            out[f"dW_{i}"] = flat.grad.reshape(-1)
        out["dW_head"] = self.w_head.grad.to(torch.bfloat16).reshape(-1)
        return out
