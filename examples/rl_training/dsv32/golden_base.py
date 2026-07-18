"""Vendored isolated reference base for this example (formerly
dataflow.models.dsv32_reference): the example is self-contained client
code and carries its own ground-truth trainer; kept in sync by the
example's parity gate. The base chain it subclasses is vendored with it
(GoldenLlama3, GoldenOlmoe, GoldenDsv3 — formerly
dataflow.models.llama3_reference / olmoe_reference / dsv3_reference).

All goldens here are plain eager torch + autograd, hand-written: forward
math composed from the tasks ops' *reference* forms, autograd-derived
backward, and a per-field optimizer replica that reproduces the runtime's
storage-dtype round-trips exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataflow.tasks import ops
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
from dataflow.tasks.layouts import (
    Dsv3Dims,
    Dsv32Dims,
    LlamaDims,
    OlmoeDims,
    PackedLayout,
    dsv3_dense_weight_layout,
    dsv3_moe_weight_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
    embed_weight_layout,
    head_weight_layout,
    olmoe_weight_layout,
    weight_layout,
)
from dataflow.tasks.base_blocks import AdamWHyper
from dataflow.tasks.modules.dsa_reference import (
    dsa_attention_rows_reference,
    dsa_selection_mask_reference,
    dsa_index_scores_reference,
    dsa_indexer_kl_reference,
    dsa_sparse_attention_reference,
)
from dataflow.tasks.modules.mla_reference import (
    mla_attention_reference,
    mla_qkv_reference,
)
from dataflow.tasks.modules.moe.reference import moe_mlp_reference, moe_topk_reference

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
    """Golden llama3 reference: the base golden contract. Parameters are
    PER-FIELD leaves at each field's storage dtype (the dims' dtype
    policy); the AdamW update replicates ops.adamw_step exactly,
    including the storage-dtype round-trips of the moments (opt dtype)
    and the gradient (grad dtype), so post-step state is structurally
    comparable to the runtime's per-field optimizer."""

    dims: LlamaDims
    n_layers: int
    hyper: AdamWHyper = field(default_factory=AdamWHyper)
    # per-field typed leaf tensors, keyed by layout field name
    w_embed: Leaves = None  # type: ignore[assignment]
    w_blocks: list[Leaves] = field(default_factory=list)
    w_head: Leaves = None  # type: ignore[assignment]
    step_count: int = 0
    _opt_state: dict[str, dict] = field(default_factory=dict)

    # --- family layout hooks (overridden by subclasses) -----------------------

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        return weight_layout(self.dims, layer=layer)

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
        model.w_blocks = [
            unpack_leaves(model.block_layout(i), b) for i, b in enumerate(w_block_bytes)
        ]
        return model

    def final_leaves(self, object_id: str) -> tuple[PackedLayout, Leaves]:
        """(layout, leaves) for a runtime weight object id — the gate-side
        comparison unit (per-field, dtype-true, padding never compared)."""
        if object_id == "W_embed":
            return self.embed_layout(), self.w_embed
        if object_id == "W_head":
            return self.head_layout(), self.w_head
        i = int(object_id.split("_")[1])
        return self.block_layout(i), self.w_blocks[i]

    # --- forward --------------------------------------------------------------

    def _segments(self, segments, device) -> "ops.Segments":
        """This model's round segmentation (materialized on ``device``).
        Explicit ``segments`` (direct-invocation gates that hand the SAME
        Segments to the engine) win; otherwise derive from dims — one
        materialization per forward, read as fields thereafter."""
        if segments is not None:
            return segments
        return ops.Segments.of_dims(self.dims).on(device)

    def block_forward(self, x: torch.Tensor, w: Leaves, segments=None) -> torch.Tensor:
        d = self.dims
        seg = self._segments(segments, x.device)
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        pos = seg.positions
        q = ops.rope_fwd(h1 @ w["wq"], pos, d.n_heads, d.head_dim, d.rope_base)
        k = ops.rope_fwd(h1 @ w["wk"], pos, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn = ops.attention_reference(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, seg)
        h_mid = x + attn @ w["wo"]
        h2 = ops.rmsnorm_reference(h_mid, w["ffn_norm_w"])
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        return h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"]

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, segments=None) -> torch.Tensor:
        seg = self._segments(segments, self.w_embed["w"].device)
        x = self.w_embed["w"][tokens.long()]
        for w in self.w_blocks:
            x = self.block_forward(x, w, seg)
        logits = ops.rmsnorm_reference(x, self.w_head["final_norm_w"]) @ self.w_head["w"].T
        return ops.ce_loss_reference(logits, targets)

    # --- training -------------------------------------------------------------

    def _field_dtypes(self, obj: str, name: str):
        ns = {"embed": "embed", "head": "head"}.get(obj)
        layer = int(obj.split("_")[1]) if obj.startswith("block_") else None
        return self.dims.dtypes.for_field(f"{ns}.{name}" if ns else name, layer)

    def _opt_obj(self, obj: str, leaves: Leaves) -> None:
        """Per-field optimizer step mirroring the runtime executor's
        POLICY DISPATCH (name kept for subclass back-compat): each field
        resolves through dims.opt_policy exactly as AdamWStep.launch does
        — embed/head fields carry their ns-prefixed key ("embed.w"), so
        the muon recipe routes them to adamw by construction. Gradients
        round through their grad STORAGE dtype; slots live at the opt
        dtype (adamw m+v, muon m, sgd none, frozen nothing at all)."""
        from dataflow.tasks.optim import reference_field_step

        ns = obj if obj in ("embed", "head") else None
        layer = int(obj.split("_")[1]) if obj.startswith("block_") else None
        for name, w in leaves.items():
            dts = self._field_dtypes(obj, name)
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
        self._opt_obj("embed", self.w_embed)
        for i, leaves in enumerate(self.w_blocks):
            self._opt_obj(f"block_{i}", leaves)
        self._opt_obj("head", self.w_head)
        return float(loss.detach())

    def parameters(self) -> list[torch.Tensor]:
        out = list(self.w_embed.values())
        for leaves in self.w_blocks:
            out.extend(leaves.values())
        out.extend(self.w_head.values())
        return out


@dataclass
class GoldenOlmoe(GoldenLlama3):
    """Golden OLMoE reference. Same contract as GoldenLlama3 (reused for
    packed-leaf handling and the exact AdamW replica). Block =
    qwen3-shaped attention with FULL-ROW qk-norm (one RMSNorm over the
    whole q/k rows) + the MoE SwiGLU tail composed from
    ``dataflow.tasks.modules.moe.reference``.

    Aux load-balance loss: the AUTOGRAD OBJECTIVE is CE + sum of
    per-layer aux terms (f detached) — reproducing the runtime's
    gradient-injected analytic form exactly — while the RETURNED/reported
    loss stays CE-only (the runtime ``loss_*`` object is CE; parity gates
    compare that)."""

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


@dataclass
class GoldenDsv3(GoldenOlmoe):
    """Golden DeepSeek-V3 reference. Extends GoldenOlmoe (CE + aux
    objective, CE-only reported loss, AdamW replica) with: MLA attention
    (tasks mla_reference forms), MIXED depth (first_k_dense dense-SwiGLU
    layers, MoE rest — kind inferred per layer), sigmoid_noaux_tc routing
    with the UNGATED shared expert, V3's sequence-wise aux, and the
    balance-bias step rule applied EXACTLY like the runtime: counts
    captured from the forward's discrete assignments, bias updated after
    AdamW with b += speed * sign(mean - count) — the bias itself never
    sees AdamW math or autograd gradients."""

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


@dataclass
class GoldenDsv32(GoldenDsv3):
    """Golden DeepSeek-V3.2 reference. Extends GoldenDsv3 (mixed depth,
    noaux bias rule, AdamW replica) with DSA in every layer's attention:
    lightning-indexer scores on DETACHED inputs, top-k selection,
    mask-form sparse core, and the indexer KL term added to the per-block
    aux (objective = CE + sum(moe aux + L_I); CE-only reported — the
    runtime injects the same gradients analytically).

    Training-schedule note: the indexer trains at the shared AdamW lr
    (the paper uses a separate lr) — the runtime does the same, so parity
    gates compare like against like.

    TWO TRAINING MODES, one golden: SPARSE (top-k of the indexer's
    scores selects the attention live set; per-layer KL on gathered
    targets) and DENSE WARM-UP (sparse_mode=False: causal attention,
    full-prefix KL targets, mains frozen via dims.opt_policy — only the
    indexer steps). The mode branches live in
    dsa_selection_mask_reference and the optimizer policy, not in this
    class's control flow."""

    dims: Dsv32Dims  # re-typed

    def train_step(self, tokens, targets, segments=None) -> float:
        if getattr(self.dims, "sparse_mode", True):
            return super().train_step(tokens, targets, segments)
        # dense warm-up: the OBJECTIVE is the indexer KL alone — the
        # specialized program has no head, no CE, no dy chain. The
        # reported loss is the summed per-layer KL(p || sigma), exactly
        # the engine's loss_{s}_{r} accumulator. ``targets`` is unused.
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
        """Forward through the blocks (no head) summing each layer's
        KL — for dsv32 the training objective and the reported value
        coincide (every layer is its own group)."""
        x = self.w_embed["w"][tokens.long()]
        seg = self._segments(segments, x.device)
        total = None
        for w in self.w_blocks:
            x, kl = self.block_forward(x, w, segments=seg)
            total = kl if total is None else total + kl
        return total

    def block_layout(self, layer: int | None = None) -> PackedLayout:
        if layer is not None and self.dims.kinds[layer] == "dense":
            return dsv32_dense_weight_layout(self.dims, layer=layer)
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    def block_forward(
        self, x: torch.Tensor, w: dict[str, torch.Tensor],
        route_ids: torch.Tensor | None = None, segments=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        seg = self._segments(segments, x.device)
        t = x.shape[0]
        h, qk, v = d.n_heads, d.qk_head_dim, d.v_head_dim
        h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])
        q_lora, q_full, k_full, v_pad = mla_qkv_reference(h1, w, d, seg)

        # indexer scores (input DETACHED — the paper's seam: CE never
        # reaches the indexer) + the mode's mask (sparse top-k live set
        # vs dense warm-up causal — dsa_selection_mask_reference)
        scores = dsa_index_scores_reference(h1.detach(), q_lora.detach(), w, d, seg)
        mask = dsa_selection_mask_reference(scores, d, t, x.device, seg)
        train_idx = getattr(d, "train_indexer", True)

        qf = q_full.reshape(t, h * qk)
        kf = k_full.reshape(t, h * qk)
        vp = v_pad.reshape(t, h * qk)
        attn = dsa_sparse_attention_reference(qf, kf, vp, mask, d, seg)
        attn = attn.view(t, h, qk)[..., :v].reshape(t, h * v)
        h_mid = x + attn @ w["wo"]

        # per-layer indexer objective: KL(p || sigma) with p = this
        # layer's own attention rows on the mask's live set
        if train_idx:
            p = dsa_attention_rows_reference(q_full, k_full, mask, d, t, seg)
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
