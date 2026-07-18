"""Llama3 family: config + declarations over the generic machinery.

One module per family, one contract for every family (docs/extending.md
§4/§6): the Shaped*Config (shapes + the param-count surface the roofline
seeds read), the kind spec(s) for ``shaped_program.build_shaped_program``,
the config->dims mapping, and the ``FamilyLayouts`` declaration consumed by
the generic lowering (``training.lowering``). Llama3 is one dense-
transformer kind; nothing here is special.

Naming conventions of the emitted chain (family-invariant, owned by the
generic builder):

    tokens_{s}_{r}, targets_{s}_{r}      int32 token/target ids (initial, backing)
    W_embed, W_{i}, W_head               parameters (initial, backing)
    O_embed, O_{i}, O_head               AdamW state, 2x params (initial, backing)
    y_embed_{s}_{r}, y_{s}_{r}_{i}       block outputs (activations)
    A_{s}_{r}_{i}                        saved backward context per block
    dy_*                                 activation gradients (logits/dlogits
                                         DO NOT EXIST — head_loss is fused
                                         + token-chunked)
    dW_embed_{s}, dW_{s}_{i}, dW_head_{s}  parameter gradients (accumulated over rounds)
    loss_{s}_{r}                         scalar loss (final: backing)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow_training.blocks.layouts import (
    DTypePolicy,
    LlamaDims,
    activation_layout,
    embed_weight_layout,
    head_weight_layout,
    sliced_layout,
    weight_layout,
)
from ..lowering.emit import FamilyLayouts, LayerLayout, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from ..lowering.shaped_program import ShapedHardware, build_shaped_program, roofline_block_kind_spec


@dataclass(frozen=True)
class ShapedLlamaConfig:
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    vocab_size: int = 128_256
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    # "interleaved" (default): each optimizer task is emitted immediately
    # after the LAST mutation of its gradient (inside the final grad-accum
    # round's backward), so its state streaming (O in, W+O writebacks out)
    # overlaps the remaining backward compute. "tail" restores the legacy
    # all-optimizers-after-all-rounds order, whose transfers drain into a
    # GPU-idle PCIe phase at the end of every step (measured 1.5-2.0 s at
    # 8B/seq-1K). Task ids are identical in both modes; only order changes.
    optimizer_placement: str = "interleaved"
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"
    # per-field dtype policy for params/grads/opt state (default: all bf16,
    # the historical convention)
    dtypes: DTypePolicy = DTypePolicy()
    # ragged packing: explicit per-round sequence lengths (sum = tokens
    # per round); None = uniform batch x seq_len
    seq_lens: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # -- parameter counts ----------------------------------------------------
    @property
    def block_params(self) -> int:
        d, kv, ff = self.d_model, self.kv_dim, self.d_ff
        attn = d * d + d * kv + d * kv + d * d  # Wq, Wk, Wv, Wo
        mlp = 3 * d * ff                        # W1 (gate), W3 (up), W2 (down)
        norms = 2 * d
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    # -- activation widths (elements per token) --------------------------------
    @property
    def saved_ctx_width(self) -> int:
        """Elements per token saved by a block forward for its backward.

        Roughly: block input (d), attention normed input (d), q (d),
        k/v (2*kv), attention output (d), softmax lse (heads), mlp normed
        input (d), gate/up projections (2*ff). Matches the scale of real
        save-all footprints for this architecture.
        """
        d, kv, ff, h = self.d_model, self.kv_dim, self.d_ff, self.n_heads
        return 5 * d + 2 * kv + h + 2 * ff

    @classmethod
    def tiny(cls) -> "ShapedLlamaConfig":
        return cls(
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=160,
            vocab_size=512,
            seq_len=64,
            batch=1,
        )

    @classmethod
    def llama3_8b(cls, *, seq_len: int = 4096, batch: int = 1, grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedLlamaConfig":
        return cls(seq_len=seq_len, batch=batch, grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def build_shaped_llama3(
    cfg: ShapedLlamaConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
) -> Program:
    """Llama3 through the generic builder: one dense-transformer kind."""
    hw = hw or ShapedHardware()
    from ..lowering.freeze_plan import derive_freeze_plan

    d = dims_of(cfg)
    plan = derive_freeze_plan(
        d, cfg.n_layers,
        lambda i: [f.name for f in weight_layout(d, layer=i).fields],
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)),
    )
    return build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        freeze=plan,
    )


def dims_of(cfg: ShapedLlamaConfig) -> LlamaDims:
    return LlamaDims(
        opt_policy=cfg.opt_policy,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens,
        seq_len=cfg.seq_len,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


# which saved activations narrow alongside a tensor-parallel weight
# slice: x1/x3 are the w1/w3 up-projections, so they share the d_ff
# column range (the same (dim, lo, hi) tuple — both slice dim 1)
TP_ACTIVATION_OF_WEIGHT = {"w1": "x1", "w3": "x3"}


def family_layouts(cfg: ShapedLlamaConfig, tp_view: dict | None = None
                   ) -> tuple[LlamaDims, FamilyLayouts]:
    """``tp_view`` ({root: {field: (dim, lo, hi)}}, from
    sharding.tp_view) makes this a PER-RANK view: listed weight
    fields — and the saved activations tied to them — materialize at
    shard shape. Grad/opt layouts, exact sizes, and init offsets all
    derive from these layouts, so the transform lands everywhere by
    construction."""
    dims = dims_of(cfg)
    cl = activation_layout(dims)
    view = tp_view or {}
    layers = []
    for i in range(cfg.n_layers):
        wl = weight_layout(dims, layer=i)
        cli = cl
        slices = view.get(f"W_{i}")
        if slices:
            wl = sliced_layout(wl, slices)
            act_slices = {}
            for wf, af in TP_ACTIVATION_OF_WEIGHT.items():
                if wf in slices:
                    act_slices[af] = slices[wf]
            if act_slices:
                cli = sliced_layout(cl, act_slices)
        layers.append(LayerLayout(kind="block", weights=wl,
                                  activations=cli))
    return dims, FamilyLayouts(
        layers=layers,
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
    )


def lower_llama3(
    cfg: ShapedLlamaConfig,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_llama3(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "llama3-exact", size_of=size_of_factory(dims, fl))


def tp_fill_slices(cfg: ShapedLlamaConfig, tp_view: dict) -> dict:
    """{root: {field: (dim, lo, hi, full_shape)}} for the init fill:
    tp-sharded fields DRAW at the full single-GPU shape (keeping the
    generator stream byte-aligned with a plain run) and write only
    the rank's slice."""
    dims = dims_of(cfg)
    out: dict = {}
    for root, slices in tp_view.items():
        layer = int(root.split("_")[1])
        full = {f.name: f.shape
                for f in weight_layout(dims, layer=layer).fields}
        per: dict = {}
        for name, sl in slices.items():
            dim, lo, hi = (int(x) for x in sl)
            per[name] = (dim, lo, hi, full[name])
        out[root] = per
    return out


def initial_values(program: Program, cfg: ShapedLlamaConfig, backend, *,
                   seed: int = 0, into=None, tp_view: dict | None = None):
    dims, fl = family_layouts(cfg, tp_view=tp_view)
    slices_by_root = tp_fill_slices(cfg, tp_view) if tp_view else None
    return initial_values_from_layouts(program, dims, fl, backend,
                                       seed=seed, into=into,
                                       tp_slices_by_root=slices_by_root)
