"""A model family defined entirely OUTSIDE ``src/`` — the external-client
fixture for the plugin gate.

"toyfam" is a single-kind residual-MLP transformer-less family (rmsnorm →
w1 → silu → w2 → residual) that brings its own config type, dims object,
packed layouts, block executables, and resolver, while composing ONLY
public surfaces the way ``examples/`` do:

- ``lowering.shaped_program.build_shaped_program`` + ``LayerKindSpec``
  for the standard chain grammar (family-invariant task/object ids);
- ``lowering.emit`` helpers (``FamilyLayouts``/``LayerLayout``/
  ``apply_exact_sizes``/``object_size_factory``/``initial_values_from_layouts``)
  for packed-byte truth + seeded init;
- ``blocks.layouts`` (``PackedLayout``/``DTypePolicy``, the embed/head
  table layouts) and ``blocks.base_blocks`` templates (``EmbedFwd``/
  ``HeadLoss``/``EmbedBwd``/``OptimizerStep``) for everything that is
  genuinely family-neutral;
- the kernels registry (``resolve_kernels``) to pin op implementations.

Importing this module registers the family
(``model_families.families.register_family``), which is how the tools'
``--plugin`` flag and the ``dataflow.families`` packaging entry point
load external families. CPU-safe: torch is only touched inside
``launch``/init bodies, never at import.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_training.blocks.layouts import (
    DTypePolicy,
    PackedLayout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
)
from dataflow_training.lowering.emit import (
    FamilyLayouts,
    LayerLayout,
    apply_exact_sizes,
    initial_values_from_layouts,
    object_size_factory,
)
from dataflow_training.lowering.shaped_program import (
    LayerKindSpec,
    build_shaped_program,
)
from dataflow_training.model_families.families import ModelFamily, register_family


@dataclass(frozen=True)
class ToyConfig:
    """Shaped config obeying the ``build_shaped_program`` protocol.

    ``n_heads``/``n_kv_heads`` are vestigial for this attention-free
    block — the generic builder stamps them into program metadata
    unconditionally, so every family must carry them."""

    n_layers: int = 1
    d_model: int = 64
    d_ff: int = 128
    n_heads: int = 1
    n_kv_heads: int = 1
    vocab_size: int = 256
    seq_len: int = 32
    batch: int = 1
    grad_accum_rounds: int = 2
    num_steps: int = 1
    optimizer_placement: str = "interleaved"
    opt_policy: object = "adamw"

    @property
    def max_tokens(self) -> int:
        return self.seq_len * self.batch

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def block_params(self) -> int:
        return self.d_model + 2 * self.d_model * self.d_ff

    @classmethod
    def tiny(cls) -> "ToyConfig":
        return cls()


@dataclass(frozen=True)
class ToyDims:
    """The family's opaque dims object. Field names shared with the
    builtin dims (``d_model``/``vocab_size``/``max_tokens``/``dtypes``/
    ``opt_policy``) are what the reused base templates and emit helpers
    duck-type on."""

    d_model: int
    d_ff: int
    vocab_size: int
    max_tokens: int
    seq_len: int
    dtypes: DTypePolicy = DTypePolicy()
    opt_policy: object = "adamw"
    seq_lens: tuple[int, ...] | None = None


def toy_dims(cfg: ToyConfig) -> ToyDims:
    return ToyDims(d_model=cfg.d_model, d_ff=cfg.d_ff,
                   vocab_size=cfg.vocab_size, max_tokens=cfg.max_tokens,
                   seq_len=cfg.seq_len, opt_policy=cfg.opt_policy)


def toy_weight_layout(dims: ToyDims) -> PackedLayout:
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build([
        ("ffn_norm_w", (d,), "bf16"),
        ("w1", (ff, d), "bf16"),
        ("w2", (d, ff), "bf16"),
    ])


def toy_activation_layout(dims: ToyDims) -> PackedLayout:
    """Saved-for-backward context of one toy block: the norm rstd, the
    normed input, and the pre-activation."""
    t, d, ff = dims.max_tokens, dims.d_model, dims.d_ff
    return PackedLayout.build([
        ("rstd", (t,), "fp32"),
        ("xn", (t, d), "bf16"),
        ("u", (t, ff), "bf16"),
    ])


def toy_layouts(dims: ToyDims) -> FamilyLayouts:
    layers = [LayerLayout(kind="block",
                          weights=toy_weight_layout(dims),
                          activations=toy_activation_layout(dims))]
    return FamilyLayouts(layers=layers,
                         embed=embed_weight_layout(dims),
                         head=head_weight_layout(dims))


def toy_kind_spec(dims: ToyDims) -> LayerKindSpec:
    wl = toy_weight_layout(dims)
    al = toy_activation_layout(dims)
    return LayerKindSpec(
        key_prefix="toy_block",
        w_bytes=wl.total_bytes,
        a_bytes=al.total_bytes,
        fwd_us=10.0, bwd_us=20.0, recompute_us=10.0, optimizer_us=5.0,
        fwd_subops=[], bwd_subops=[], recompute_subops=[],
        optimizer_subops=[],
    )


def lower_toy(cfg: ToyConfig, recompute_levels=None):
    dims = toy_dims(cfg)
    shaped = build_shaped_program(
        cfg,
        kinds={"block": toy_kind_spec(dims)},
        family="toyfam",
        recompute_levels=recompute_levels,
    )
    return apply_exact_sizes(shaped, "toyfam-exact",
                             object_size=object_size_factory(dims, toy_layouts(dims)))


def toy_initial_values(program, cfg: ToyConfig, backend, *, seed: int = 0,
                       into=None):
    dims = toy_dims(cfg)
    return initial_values_from_layouts(program, dims, toy_layouts(dims),
                                       backend, seed=seed, into=into)


def toy_rmsnorm(x, norm_w, eps: float = 1e-6):
    import torch

    rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
    xn = (x.float() * rstd[:, None] * norm_w.float()).to(x.dtype)
    return xn, rstd


@dataclass(frozen=True)
class ToyBlockFwd:
    """toy_block_fwd: y = x + silu(rmsnorm(x) @ w1.T) @ w2.T, saving
    (rstd, xn, u) for the backward when the A object is materialized."""

    dims: ToyDims
    kernels: object = None

    def launch(self, ctx) -> None:
        import torch

        from dataflow.runtime.interop import torch_view

        d = self.dims
        x = torch_view(ctx.inputs[ctx.task.inputs[0]],
                       (d.max_tokens, d.d_model), torch.bfloat16)
        w = toy_weight_layout(d).views(ctx.inputs[ctx.task.inputs[1]])
        y = torch_view(ctx.outputs[ctx.task.outputs[0].id],
                       (d.max_tokens, d.d_model), torch.bfloat16)
        xn, rstd = toy_rmsnorm(x, w["ffn_norm_w"])
        u = xn @ w["w1"].T
        y.copy_(x + torch.nn.functional.silu(u) @ w["w2"].T)
        if len(ctx.task.outputs) > 1:  # A saved (recompute level 0)
            a = toy_activation_layout(d).views(
                ctx.outputs[ctx.task.outputs[1].id])
            a["rstd"].copy_(rstd)
            a["xn"].copy_(xn)
            a["u"].copy_(u)


@dataclass(frozen=True)
class ToyBlockRecompute:
    """toy_block_recompute: repopulate the A object from (x, W)."""

    dims: ToyDims
    kernels: object = None

    def launch(self, ctx) -> None:
        import torch

        from dataflow.runtime.interop import torch_view

        d = self.dims
        x = torch_view(ctx.inputs[ctx.task.inputs[0]],
                       (d.max_tokens, d.d_model), torch.bfloat16)
        w = toy_weight_layout(d).views(ctx.inputs[ctx.task.inputs[1]])
        a = toy_activation_layout(d).views(
            ctx.outputs[ctx.task.outputs[0].id])
        xn, rstd = toy_rmsnorm(x, w["ffn_norm_w"])
        a["rstd"].copy_(rstd)
        a["xn"].copy_(xn)
        a["u"].copy_(xn @ w["w1"].T)


@dataclass(frozen=True)
class ToyBlockBwd:
    """toy_block_bwd: inputs (dy, A, x, W [, dW accum]); outputs
    (dx [, dW on round 0]) with the create-vs-accumulate convention."""

    dims: ToyDims
    kernels: object = None

    def launch(self, ctx) -> None:
        import torch

        from dataflow.runtime.interop import torch_view

        d = self.dims
        dy = torch_view(ctx.inputs[ctx.task.inputs[0]],
                        (d.max_tokens, d.d_model), torch.bfloat16)
        a = toy_activation_layout(d).views(ctx.inputs[ctx.task.inputs[1]])
        w = toy_weight_layout(d).views(ctx.inputs[ctx.task.inputs[3]])
        dx = torch_view(ctx.outputs[ctx.task.outputs[0].id],
                        (d.max_tokens, d.d_model), torch.bfloat16)
        accum = bool(ctx.task.mutates)
        if accum:
            gbuf = ctx.mutates[ctx.task.mutates[0]]
        else:
            gbuf = ctx.outputs[ctx.task.outputs[1].id]
        g = grad_layout(toy_weight_layout(d), d.dtypes).views(gbuf)
        u, xn, rstd = a["u"].float(), a["xn"].float(), a["rstd"]
        h = torch.nn.functional.silu(u)
        dh = dy.float() @ w["w2"].float()
        sig = torch.sigmoid(u)
        du = dh * (sig * (1.0 + u * (1.0 - sig)))
        dw2 = dy.float().T @ h
        dw1 = du.T @ xn
        dxn = du @ w["w1"].float()
        xhat = xn / w["ffn_norm_w"].float()
        dnorm = (dxn * xhat).sum(dim=0)
        dxhat = dxn * w["ffn_norm_w"].float()
        dx_norm = rstd[:, None] * (
            dxhat - xhat * (dxhat * xhat).mean(dim=-1, keepdim=True))
        if accum:
            g["w1"].add_(dw1.to(g["w1"].dtype))
            g["w2"].add_(dw2.to(g["w2"].dtype))
            g["ffn_norm_w"].add_(dnorm.to(g["ffn_norm_w"].dtype))
        else:
            g["w1"].copy_(dw1.to(g["w1"].dtype))
            g["w2"].copy_(dw2.to(g["w2"].dtype))
            g["ffn_norm_w"].copy_(dnorm.to(g["ffn_norm_w"].dtype))
        dx.copy_((dy.float() + dx_norm).to(dx.dtype))


def toy_optimizer_layout(dims, task, w_size: int):
    """``OptimizerStep`` layout hook: every toy block optimizer views its
    W/dW/O through the family's own packed layout."""
    return toy_weight_layout(dims), None


class ToyResolver:
    """compute_block_key -> executable; loud on unknown keys so a
    lowering/resolver drift fails at resolution, not at launch."""

    def __init__(self, table: dict):
        self.table = table

    def __call__(self, task):
        key = task.compute_block_key
        if key not in self.table:
            raise KeyError(
                f"no toyfam executable for compute_block_key {key!r} "
                f"(task {task.id!r})")
        return self.table[key]


def build_toy_resolver(dims: ToyDims, hyper=None):
    from dataflow_training.blocks.base_blocks import (
        AdamWHyper,
        EmbedBwd,
        EmbedFwd,
        HeadLoss,
        OptimizerStep,
    RoundPrologue,
)
    from dataflow_training.kernels import resolve_kernels

    kernels = resolve_kernels()
    hyper = hyper if hyper is not None else AdamWHyper()
    table = {
        # the round prologue is UNIVERSAL: every family (external ones
        # included) opens each round with it — it publishes the round
        # index + content token count and materializes Segments
        "prologue_round": RoundPrologue(dims, kernels),
        "embed_fwd": EmbedFwd(dims, kernels),
        "toy_block_fwd": ToyBlockFwd(dims, kernels),
        "toy_block_recompute": ToyBlockRecompute(dims, kernels),
        "toy_block_bwd": ToyBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": OptimizerStep(dims, kernels, hyper,
                                         resolve_layout=toy_optimizer_layout),
        "optimizer_embed": OptimizerStep(dims, kernels, hyper, kind="embed"),
        "optimizer_head": OptimizerStep(dims, kernels, hyper, kind="head"),
    }
    return ToyResolver(table)


def toy_family() -> ModelFamily:
    return ModelFamily(
        name="toyfam",
        config_type=ToyConfig,
        derive_dims=toy_dims,
        lower=lower_toy,
        initial_values=toy_initial_values,
        build_resolver=build_toy_resolver,
    )


register_family("toyfam", toy_family)
