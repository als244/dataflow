"""Qwen3.5-MoE block executables: the dense hybrid's attention parts
VERBATIM (subclassed from qwen35_blocks — the dense-tail template split's payoff),
with the dense SwiGLU tail swapped for the pluggable MoE module.

Forward: the dense kinds' stage prefixes (through ``ffn_norm``) +
``MOE_SHARED_STAGES`` (this family carries ONE sigmoid-gated shared
expert). Backward: ONLY ``_mlp_bwd`` is overridden — the DeltaNet and
gated-attention backwards are inherited untouched. Distinct compute keys
(``linmoe_*`` / ``gattnmoe_*``) keep profiles/caches per-family.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.core import TaskSpec

from ..kernels import KernelSet, resolve_kernels
from ..layouts import (
    PackedLayout,
    Qwen35MoeDims,
    embed_weight_layout,
    head_weight_layout,
    qwen35moe_attn_activation_layout,
    qwen35moe_attn_weight_layout,
    qwen35moe_lin_activation_layout,
    qwen35moe_lin_weight_layout,
)
from ..base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss, RoundPrologue
from ..modules.moe.stages import MOE_SHARED_STAGES, MoEAuxTempState, MoEProfileFill, moe_mlp_tail_bwd
from .qwen35_blocks import (
    Qwen35AttnBlockBwd,
    Qwen35AttnBlockFwd,
    Qwen35LinBlockBwd,
    Qwen35LinBlockFwd,
    Qwen35LinBlockRecompute,
)


def _stage_prefix(stages, upto: str):
    names = [s[0] for s in stages]
    return stages[: names.index(upto) + 1]


# ---------------------------------------------------------------------------
# Gated DeltaNet kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35MoeLinBlockFwd(MoEAuxTempState, MoEProfileFill, Qwen35LinBlockFwd):
    dims: Qwen35MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35moe_lin_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35moe_lin_activation_layout(self.dims)

    STAGES = _stage_prefix(Qwen35LinBlockFwd.STAGES, "ffn_norm") + MOE_SHARED_STAGES


@dataclass(frozen=True)
class Qwen35MoeLinBlockRecompute(Qwen35MoeLinBlockFwd, Qwen35LinBlockRecompute):
    pass


@dataclass(frozen=True)
class Qwen35MoeLinBlockBwd(MoEAuxTempState, MoEProfileFill, Qwen35LinBlockBwd):
    dims: Qwen35MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35moe_lin_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35moe_lin_activation_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,  # "xo"
        )


# ---------------------------------------------------------------------------
# Gated full-attention kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35MoeAttnBlockFwd(MoEAuxTempState, MoEProfileFill, Qwen35AttnBlockFwd):
    dims: Qwen35MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35moe_attn_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35moe_attn_activation_layout(self.dims)

    STAGES = _stage_prefix(Qwen35AttnBlockFwd.STAGES, "ffn_norm") + MOE_SHARED_STAGES


@dataclass(frozen=True)
class Qwen35MoeAttnBlockRecompute(Qwen35MoeAttnBlockFwd, Qwen35LinBlockRecompute):
    pass


@dataclass(frozen=True)
class Qwen35MoeAttnBlockBwd(MoEAuxTempState, MoEProfileFill, Qwen35AttnBlockBwd):
    dims: Qwen35MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35moe_attn_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35moe_attn_activation_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,  # "xo"
        )


def _opt_block_layout(d, task, w_size):
    """optimizer_block spans BOTH kinds: pick by the layer's kind (the
    AdamWStep size assert stays as the tripwire)."""
    layer = AdamWStep.layer_of(task)
    build = (
        qwen35moe_attn_weight_layout if d.kinds[layer] == "full"
        else qwen35moe_lin_weight_layout
    )
    return build(d, layer=layer), None


def build_qwen35moe_resolver(
    dims: Qwen35MoeDims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "prologue_round": RoundPrologue(dims, kernels),
        "linmoe_fwd": Qwen35MoeLinBlockFwd(dims, kernels),
        "linmoe_recompute": Qwen35MoeLinBlockRecompute(dims, kernels),
        "linmoe_bwd": Qwen35MoeLinBlockBwd(dims, kernels),
        "gattnmoe_fwd": Qwen35MoeAttnBlockFwd(dims, kernels),
        "gattnmoe_recompute": Qwen35MoeAttnBlockRecompute(dims, kernels),
        "gattnmoe_bwd": Qwen35MoeAttnBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(dims, kernels, hyper, layout_for=_opt_block_layout),
        "optimizer_embed": AdamWStep(dims, kernels, hyper, kind="embed"),
        "optimizer_head": AdamWStep(dims, kernels, hyper, kind="head"),
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
