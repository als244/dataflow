"""Qwen3-MoE block executables: qwen3 attention verbatim + the pluggable
MoE tail.

The lightest family plug-in yet: attention (per-head qk-norm, GQA, rope
1e6) is INHERITED from the qwen3 dense classes unchanged — stages for
forward, ``_attn_bwd`` for backward. Only the FFN differs: the dense
SwiGLU stages are replaced by the spliced ``MOE_STAGES`` (route ->
dispatch -> experts -> combine; NO shared expert) and ``_mlp_bwd`` maps
to ``moe_mlp_tail_bwd``. The ``MoEProfileFill`` mixin seeds valid
balanced routing for the profiler. See docs/extending.md for the 5-point
recipe this follows.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.core import TaskSpec

from ..kernels import KernelSet, resolve_kernels
from ..layouts import (
    PackedLayout,
    Qwen3MoeDims,
    qwen3moe_context_layout,
    qwen3moe_weight_layout,
)
from ..base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss
from .llama3_blocks import BlockRecompute
from ..modules.moe.stages import MOE_STAGES, MoEMetaState, MoEProfileFill, moe_mlp_tail_bwd
from .qwen3_blocks import Qwen3BlockBwd, Qwen3BlockFwd


@dataclass(frozen=True)
class Qwen3MoeBlockFwd(MoEMetaState, MoEProfileFill, Qwen3BlockFwd):
    dims: Qwen3MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen3moe_context_layout(self.dims)

    # qwen3's attention stages through resid1_norm2, then the MoE tail
    STAGES = Qwen3BlockFwd.STAGES[:5] + MOE_STAGES


@dataclass(frozen=True)
class Qwen3MoeBlockRecompute(Qwen3MoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Qwen3MoeBlockBwd(MoEMetaState, MoEProfileFill, Qwen3BlockBwd):
    dims: Qwen3MoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen3moe_context_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,
        )

    # _attn_bwd inherited from Qwen3BlockBwd verbatim (per-head qk-norm)


def build_qwen3moe_resolver(
    dims: Qwen3MoeDims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "q3moeattn_fwd": Qwen3MoeBlockFwd(dims, kernels),
        "q3moeattn_recompute": Qwen3MoeBlockRecompute(dims, kernels),
        "q3moeattn_bwd": Qwen3MoeBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(
            dims, kernels, hyper,
            layout_for=lambda d, task, size: (
                qwen3moe_weight_layout(d, layer=AdamWStep.layer_of(task)), None,
            ),
        ),
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
