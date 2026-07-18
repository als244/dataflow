"""Pluggable MoE-MLP module (route -> dispatch -> experts -> combine).

Self-contained: nothing outside this package imports MoE symbols except
families that opt in; the engine/planner/sim/chain-grammar never learn MoE
exists. A family plugs in through five points (docs/extending.md):

  1. layout builders compose ``spec.moe_weight_specs`` / ``moe_context_specs``;
  2. block STAGES splice ``stages.MOE_STAGES`` / ``MOE_SHARED_STAGES`` after
     the family's ffn-norm stage;
  3. the block backward's ``_mlp_bwd`` delegates to ``stages.moe_mlp_tail_bwd``
     (+ the ``MoEProfileFill`` mixin for profiling);
  4. the family golden composes ``reference.moe_mlp_reference``;
  5. the family Dims carries ``moe: MoESpec``.

Kernel implementations live in ``dataflow_training.kernels.moe_*`` (registry
families with eager fallbacks; dispatch/combine are the future
expert-parallelism all-to-all seam).

Import shape: ``spec`` is torch-free and imported EAGERLY (the layouts /
lowering layer composes it without torch); ``reference``/``stages`` import
torch and resolve LAZILY through module ``__getattr__``.
"""
from .spec import (  # noqa: F401  (torch-free)
    MoESpec,
    moe_context_specs,
    moe_local_rows,
    moe_weight_specs,
)

_LAZY = {
    "moe_topk_reference": "reference",
    "moe_aux_loss_reference": "reference",
    "moe_mlp_reference": "reference",
    "MOE_STAGES": "stages",
    "MOE_SHARED_STAGES": "stages",
    "MoEProfileFill": "stages",
    "moe_mlp_tail_bwd": "stages",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)
