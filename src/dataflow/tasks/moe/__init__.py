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

Kernel implementations live in ``dataflow.tasks.kernels.moe_*`` (registry
families with eager fallbacks; dispatch/combine are the future
expert-parallelism all-to-all seam).
"""
from .reference import (  # noqa: F401
    moe_aux_loss_reference,
    moe_mlp_reference,
    moe_topk_reference,
)
from .spec import (  # noqa: F401
    MoESpec,
    moe_context_specs,
    moe_local_rows,
    moe_weight_specs,
)
from .stages import (  # noqa: F401
    MOE_SHARED_STAGES,
    MOE_STAGES,
    MoEProfileFill,
    moe_mlp_tail_bwd,
)
