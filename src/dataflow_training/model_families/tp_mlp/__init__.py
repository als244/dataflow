"""tp_mlp family package: the tensor-parallel MLP demonstration family.
``model.py`` is self-contained — config, dims, lowering, seeded init,
AND the block executables live in the one module (no separate blocks
module, and no bridge: the toy has no ``reference_models`` twin; its
bitwise gates in tests/fleet/test_tp_mlp.py carry their own reference).
Importing this package registers the family (``register_family``), which
is how ``--plugin dataflow_training.model_families.tp_mlp`` loads it.
"""
from .model import (  # noqa: F401
    GROUP_ROLE,
    TpMlpBwd,
    TpMlpConfig,
    TpMlpDims,
    TpMlpFwd,
    build_tp_mlp_resolver,
    dims_of_tp_mlp,
    full_width_draws,
    initial_values_tp_mlp,
    lower_tp_mlp,
    register,
    silu_grads,
    tp_mlp_family,
    tp_saved_layout,
    tp_weight_layout,
)
