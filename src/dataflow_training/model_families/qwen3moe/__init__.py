"""qwen3moe family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(qwen3 attention + routed-MoE FFN); ``presets.py`` the smoke preset
builder; ``bridge.py`` the weight bridge into the isolated
``reference_models.qwen3moe`` twin. Only the model + preset surfaces are
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedQwen3MoeConfig,
    dims_of_qwen3moe,
    initial_values_qwen3moe,
    lower_qwen3moe,
)
from .presets import (  # noqa: F401
    qwen3moe_cfg_dict,
    qwen3moe_smoke_preset,
)
