"""qwen35moe family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(heterogeneous linmoe/gattnmoe kinds); ``presets.py`` the smoke preset
builder; ``bridge.py`` the weight bridge into the isolated
``reference_models.qwen35moe`` twin. Only the model + preset surfaces
are re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedQwen35MoeConfig,
    dims_of_qwen35moe,
    initial_values_qwen35moe,
    lower_qwen35moe,
)
from .presets import (  # noqa: F401
    qwen35moe_cfg_dict,
    qwen35moe_smoke_preset,
)
