"""qwen3 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables;
``presets.py`` the smoke preset builder; ``bridge.py`` the weight bridge
into the isolated ``reference_models.qwen3`` twin. Only the model +
preset surfaces are re-exported here — the family OBJECT is constructed
solely by its registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedQwen3Config,
    dims_of_qwen3,
    initial_values_qwen3,
    lower_qwen3,
)
from .presets import (  # noqa: F401
    qwen3_cfg_dict,
    qwen3_smoke_preset,
)
