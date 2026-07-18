"""qwen35 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(heterogeneous lin/full kinds); ``bridge.py`` the weight bridge into the
isolated ``reference_models.qwen35`` twin. Only the model surface is
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedQwen35Config,
    dims_of_qwen35,
    initial_values_qwen35,
    lower_qwen35,
)
