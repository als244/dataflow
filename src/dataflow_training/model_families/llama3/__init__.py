"""llama3 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
composing the shared templates; ``bridge.py`` the weight bridge into the
isolated ``reference_models.llama3`` twin. Only the model surface is
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedHardware,
    ShapedLlamaConfig,
    build_shaped_llama3,
    dims_of,
    family_layouts,
    initial_values,
    lower_llama3,
    tp_fill_slices,
)
