"""glm52 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(IndexShare: cross-layer selection via M/dM objects); ``bridge.py`` the
weight bridge into the isolated ``reference_models.glm52`` twin. Only
the model surface is re-exported here — the family OBJECT is constructed
solely by its registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedGlm52Config,
    dims_of_glm52,
    initial_values_glm52,
    lower_glm52,
)
