"""dsv3 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(MLA + hybrid dense/MoE depth); ``bridge.py`` the weight bridge into the
isolated ``reference_models.dsv3`` twin. Only the model surface is
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedDsv3Config,
    dims_of_dsv3,
    family_layouts,
    initial_values_dsv3,
    lower_dsv3,
)
