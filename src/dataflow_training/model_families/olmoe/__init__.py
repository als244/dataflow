"""olmoe family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(routed-MoE FFN over the shared MoE stages); ``bridge.py`` the weight
bridge into the isolated ``reference_models.olmoe`` twin. Only the model
surface is re-exported here — the family OBJECT is constructed solely by
its registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedOlmoeConfig,
    dims_of_olmoe,
    family_layouts,
    initial_values_olmoe,
    lower_olmoe,
)
