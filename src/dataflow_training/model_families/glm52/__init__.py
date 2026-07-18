"""glm52 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(IndexShare: cross-layer selection via M/dM objects); ``presets.py`` the
smoke preset builder; ``bridge.py`` the weight bridge into the isolated
``reference_models.glm52`` twin. Only the model + preset surfaces are
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedGlm52Config,
    dims_of_glm52,
    initial_values_glm52,
    lower_glm52,
)
from .presets import (  # noqa: F401
    glm52_cfg_dict,
    glm52_smoke_preset,
)
