"""dsv3 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(MLA + hybrid dense/MoE depth); ``presets.py`` the study/smoke preset
builders; ``bridge.py`` the weight bridge into the isolated
``reference_models.dsv3`` twin. Only the model + preset surfaces are
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedDsv3Config,
    derive_dims,
    family_layouts,
    initial_values_dsv3,
    lower_dsv3,
)
from .presets import (  # noqa: F401
    dsv3_2b_preset,
    dsv3_cfg_dict,
    dsv3_smoke_preset,
)
