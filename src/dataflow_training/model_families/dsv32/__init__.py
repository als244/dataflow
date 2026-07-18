"""dsv32 family package: ``model.py`` holds the Shaped config, dims,
seeded init, and lowering entry; ``blocks.py`` the block executables
(dsv3 + DSA lightning indexer, sparse mode); ``presets.py`` the smoke
preset builder; ``bridge.py`` the weight bridge into the isolated
``reference_models.dsv32`` twin. Only the model + preset surfaces are
re-exported here — the family OBJECT is constructed solely by its
registry thunk in ``..families``.
"""
from .model import (  # noqa: F401
    ShapedDsv32Config,
    dims_of_dsv32,
    initial_values_dsv32,
    lower_dsv32,
)
from .presets import (  # noqa: F401
    dsv32_cfg_dict,
    dsv32_smoke_preset,
)
