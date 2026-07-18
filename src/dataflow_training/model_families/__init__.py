"""Model families, one package per family: ``X/model.py`` holds the
Shaped config + dims + lowering entry, ``X/blocks.py`` the block
executables composing the shared templates (../blocks/base_blocks.py),
and ``X/bridge.py`` the weight bridge into the isolated
``reference_models.X`` twin. Cross-cutting modules stay flat:
``families.py`` (the ModelFamily/Model registry — the ONE construction
point for family objects), ``bridges.py`` (config-dispatched bridge
entry points), ``bridge_common.py`` (shared bridge byte plumbing), and
``init_policy.py``. ``tp_mlp/`` is the self-registering
tensor-parallel demonstration family (model+blocks in one module, no
twin, so no bridge).
"""
from __future__ import annotations

import sys
from pathlib import Path

# reference_models/ lives at the repo root (outside the installed src/
# tree). This package init runs before ANY family submodule, so the path
# is armed before a family's ``bridge`` imports reference_models at top
# level. (Tests get the same path from the root conftest; this covers
# script use.)
ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
