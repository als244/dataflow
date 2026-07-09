"""Repo-root conftest: put the repo root on ``sys.path`` so the top-level
``reference_models/`` package (isolated ground-truth models, deliberately outside
the installed ``src/`` tree) is importable in tests without a pip reinstall.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
