"""Per-family weight bridges: the engine's packed init bytes -> the isolated
``reference_models`` ``nn.Module`` state_dicts, byte-identical.

Parity invariant #2: the initialization is seeded ONCE by the engine
(``initial_values`` / the daemon's init program); a bridge loads those
exact bytes into the reference model so the ONLY variable between the two
training runs is the execution engine.

One module per family (``llama3.py``, ``qwen3.py``, ...), each exposing the
uniform pair the driver dispatches on —

    build_reference_model(cfg, *, device, dtype) -> nn.Module
    load_reference_init(model, cfg, dims, get_bytes) -> nn.Module

— plus the family's ``to_*_state_dict`` mapping (the testable core). Shared
byte plumbing lives in ``common``. The engine stores each block's projection
matrices packed ``(in, out)``; references use ``nn.Linear`` (weight
``(out, in)``), so projections are TRANSPOSED (a pure layout change — same
values, same bits). Embedding / LM-head tables ``(vocab, d)``, 1-D gains and
other vectors load directly; a depthwise conv reshapes ``(D, W) -> (D, 1, W)``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# reference_models/ lives at the repo root (outside the installed src/ tree).
# Importing this package always runs before its family submodules, so the
# path is armed before they import reference_models at top level. (Tests get
# the same path from the root conftest; this covers script use.)
ROOT = str(Path(__file__).resolve().parents[4])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import (  # noqa: E402
    dsv3,
    dsv32,
    glm52,
    llama3,
    olmoe,
    qwen3,
    qwen35,
    qwen35moe,
    qwen3moe,
)
from .common import (  # noqa: E402,F401
    assert_state_dict_byte_identical,
    bytes_from_buffer,
    get_bytes_from_client,
    get_bytes_from_values,
    load_state_dict_strict,
    transposed,
)

# Shaped config type name -> the family's bridge module.
FAMILY_BRIDGES = {
    "ShapedDsv3Config": dsv3,
    "ShapedDsv32Config": dsv32,
    "ShapedGlm52Config": glm52,
    "ShapedLlamaConfig": llama3,
    "ShapedOlmoeConfig": olmoe,
    "ShapedQwen3Config": qwen3,
    "ShapedQwen3MoeConfig": qwen3moe,
    "ShapedQwen35Config": qwen35,
    "ShapedQwen35MoeConfig": qwen35moe,
}


def bridge_module(cfg):
    """The family bridge module for ``cfg``'s config type."""
    name = type(cfg).__name__
    if name not in FAMILY_BRIDGES:
        raise KeyError(
            f"no weight bridge for config type {name}; "
            f"bridged families: {sorted(FAMILY_BRIDGES)}")
    return FAMILY_BRIDGES[name]


def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16):
    """Build the reference nn.Module for ``cfg``'s family (family-dispatched)."""
    return bridge_module(cfg).build_reference_model(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    """Load the engine's packed init into ``model`` (family-dispatched)."""
    return bridge_module(cfg).load_reference_init(model, cfg, dims, get_bytes)


def to_reference_state_dict(cfg, get_bytes):
    """Engine packed bytes -> twin-named state dict (family-dispatched);
    the comparison space of the engine-vs-reference gates."""
    return bridge_module(cfg).to_reference_state_dict(cfg, get_bytes)
