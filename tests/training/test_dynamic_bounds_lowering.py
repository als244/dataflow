"""C4 increment-1 gates (CPU): dynamic-bounds lowering emission.

s_max=None must be a byte-level no-op (legacy programs unchanged —
the tripwire hashes elsewhere pin this globally); s_max set must
emit bounds/positions per round and thread them APPEND-ONLY into
block fwd / recompute / bwd inputs (existing positional indices
stable — the audit rule).
"""
from __future__ import annotations

import dataclasses
import json

from dataflow.core.jsonio import program_to_dict
from dataflow.training.models.llama3 import ShapedLlamaConfig, lower_llama3


def _tiny(**kw):
    return dataclasses.replace(ShapedLlamaConfig.tiny(), **kw)


def test_none_is_inert():
    p = lower_llama3(_tiny())
    assert not any(o.id.startswith(("bounds_", "positions_"))
                   for o in p.initial_objects)
    assert not any("bounds_" in i for t in p.tasks for i in t.inputs)


def test_dynamic_emission_and_threading():
    cfg = _tiny(s_max=64, grad_accum_rounds=2)
    p = lower_llama3(cfg)
    ids = {o.id for o in p.initial_objects}
    for r in range(2):
        assert f"bounds_0_{r}" in ids and f"positions_0_{r}" in ids
    b = next(o for o in p.initial_objects if o.id == "bounds_0_0")
    assert b.size_bytes == (64 + 1) * 4
    assert tuple(b.tensor.shape) == (65,)
    pos = next(o for o in p.initial_objects if o.id == "positions_0_0")
    assert pos.size_bytes == cfg.seq_len * cfg.batch * 4

    for kind in ("block_fwd", "block_bwd"):
        for r in range(2):
            t = next(t for t in p.tasks if t.id == f"{kind}_0_{r}_0")
            assert t.inputs[-2:] == (f"bounds_0_{r}", f"positions_0_{r}"), \
                (kind, t.inputs)


def test_append_only_prefix_stable():
    """Every legacy input appears at the SAME index with and without
    dynamic mode — the freeze-era positional-index lesson, gated."""
    p0 = lower_llama3(_tiny())
    p1 = lower_llama3(_tiny(s_max=64))
    t0 = {t.id: t for t in p0.tasks}
    for t1 in p1.tasks:
        legacy = t0.get(t1.id)
        if legacy is None:
            continue
        assert t1.inputs[:len(legacy.inputs)] == legacy.inputs, t1.id


def test_roundtrip_serialization():
    p = lower_llama3(_tiny(s_max=32))
    d = program_to_dict(p)
    js = json.dumps(d)
    assert "bounds_0_0" in js and "positions_0_0" in js
