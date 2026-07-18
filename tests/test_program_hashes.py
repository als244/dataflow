"""Lowered-program content hashes, pinned across the refactor.

The engine/workload split moves and reorganizes nearly every module
but must not change WHAT any family lowers to: same tasks, same
compute keys, same block_params, same sizes, same order. This gate
pins the canonical-JSON sha256 of every family's smoke-preset
program — bare (family lowering truth) and planned at a fixed budget
(planner determinism) — against a committed fixture.

Regenerate ONLY when a content change is intentional:

    DATAFLOW_REGEN_HASHES=1 python -m pytest tests/test_program_hashes.py
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

FIXTURE = Path(__file__).parent / "fixtures" / "program_hashes.json"
PLAN_BUDGET = 1 << 30

SMOKE_PRESETS = {
    "llama3": "smoke_preset",
    "qwen3": "qwen3_smoke_preset",
    "qwen35": "qwen35_preset",   # hybrid ~300M; no dedicated smoke yet (A1 adds one)
    "qwen35moe": "qwen35moe_smoke_preset",
    "olmoe": "olmoe_smoke_preset",
    "qwen3moe": "qwen3moe_smoke_preset",
    "dsv3": "dsv3_smoke_preset",
    "dsv32": "dsv32_smoke_preset",
    "glm52": "glm52_smoke_preset",
}


def program_hash(program) -> str:
    from dataflow.core.jsonio import program_to_dict

    blob = json.dumps(program_to_dict(program), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def current_hashes() -> dict:
    from dataflow_training.run import presets
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    out = {}
    for name, preset_fn in SMOKE_PRESETS.items():
        cfg = getattr(presets, preset_fn)()
        fam = resolve_family(cfg)
        bare = fam.lower(cfg)
        planned = plan_program(bare, fast_memory_capacity=PLAN_BUDGET)
        out[name] = {"bare": program_hash(bare),
                     "planned": program_hash(planned.program)}
    return out


def test_lowered_program_hashes_stable():
    got = current_hashes()
    if os.environ.get("DATAFLOW_REGEN_HASHES"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(got, indent=2, sort_keys=True))
        pytest.skip(f"fixture regenerated at {FIXTURE}")
    assert FIXTURE.is_file(), (
        f"no fixture at {FIXTURE} — run once with "
        f"DATAFLOW_REGEN_HASHES=1 to record the pre-refactor truth")
    want = json.loads(FIXTURE.read_text())
    diffs = []
    for name, hashes in got.items():
        for kind in ("bare", "planned"):
            if want.get(name, {}).get(kind) != hashes[kind]:
                diffs.append(f"{name}.{kind}")
    assert not diffs, (
        f"lowered-program content changed for {diffs} — the refactor "
        f"altered WHAT these families lower to (wire strings, "
        f"block_params, sizes, or task order). Either fix the "
        f"regression or, if intentional, regenerate the fixture WITH "
        f"Shein's sign-off.")
