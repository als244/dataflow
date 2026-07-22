"""Layout-registry gates: every family's layouts are keyed, validated,
and BYTE-IDENTITY-PINNED. The digests cover field tables (name, shape,
dtype, offset, size), totals, couplings, and roots — init byte order
rides field order (certified), so these pins are what allow layout
construction to migrate to declarative form incrementally: any
reordering, resizing, or renaming trips here first. Update digests
ONLY with a deliberate layout change, in the same commit.

Tests:
- test_registry_validates_and_digest_pinned: each family's registered layouts validate cleanly and hash to the pinned digest.
- test_registry_addresses_every_weight_root: every emitted W root is reachable through exactly one registered key, with no duplicates.
- test_registry_covers_external_family: a plugin-loaded external family validates and registers under its own namespaced keys.
"""
import pytest

from dataflow_training.model_families.layout_registry import (
    layouts_digest,
    registered_layouts,
    validate_layouts,
)

# tiny-config digests captured 2026-07-21 from the battery-certified
# builders, BEFORE any declarative migration.
PINNED = {
    "gpt2": "1f753d515bd23990",
    "llama3": "44c503e642931caf",
    "qwen3": "27b6b862319f6095",
    "qwen35": "78746cee5d18fd47",
    "olmoe": "12a369239441f5cb",
    "qwen35moe": "5becac2e5252bf0a",
    "qwen3moe": "44ebe4a7e88bf101",
    "dsv3": "a635ee8c1cfcb990",
    "dsv32": "148f78e0055b4f81",
    "glm52": "592dbce370f25b4b",
}


def tiny(family_name):
    from dataflow_training.model_families.families import _FAMILIES

    fam = _FAMILIES[family_name]()
    return fam, fam.config_type.tiny()


@pytest.mark.parametrize("family_name", sorted(PINNED))
def test_registry_validates_and_digest_pinned(family_name):
    fam, cfg = tiny(family_name)
    assert validate_layouts(cfg) == []
    assert layouts_digest(cfg) == PINNED[family_name]


@pytest.mark.parametrize("family_name", sorted(PINNED))
def test_registry_addresses_every_weight_root(family_name):
    """Every W root the lowering emits is reachable through exactly
    one registered key — the coordinate-system property the
    responsibility map derives byte ranges through."""
    fam, cfg = tiny(family_name)
    reg = registered_layouts(cfg)
    roots = [r for rl in reg.values() for r in rl.roots]
    assert len(roots) == len(set(roots)), "root claimed twice"
    expect = {f"W_{i}" for i in range(cfg.n_layers)} | {"W_embed", "W_head"}
    assert set(roots) == expect


def test_registry_covers_external_family(monkeypatch):
    """The toy external family conforms to the same contract — loaded
    through the PLUGIN path (single module identity; the same
    fresh-registration lifecycle the external suite owns)."""
    import sys

    from dataflow_training.distributed.topology import repo_root

    import dataflow_training.model_families.families as F
    from dataflow.service import registry as service_registry

    F._FAMILIES.pop("toyfam", None)
    F._cache.pop("toyfam", None)
    for key in [k for k in service_registry._CACHE if "toyfam" in k[1]]:
        del service_registry._CACHE[key]
    sys.modules.pop("toy_family", None)
    monkeypatch.syspath_prepend(
        str(repo_root() / "tests" / "fixtures"
            / "external_family"))
    F.load_plugins(explicit=["toy_family"])
    try:
        fam = F.family("toyfam")
        cfg = fam.config_type.tiny()
        assert validate_layouts(cfg) == []
        reg = registered_layouts(cfg)
        assert any(k.startswith("toyfam/") for k in reg)
    finally:
        F._FAMILIES.pop("toyfam", None)
        F._cache.pop("toyfam", None)
        for key in [k for k in service_registry._CACHE
                    if "toyfam" in k[1]]:
            del service_registry._CACHE[key]
        sys.modules.pop("toy_family", None)
