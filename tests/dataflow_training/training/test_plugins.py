"""External-family plugin seam (docs/extending_external.md).

An out-of-package family registers through either discovery path —
a packaging entry point (installed distributions) or an explicit
module list (tools' --plugin flag) — and becomes visible to
family()/resolve_family(), the bench preset table, and the structural
contract validator, without editing anything under src/dataflow or
tools/.
"""
from __future__ import annotations

import sys
import textwrap

STUB = textwrap.dedent('''
    """Stub external family: llama3's machinery under a new name +
    config type — the minimum a plugin must do."""
    from dataclasses import dataclass

    from dataflow_training.model_families.families import ModelFamily, register_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig, lower_llama3
    from dataflow_training.run.bench_presets import register_bench_config


    @dataclass(frozen=True)
    class ShapedStubConfig(ShapedLlamaConfig):
        pass


    def _stub() -> ModelFamily:
        from dataflow_training.model_families.families import family

        base = family("llama3")
        return ModelFamily(
            name="stubfam", config_type=ShapedStubConfig,
            derive_dims=base.derive_dims, lower=lower_llama3,
            initial_values=base.initial_values,
            build_resolver=base.build_resolver,
            family_layouts=base.family_layouts,
        )


    register_family("stubfam", _stub)
    register_bench_config("stubfam-tiny", ShapedStubConfig.tiny())
''')


def _cleanup():
    import dataflow_training.model_families.families as F
    from dataflow_training.run.bench_presets import EXTRA_CONFIGS

    F._FAMILIES.pop("stubfam", None)
    F._cache.pop("stubfam", None)
    EXTRA_CONFIGS.pop("stubfam-tiny", None)
    sys.modules.pop("stub_plugin", None)


def test_explicit_plugin_load_end_to_end(tmp_path, monkeypatch):
    """The tools' --plugin path: load_plugins(explicit=[module])."""
    (tmp_path / "stub_plugin.py").write_text(STUB)
    monkeypatch.syspath_prepend(str(tmp_path))

    import dataflow_training.model_families.families as F

    F.load_plugins(explicit=["stub_plugin"])
    try:
        fam = F.family("stubfam")
        import stub_plugin

        cfg = stub_plugin.ShapedStubConfig.tiny()
        # exact-type-first dispatch: the subclassed config resolves to
        # the EXTERNAL family, not llama3
        assert F.resolve_family(cfg).name == "stubfam"
        assert fam.lower(cfg).task_by_id()
        # registered bench presets resolve by name everywhere the
        # tools resolve builtin preset names
        from dataflow_training.run.bench_presets import EXTRA_CONFIGS
        from dataflow_training.run.presets import resolve_preset

        assert type(EXTRA_CONFIGS["stubfam-tiny"]) is stub_plugin.ShapedStubConfig
        assert resolve_preset("stubfam-tiny") is EXTRA_CONFIGS["stubfam-tiny"]
        # the structural contract validator accepts it
        assert F.validate_family("stubfam") == []
    finally:
        _cleanup()


def test_entry_point_discovery(tmp_path, monkeypatch):
    """The packaging path: a dataflow.families entry point is loaded by
    load_plugins() with no flags or env."""
    (tmp_path / "stub_plugin.py").write_text(STUB)
    monkeypatch.syspath_prepend(str(tmp_path))

    import dataflow_training.model_families.families as F

    class _EP:
        name = "stubfam"

        @staticmethod
        def load():
            import importlib

            return importlib.import_module("stub_plugin")

    monkeypatch.setattr(F, "_plugins_loaded", False)
    import importlib.metadata as md

    real = md.entry_points
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None, **kw: [_EP] if group == "dataflow.families"
        else real(group=group, **kw),
    )
    F.load_plugins()
    try:
        assert "stubfam" in F._FAMILIES
    finally:
        _cleanup()


def test_validate_family_reports_broken_surface():
    import dataflow_training.model_families.families as F

    def _broken():
        base = F.family("llama3")
        return F.ModelFamily(
            name="brokenfam", config_type=dict,  # not a dataclass, no tiny()
            derive_dims=base.derive_dims, lower=base.lower,
            initial_values=base.initial_values,
            build_resolver=base.build_resolver,
            family_layouts=base.family_layouts,
        )

    F.register_family("brokenfam", _broken)
    try:
        problems = F.validate_family("brokenfam")
        assert problems and any("dataclass" in p for p in problems)
    finally:
        F._FAMILIES.pop("brokenfam", None)
        F._cache.pop("brokenfam", None)


def test_register_family_rejects_duplicates():
    import pytest

    from dataflow_training.model_families.families import register_family

    with pytest.raises(ValueError):
        register_family("llama3", lambda: None)
