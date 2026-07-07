"""External-family plugin seam (docs/extending_external.md).

An out-of-package family registers via DATAFLOW_PLUGINS and becomes
visible to family()/resolve_family() and to bench_train's CONFIGS —
without editing anything under src/dataflow or tools/.
"""
from __future__ import annotations

import sys
import textwrap

STUB = textwrap.dedent('''
    """Stub external family: llama3's machinery under a new name +
    config type — the minimum a plugin must do."""
    from dataclasses import dataclass

    from dataflow.training.families import Family, register_family
    from dataflow.training.llama3 import ShapedLlamaConfig, lower_llama3
    from dataflow.training.presets import register_bench_config


    @dataclass(frozen=True)
    class ShapedStubConfig(ShapedLlamaConfig):
        pass


    def _stub() -> Family:
        from dataflow.training.families import family

        base = family("llama3")
        return Family(
            name="stubfam", config_type=ShapedStubConfig,
            dims_of=base.dims_of, lower=lower_llama3,
            initial_values=base.initial_values,
            build_resolver=base.build_resolver, golden=base.golden,
        )


    register_family("stubfam", _stub)
    register_bench_config("stubfam-tiny", ShapedStubConfig.tiny())
''')


def test_plugin_registration_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "stub_plugin.py").write_text(STUB)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DATAFLOW_PLUGINS", "stub_plugin")

    import dataflow.training.families as F

    monkeypatch.setattr(F, "_plugins_loaded", False)
    F.load_plugins()
    try:
        # family() + resolve_family dispatch on the external config type
        fam = F.family("stubfam")
        import stub_plugin

        cfg = stub_plugin.ShapedStubConfig.tiny()
        assert F.resolve_family(cfg).name == "stubfam"
        # lowering works through the external entry
        prog = fam.lower(cfg)
        assert prog.task_by_id
        # the bench preset is registered and mergeable the way
        # bench_train merges it
        from dataflow.training.presets import EXTRA_CONFIGS

        assert type(EXTRA_CONFIGS["stubfam-tiny"]) is stub_plugin.ShapedStubConfig
    finally:
        F._FAMILIES.pop("stubfam", None)
        F._cache.pop("stubfam", None)
        from dataflow.training.presets import EXTRA_CONFIGS

        EXTRA_CONFIGS.pop("stubfam-tiny", None)
        sys.modules.pop("stub_plugin", None)


def test_register_family_rejects_duplicates():
    import pytest

    from dataflow.training.families import register_family

    with pytest.raises(ValueError):
        register_family("llama3", lambda: None)
