"""Bench preset registry — the seam external families use to add named
configs to the benchmark tools without editing them.

A plugin module (imported via DATAFLOW_PLUGINS or a tool's ``--plugin``
flag, see docs/extending_external.md) calls::

    from dataflow_training.run.bench_presets import register_bench_config
    register_bench_config("mymodel-tiny-s1k-bs8ga2", cfg)

``run.presets.resolve_preset`` consults this registry, so the name
works everywhere a builtin preset name does (train_solo, predict_step,
measure_step, nsys_profile).
"""
from __future__ import annotations

EXTRA_CONFIGS: dict[str, object] = {}


def register_bench_config(name: str, cfg: object) -> None:
    if name in EXTRA_CONFIGS:
        raise ValueError(f"bench config {name!r} already registered")
    EXTRA_CONFIGS[name] = cfg
