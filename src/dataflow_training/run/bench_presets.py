"""Bench preset registry — the seam external families use to add named
configs to tools/bench_train.py without editing it.

A plugin module (imported via DATAFLOW_PLUGINS, see
docs/extending_external.md) calls::

    from dataflow_training.run.bench_presets import register_bench_config
    register_bench_config("mymodel-tiny-s1k-bs8ga2", cfg)

bench_train merges these over its builtin CONFIGS at startup, so the
name works everywhere a builtin config name does (bench_train,
bench_frontier cells, best_config comparisons).
"""
from __future__ import annotations

EXTRA_CONFIGS: dict[str, object] = {}


def register_bench_config(name: str, cfg: object) -> None:
    if name in EXTRA_CONFIGS:
        raise ValueError(f"bench config {name!r} already registered")
    EXTRA_CONFIGS[name] = cfg
