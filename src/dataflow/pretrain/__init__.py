"""Pretraining scaffold: data pipeline, LR schedule, recipe, model presets,
the parity bridge and the one-loop-two-backends driver.

One-way dependency: ``pretrain`` imports ``dataflow`` (families, service
client, tasks) — never the reverse. The pytorch reference backend touches
only ``torch`` + the isolated top-level ``references/`` package.
"""
from __future__ import annotations

from .fineweb import DEFAULT_ROOT, FinewebStream, ShardCorpus, make_stream
from .presets import (
    LADDER,
    LADDER_NAMES,
    cfg_dict,
    param_counts,
    preset,
    smoke_preset,
    tokens_per_step,
)
from .recipe import DEFAULT_RECIPE, Recipe
from .schedule import CosineSchedule

__all__ = [
    "DEFAULT_ROOT", "FinewebStream", "ShardCorpus", "make_stream",
    "LADDER", "LADDER_NAMES", "cfg_dict", "param_counts", "preset",
    "smoke_preset", "tokens_per_step", "DEFAULT_RECIPE", "Recipe",
    "CosineSchedule",
]
