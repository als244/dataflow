"""Model-family registry: the ONE place that maps a shaped config to its
lowering, dims, executables, golden reference, and gradcheck bundle.

The train loop, gradcheck harness, and sweep tools dispatch through
``resolve_family(cfg)`` instead of importing a family's modules directly —
adding a family means one `Family` entry here (docs/extending.md §6), not
edits across the harnesses.

Golden classes resolve lazily (models is the layer ABOVE training; the
import-boundary rule allows training→models only inside functions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Family:
    name: str
    config_type: type
    dims_of: Callable
    lower: Callable
    initial_values: Callable
    build_resolver: Callable
    golden: Callable  # zero-arg, returns the golden model class (lazy import)
    # gradcheck bundle (ladder level 2) — None for heterogeneous families,
    # whose per-kind block ladders live in their own test module instead of
    # the generic check_block_backward harness
    block_fwd: type | None = None
    block_bwd: type | None = None
    block_recompute: type | None = None
    weight_layout: Callable | None = None
    context_layout: Callable | None = None


def _llama3() -> Family:
    from dataflow.tasks.layouts import context_layout, weight_layout
    from dataflow.tasks.llama3_blocks import (
        BlockBwd,
        BlockFwd,
        BlockRecompute,
        build_resolver,
    )
    from .llama3 import dims_of, initial_values, lower_llama3
    from .llama3 import ShapedLlamaConfig

    def golden():
        from dataflow.models.llama3_reference import GoldenLlama3

        return GoldenLlama3

    return Family(
        name="llama3",
        config_type=ShapedLlamaConfig,
        dims_of=dims_of,
        lower=lower_llama3,
        initial_values=initial_values,
        build_resolver=build_resolver,
        golden=golden,
        block_fwd=BlockFwd,
        block_bwd=BlockBwd,
        block_recompute=BlockRecompute,
        weight_layout=weight_layout,
        context_layout=context_layout,
    )


def _qwen3() -> Family:
    from dataflow.tasks.layouts import qwen3_context_layout, qwen3_weight_layout
    from dataflow.tasks.qwen3_blocks import (
        Qwen3BlockBwd,
        Qwen3BlockFwd,
        Qwen3BlockRecompute,
        build_qwen3_resolver,
    )
    from .qwen3 import dims_of_qwen3, initial_values_qwen3, lower_qwen3
    from .qwen3 import ShapedQwen3Config

    def golden():
        from dataflow.models.qwen3_reference import GoldenQwen3

        return GoldenQwen3

    return Family(
        name="qwen3",
        config_type=ShapedQwen3Config,
        dims_of=dims_of_qwen3,
        lower=lower_qwen3,
        initial_values=initial_values_qwen3,
        build_resolver=build_qwen3_resolver,
        golden=golden,
        block_fwd=Qwen3BlockFwd,
        block_bwd=Qwen3BlockBwd,
        block_recompute=Qwen3BlockRecompute,
        weight_layout=qwen3_weight_layout,
        context_layout=qwen3_context_layout,
    )


def _qwen35() -> Family:
    from dataflow.tasks.qwen35_blocks import build_qwen35_resolver
    from .qwen35 import initial_values_qwen35, lower_qwen35
    from .qwen35 import ShapedQwen35Config, dims_of_qwen35

    def golden():
        from dataflow.models.qwen35_reference import GoldenQwen35

        return GoldenQwen35

    # heterogeneous (lin/full kinds) — the per-kind block ladders live in
    # tests/tasks/test_qwen35_math.py, so no generic gradcheck bundle here
    return Family(
        name="qwen35",
        config_type=ShapedQwen35Config,
        dims_of=dims_of_qwen35,
        lower=lower_qwen35,
        initial_values=initial_values_qwen35,
        build_resolver=build_qwen35_resolver,
        golden=golden,
    )


def _olmoe() -> Family:
    from dataflow.tasks.olmoe_blocks import build_olmoe_resolver
    from .olmoe import ShapedOlmoeConfig, dims_of_olmoe, initial_values_olmoe, lower_olmoe

    def golden():
        from dataflow.models.olmoe_reference import GoldenOlmoe

        return GoldenOlmoe

    # MoE family — the block ladder needs the aux-loss term in the golden
    # objective, so it lives in tests/tasks/test_olmoe_math.py rather than
    # the generic check_block_backward harness (no gradcheck bundle)
    return Family(
        name="olmoe",
        config_type=ShapedOlmoeConfig,
        dims_of=dims_of_olmoe,
        lower=lower_olmoe,
        initial_values=initial_values_olmoe,
        build_resolver=build_olmoe_resolver,
        golden=golden,
    )


def _qwen35moe() -> Family:
    from dataflow.tasks.qwen35moe_blocks import build_qwen35moe_resolver
    from .qwen35moe import (
        ShapedQwen35MoeConfig,
        dims_of_qwen35moe,
        initial_values_qwen35moe,
        lower_qwen35moe,
    )

    def golden():
        from dataflow.models.qwen35moe_reference import GoldenQwen35Moe

        return GoldenQwen35Moe

    # heterogeneous MoE (linmoe/gattnmoe kinds) — per-kind ladders live in
    # tests/tasks/test_qwen35moe_math.py
    return Family(
        name="qwen35moe",
        config_type=ShapedQwen35MoeConfig,
        dims_of=dims_of_qwen35moe,
        lower=lower_qwen35moe,
        initial_values=initial_values_qwen35moe,
        build_resolver=build_qwen35moe_resolver,
        golden=golden,
    )


def _qwen3moe() -> Family:
    from dataflow.tasks.qwen3moe_blocks import build_qwen3moe_resolver
    from .qwen3moe import (
        ShapedQwen3MoeConfig,
        dims_of_qwen3moe,
        initial_values_qwen3moe,
        lower_qwen3moe,
    )

    def golden():
        from dataflow.models.qwen3moe_reference import GoldenQwen3Moe

        return GoldenQwen3Moe

    # MoE family (aux objective) — block ladder lives in
    # tests/tasks/test_qwen3moe_math.py (no gradcheck bundle)
    return Family(
        name="qwen3moe",
        config_type=ShapedQwen3MoeConfig,
        dims_of=dims_of_qwen3moe,
        lower=lower_qwen3moe,
        initial_values=initial_values_qwen3moe,
        build_resolver=build_qwen3moe_resolver,
        golden=golden,
    )


_FAMILIES: dict[str, Callable[[], Family]] = {
    "llama3": _llama3,
    "qwen3": _qwen3,
    "qwen35": _qwen35,
    "olmoe": _olmoe,
    "qwen35moe": _qwen35moe,
    "qwen3moe": _qwen3moe,
}
_cache: dict[str, Family] = {}


def family(name: str) -> Family:
    if name not in _cache:
        _cache[name] = _FAMILIES[name]()
    return _cache[name]


def resolve_family(cfg) -> Family:
    """Dispatch on the shaped-config type."""
    for name in _FAMILIES:
        fam = family(name)
        if isinstance(cfg, fam.config_type):
            return fam
    raise TypeError(
        f"no registered model family for config type {type(cfg).__name__!r} "
        f"(known: {sorted(_FAMILIES)})"
    )
