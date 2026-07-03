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
    # gradcheck bundle (ladder level 2)
    block_fwd: type
    block_bwd: type
    block_recompute: type
    weight_layout: Callable
    context_layout: Callable


def _llama3() -> Family:
    from dataflow.tasks.layouts import context_layout, weight_layout
    from dataflow.tasks.llama3_blocks import (
        BlockBwd,
        BlockFwd,
        BlockRecompute,
        build_resolver,
    )
    from .llama3_lowering import dims_of, initial_values, lower_llama3
    from .shaped_llama3 import ShapedLlamaConfig

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
    from .qwen3_lowering import dims_of_qwen3, initial_values_qwen3, lower_qwen3
    from .shaped_qwen3 import ShapedQwen3Config

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


_FAMILIES: dict[str, Callable[[], Family]] = {
    "llama3": _llama3,
    "qwen3": _qwen3,
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
