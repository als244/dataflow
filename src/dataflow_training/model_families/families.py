"""Model-family registry: the ONE place that maps a shaped config to its
lowering, dims, executables, and gradcheck bundle; the
correctness authority is the isolated reference twin (reference_models/).

The train loop, gradcheck harness, and sweep tools dispatch through
``resolve_family(cfg)`` instead of importing a family's modules directly —
adding a family means one `Family` entry here (docs/extending.md §6), not
edits across the harnesses.

Golden classes resolve lazily (models is the layer ABOVE training; the
import-boundary rule allows training→models only inside functions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class DimsOfFn(Protocol):
    """cfg -> the family's Dims object (validated: layer-0/pattern rules,
    incompatible knob combinations raise here, at build time)."""

    def __call__(self, cfg) -> object: ...


class LowerFn(Protocol):
    """cfg -> Program. Task/object ids MUST keep the repo naming shape
    ``<prefix>_{step}_{round}_{layer}`` / ``A_ dW_ W_ O_ M_ dM_`` — the
    planner, train loop, and analyzers key on it. Accepts
    ``recompute_levels=`` so the planner can re-lower variants."""

    def __call__(self, cfg, recompute_levels=None) -> object: ...


class InitialValuesFn(Protocol):
    """(program, cfg, backend, seed) -> {object_id: pinned host tensor}.
    Generation ORDER is part of reference comparability."""

    def __call__(self, program, cfg, backend, seed: int = 0) -> dict: ...


class BuildResolverFn(Protocol):
    """dims -> resolver. The resolver is a CALLABLE ``task -> executable``
    where the executable exposes ``launch(ctx)`` (see docs/task-contract.md
    for what launch may do). It must resolve every task the family's
    lowering emits, including planner-inserted recompute tasks (key by
    compute_block_key, never task id)."""

    def __call__(self, dims) -> object: ...


@dataclass(frozen=True)
class Family:
    name: str
    config_type: type          # a frozen dataclass with preset classmethods
    dims_of: DimsOfFn
    lower: LowerFn
    initial_values: InitialValuesFn
    build_resolver: BuildResolverFn
    # gradcheck bundle (ladder level 2) — None for heterogeneous families,
    # whose per-kind block ladders live in their own test module instead of
    # the generic check_block_backward harness
    block_fwd: type | None = None
    block_bwd: type | None = None
    block_recompute: type | None = None
    weight_layout: Callable | None = None
    activation_layout: Callable | None = None


def _llama3() -> Family:
    from dataflow_training.blocks.layouts import activation_layout, weight_layout
    from dataflow_training.model_families.llama3_blocks import BlockBwd, BlockFwd, BlockRecompute, build_resolver
    from .llama3 import dims_of, initial_values, lower_llama3
    from .llama3 import ShapedLlamaConfig

    return Family(
        name="llama3",
        config_type=ShapedLlamaConfig,
        dims_of=dims_of,
        lower=lower_llama3,
        initial_values=initial_values,
        build_resolver=build_resolver,
        block_fwd=BlockFwd,
        block_bwd=BlockBwd,
        block_recompute=BlockRecompute,
        weight_layout=weight_layout,
        activation_layout=activation_layout,
    )


def _qwen3() -> Family:
    from dataflow_training.blocks.layouts import qwen3_activation_layout, qwen3_weight_layout
    from dataflow_training.model_families.qwen3_blocks import (
        Qwen3BlockBwd,
        Qwen3BlockFwd,
        Qwen3BlockRecompute,
        build_qwen3_resolver,
    )
    from .qwen3 import dims_of_qwen3, initial_values_qwen3, lower_qwen3
    from .qwen3 import ShapedQwen3Config

    return Family(
        name="qwen3",
        config_type=ShapedQwen3Config,
        dims_of=dims_of_qwen3,
        lower=lower_qwen3,
        initial_values=initial_values_qwen3,
        build_resolver=build_qwen3_resolver,
        block_fwd=Qwen3BlockFwd,
        block_bwd=Qwen3BlockBwd,
        block_recompute=Qwen3BlockRecompute,
        weight_layout=qwen3_weight_layout,
        activation_layout=qwen3_activation_layout,
    )


def _qwen35() -> Family:
    from dataflow_training.model_families.qwen35_blocks import build_qwen35_resolver
    from .qwen35 import initial_values_qwen35, lower_qwen35
    from .qwen35 import ShapedQwen35Config, dims_of_qwen35

    # heterogeneous (lin/full kinds) — the per-kind block ladders live in
    # tests/models/test_qwen35.py, so no generic gradcheck bundle here
    return Family(
        name="qwen35",
        config_type=ShapedQwen35Config,
        dims_of=dims_of_qwen35,
        lower=lower_qwen35,
        initial_values=initial_values_qwen35,
        build_resolver=build_qwen35_resolver,
    )


def _olmoe() -> Family:
    from dataflow_training.model_families.olmoe_blocks import build_olmoe_resolver
    from .olmoe import ShapedOlmoeConfig, dims_of_olmoe, initial_values_olmoe, lower_olmoe

    # MoE family — the block ladder needs the aux-loss term in the
    # reference objective, so it lives in tests/models/test_olmoe.py
    # rather than the generic check_block_backward harness
    return Family(
        name="olmoe",
        config_type=ShapedOlmoeConfig,
        dims_of=dims_of_olmoe,
        lower=lower_olmoe,
        initial_values=initial_values_olmoe,
        build_resolver=build_olmoe_resolver,
    )


def _qwen35moe() -> Family:
    from dataflow_training.model_families.qwen35moe_blocks import build_qwen35moe_resolver
    from .qwen35moe import (
        ShapedQwen35MoeConfig,
        dims_of_qwen35moe,
        initial_values_qwen35moe,
        lower_qwen35moe,
    )

    # heterogeneous MoE (linmoe/gattnmoe kinds) — per-kind ladders live in
    # tests/models/test_qwen35moe.py
    return Family(
        name="qwen35moe",
        config_type=ShapedQwen35MoeConfig,
        dims_of=dims_of_qwen35moe,
        lower=lower_qwen35moe,
        initial_values=initial_values_qwen35moe,
        build_resolver=build_qwen35moe_resolver,
    )


def _qwen3moe() -> Family:
    from dataflow_training.model_families.qwen3moe_blocks import build_qwen3moe_resolver
    from .qwen3moe import (
        ShapedQwen3MoeConfig,
        dims_of_qwen3moe,
        initial_values_qwen3moe,
        lower_qwen3moe,
    )

    # MoE family (aux objective) — block ladder lives in
    # tests/models/test_qwen3moe.py (no gradcheck bundle)
    return Family(
        name="qwen3moe",
        config_type=ShapedQwen3MoeConfig,
        dims_of=dims_of_qwen3moe,
        lower=lower_qwen3moe,
        initial_values=initial_values_qwen3moe,
        build_resolver=build_qwen3moe_resolver,
    )


def _dsv3() -> Family:
    from dataflow_training.model_families.dsv3_blocks import build_dsv3_resolver
    from .dsv3 import (
        ShapedDsv3Config,
        dims_of_dsv3,
        initial_values_dsv3,
        lower_dsv3,
    )

    # MLA + hybrid dense/MoE depth + sigmoid_noaux_tc — block ladder lives
    # in tests/modules/test_mla.py, family ladder in tests/models/test_dsv3.py
    return Family(
        name="dsv3",
        config_type=ShapedDsv3Config,
        dims_of=dims_of_dsv3,
        lower=lower_dsv3,
        initial_values=initial_values_dsv3,
        build_resolver=build_dsv3_resolver,
    )


def _dsv32() -> Family:
    from dataflow_training.model_families.dsv32_blocks import build_dsv32_resolver
    from .dsv32 import (
        ShapedDsv32Config,
        dims_of_dsv32,
        initial_values_dsv32,
        lower_dsv32,
    )

    # dsv3 + DSA (lightning indexer, sparse mode) — ladders in
    # tests/modules/test_dsa.py + tests/models/test_dsv32.py
    return Family(
        name="dsv32",
        config_type=ShapedDsv32Config,
        dims_of=dims_of_dsv32,
        lower=lower_dsv32,
        initial_values=initial_values_dsv32,
        build_resolver=build_dsv32_resolver,
    )


def _glm52() -> Family:
    from .glm52 import (
        ShapedGlm52Config,
        dims_of_glm52,
        initial_values_glm52,
        lower_glm52,
    )

    from dataflow_training.model_families.glm52_blocks import build_glm52_resolver

    # IndexShare: cross-layer selection via M/dM objects — ladder in
    # tests/models/test_glm52.py + tests/models/test_glm52_lowering.py
    return Family(
        name="glm52",
        config_type=ShapedGlm52Config,
        dims_of=dims_of_glm52,
        lower=lower_glm52,
        initial_values=initial_values_glm52,
        build_resolver=build_glm52_resolver,
    )


_FAMILIES: dict[str, Callable[[], Family]] = {
    "llama3": _llama3,
    "qwen3": _qwen3,
    "qwen35": _qwen35,
    "olmoe": _olmoe,
    "qwen35moe": _qwen35moe,
    "qwen3moe": _qwen3moe,
    "dsv3": _dsv3,
    "dsv32": _dsv32,
    "glm52": _glm52,
}
_cache: dict[str, Family] = {}


def register_family(name: str, thunk: Callable[[], Family]) -> None:
    """Register a model family from OUTSIDE the dataflow package.

    ``thunk`` is a zero-arg callable returning a ``Family`` (lazy, so
    registration is import-cheap). External families become visible to
    ``family()`` / ``resolve_family()`` and thereby to every tool
    (bench_train, best_config, bench_frontier, verify_family). See
    docs/extending_external.md.
    """
    if name in _FAMILIES:
        raise ValueError(f"family {name!r} already registered")
    _FAMILIES[name] = thunk


_plugins_loaded = False


def load_plugins(explicit: list[str] | None = None) -> None:
    """Discover and import external-family plugins.

    Two mechanisms, both importing modules that self-register via
    ``register_family`` (+ optionally ``presets.register_bench_config``):

    1. PACKAGING (the normal path): installed distributions declaring a
       ``dataflow.families`` entry point are discovered automatically —
       in the external package's pyproject.toml::

           [project.entry-points."dataflow.families"]
           mymodel = "mypkg.dataflow_plugin"

    2. EXPLICIT (dev loop / uninstalled code): tools accept
       ``--plugin mypkg.dataflow_plugin`` and pass it here.

    Idempotent for the entry-point scan; explicit modules import once
    via the interpreter's module cache.
    """
    global _plugins_loaded
    import importlib

    if not _plugins_loaded:
        _plugins_loaded = True
        from importlib.metadata import entry_points

        for ep in entry_points(group="dataflow.families"):
            ep.load()
    for name in explicit or ():
        importlib.import_module(name.strip())


def validate_family(name: str, *, preset: str = "tiny") -> list[str]:
    """Structural contract check for a (typically external) family —
    fast, no GPU math: catches wiring mistakes before the deep ladders.
    Returns human-readable problems (empty = surface OK)."""
    import dataclasses
    import re as _re

    problems: list[str] = []
    fam = family(name)
    cls = fam.config_type
    if not dataclasses.is_dataclass(cls):
        problems.append(f"config_type {cls.__name__} is not a dataclass")
    method = getattr(cls, preset, None)
    if method is None:
        problems.append(f"{cls.__name__} lacks a {preset}() preset classmethod")
        return problems
    cfg = method()
    try:
        dims = fam.dims_of(cfg)
    except Exception as exc:
        return problems + [f"dims_of raised: {exc!r}"]
    try:
        prog = fam.lower(cfg)
    except Exception as exc:
        return problems + [f"lower raised: {exc!r}"]
    ids = list(prog.task_by_id())
    shape = _re.compile(r"^[a-z0-9_]+_\d+_\d+_\d+$|^(head_loss|embed_fwd|embed_bwd|optimizer)")
    bad = [i for i in ids if not shape.match(i)]
    if bad:
        problems.append(f"task ids off the naming shape (first 3): {bad[:3]}")
    try:
        resolver = fam.build_resolver(dims)
        unresolved = []
        for task in prog.task_by_id().values():
            try:
                ex = resolver(task)
            except Exception:
                unresolved.append(task.id)
                continue
            if not hasattr(ex, "launch"):
                unresolved.append(f"{task.id} (no .launch)")
        if unresolved:
            problems.append(f"resolver failed for {len(unresolved)} tasks "
                            f"(first 3): {unresolved[:3]}")
    except Exception as exc:
        problems.append(f"build_resolver raised: {exc!r}")
    return problems


def family(name: str) -> Family:
    if name not in _cache:
        _cache[name] = _FAMILIES[name]()
    return _cache[name]


def resolve_family(cfg) -> Family:
    """Dispatch on the shaped-config type — EXACT type first, then
    isinstance. Exact-first makes it safe for an external family to
    subclass a builtin config (docs/extending_external.md); builtin
    families all use distinct types, so their dispatch is unchanged."""
    for name in _FAMILIES:
        fam = family(name)
        if type(cfg) is fam.config_type:
            return fam
    for name in _FAMILIES:
        fam = family(name)
        if isinstance(cfg, fam.config_type):
            return fam
    raise TypeError(
        f"no registered model family for config type {type(cfg).__name__!r} "
        f"(known: {sorted(_FAMILIES)})"
    )
