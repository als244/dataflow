"""Model-family registry: the ONE place that maps a shaped config to its
lowering, dims, executables, and gradcheck bundle; the
correctness authority is the isolated reference twin (reference_models/).

The train loop, gradcheck harness, and sweep tools dispatch through
``resolve_family(cfg)`` instead of importing a family's modules directly —
adding a family means one `ModelFamily` entry here (docs/extending.md §7), not
edits across the harnesses.

Golden classes resolve lazily (models is the layer ABOVE training; the
import-boundary rule allows training→models only inside functions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class DeriveDimsFn(Protocol):
    """cfg -> the family's Dims object (validated: layer-0/pattern rules,
    incompatible knob combinations raise here, at build time)."""

    def __call__(self, cfg) -> object: ...


class LowerFn(Protocol):
    """cfg -> Program. Task/object ids MUST keep the repo naming shape
    ``<prefix>_{step}_{round}_{layer}`` / ``A_ dW_ W_ O_ Aux_ AuxTemp_``
    — the planner, drivers, and analyzers key on it. Accepts
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
class ModelFamily:
    """Everything one model family contributes, as data: the config
    type, the lowering surface, the resolver builder, the parity-twin
    hooks, and the gradcheck bundle. A NEW family is three files (model
    / blocks / bridge), one twin, and one registry line — the
    parametrized gates apply automatically. ``Model`` binds a family to
    ONE concrete config for ergonomic call sites."""

    name: str
    config_type: type          # a frozen dataclass with preset classmethods
    derive_dims: DeriveDimsFn
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
    # parity-twin hooks: import path into reference_models/ + the bridge
    # module exposing build_reference_model/load_reference_init/
    # to_reference_state_dict (None until the family grows a twin)
    twin_module: str | None = None
    bridge_module: str | None = None

    def cfg_dict(self, cfg) -> dict:
        from dataflow_training.run.presets import cfg_dict as cfg_to_dict

        return cfg_to_dict(cfg)

    def smoke_config(self):
        return self.config_type.tiny()

    def resolver_spec(self, cfg, hyper: dict | None = None) -> dict:
        from dataflow_training.register import canonical_spec

        return canonical_spec(self.name, self.cfg_dict(cfg), hyper)

    def bind(self, cfg) -> "Model":
        return Model(self, cfg)


@dataclass(frozen=True)
class Model:
    """A ModelFamily bound to ONE concrete config: the ergonomic
    handle run/ drivers and tools pass around instead of (family, cfg)
    tuples."""

    family: ModelFamily
    cfg: object

    @property
    def dims(self):
        return self.family.derive_dims(self.cfg)

    def lower(self, **kw):
        return self.family.lower(self.cfg, **kw)

    def cfg_dict(self) -> dict:
        return self.family.cfg_dict(self.cfg)

    def resolver_spec(self, hyper: dict | None = None) -> dict:
        return self.family.resolver_spec(self.cfg, hyper)

    def reference(self, *, device="cuda"):
        """Build the isolated pure-torch twin for this config (the
        family's bridge module owns construction + init loading)."""
        from dataflow_training.model_families import bridges

        return bridges.build_reference_model(self.cfg, device=device)



def _llama3() -> ModelFamily:
    from dataflow_training.blocks.layouts import activation_layout, weight_layout
    from dataflow_training.model_families.llama3.blocks import BlockBwd, BlockFwd, BlockRecompute, build_resolver
    from .llama3 import derive_dims, initial_values, lower_llama3
    from .llama3 import ShapedLlamaConfig

    return ModelFamily(
        name="llama3",
        config_type=ShapedLlamaConfig,
        derive_dims=derive_dims,
        lower=lower_llama3,
        initial_values=initial_values,
        build_resolver=build_resolver,
        block_fwd=BlockFwd,
        block_bwd=BlockBwd,
        block_recompute=BlockRecompute,
        weight_layout=weight_layout,
        activation_layout=activation_layout,
        twin_module="reference_models.llama3",
        bridge_module="dataflow_training.model_families.llama3.bridge",
    )


def _qwen3() -> ModelFamily:
    from dataflow_training.blocks.layouts import qwen3_activation_layout, qwen3_weight_layout
    from dataflow_training.model_families.qwen3.blocks import (
        Qwen3BlockBwd,
        Qwen3BlockFwd,
        Qwen3BlockRecompute,
        build_qwen3_resolver,
    )
    from .qwen3 import derive_dims, initial_values_qwen3, lower_qwen3
    from .qwen3 import ShapedQwen3Config

    return ModelFamily(
        name="qwen3",
        config_type=ShapedQwen3Config,
        derive_dims=derive_dims,
        lower=lower_qwen3,
        initial_values=initial_values_qwen3,
        build_resolver=build_qwen3_resolver,
        block_fwd=Qwen3BlockFwd,
        block_bwd=Qwen3BlockBwd,
        block_recompute=Qwen3BlockRecompute,
        weight_layout=qwen3_weight_layout,
        activation_layout=qwen3_activation_layout,
        twin_module="reference_models.qwen3",
        bridge_module="dataflow_training.model_families.qwen3.bridge",
    )


def _qwen35() -> ModelFamily:
    from dataflow_training.model_families.qwen35.blocks import build_qwen35_resolver
    from .qwen35 import initial_values_qwen35, lower_qwen35
    from .qwen35 import ShapedQwen35Config, derive_dims

    # heterogeneous (lin/full kinds) — the per-kind block ladders live in
    # tests/dataflow_training/models/test_qwen35.py, so no generic gradcheck bundle here
    return ModelFamily(
        name="qwen35",
        config_type=ShapedQwen35Config,
        derive_dims=derive_dims,
        lower=lower_qwen35,
        initial_values=initial_values_qwen35,
        build_resolver=build_qwen35_resolver,
        twin_module="reference_models.qwen35",
        bridge_module="dataflow_training.model_families.qwen35.bridge",
    )


def _olmoe() -> ModelFamily:
    from dataflow_training.model_families.olmoe.blocks import build_olmoe_resolver
    from .olmoe import ShapedOlmoeConfig, derive_dims, initial_values_olmoe, lower_olmoe

    # MoE family — the block ladder needs the aux-loss term in the
    # reference objective, so it lives in tests/dataflow_training/models/test_olmoe.py
    # rather than the generic check_block_backward harness
    return ModelFamily(
        name="olmoe",
        config_type=ShapedOlmoeConfig,
        derive_dims=derive_dims,
        lower=lower_olmoe,
        initial_values=initial_values_olmoe,
        build_resolver=build_olmoe_resolver,
        twin_module="reference_models.olmoe",
        bridge_module="dataflow_training.model_families.olmoe.bridge",
    )


def _qwen35moe() -> ModelFamily:
    from dataflow_training.model_families.qwen35moe.blocks import build_qwen35moe_resolver
    from .qwen35moe import (
        ShapedQwen35MoeConfig,
        derive_dims,
        initial_values_qwen35moe,
        lower_qwen35moe,
    )

    # heterogeneous MoE (linmoe/gattnmoe kinds) — per-kind ladders live in
    # tests/dataflow_training/models/test_qwen35moe.py
    return ModelFamily(
        name="qwen35moe",
        config_type=ShapedQwen35MoeConfig,
        derive_dims=derive_dims,
        lower=lower_qwen35moe,
        initial_values=initial_values_qwen35moe,
        build_resolver=build_qwen35moe_resolver,
        twin_module="reference_models.qwen35moe",
        bridge_module="dataflow_training.model_families.qwen35moe.bridge",
    )


def _qwen3moe() -> ModelFamily:
    from dataflow_training.model_families.qwen3moe.blocks import build_qwen3moe_resolver
    from .qwen3moe import (
        ShapedQwen3MoeConfig,
        derive_dims,
        initial_values_qwen3moe,
        lower_qwen3moe,
    )

    # MoE family (aux objective) — block ladder lives in
    # tests/dataflow_training/models/test_qwen3moe.py (no gradcheck bundle)
    return ModelFamily(
        name="qwen3moe",
        config_type=ShapedQwen3MoeConfig,
        derive_dims=derive_dims,
        lower=lower_qwen3moe,
        initial_values=initial_values_qwen3moe,
        build_resolver=build_qwen3moe_resolver,
        twin_module="reference_models.qwen3moe",
        bridge_module="dataflow_training.model_families.qwen3moe.bridge",
    )


def _dsv3() -> ModelFamily:
    from dataflow_training.model_families.dsv3.blocks import build_dsv3_resolver
    from .dsv3 import (
        ShapedDsv3Config,
        derive_dims,
        initial_values_dsv3,
        lower_dsv3,
    )

    # MLA + hybrid dense/MoE depth + sigmoid_noaux_tc — block ladder lives
    # in tests/dataflow_training/modules/test_mla.py, family ladder in tests/dataflow_training/models/test_dsv3.py
    return ModelFamily(
        name="dsv3",
        config_type=ShapedDsv3Config,
        derive_dims=derive_dims,
        lower=lower_dsv3,
        initial_values=initial_values_dsv3,
        build_resolver=build_dsv3_resolver,
        twin_module="reference_models.dsv3",
        bridge_module="dataflow_training.model_families.dsv3.bridge",
    )


def _dsv32() -> ModelFamily:
    from dataflow_training.model_families.dsv32.blocks import build_dsv32_resolver
    from .dsv32 import (
        ShapedDsv32Config,
        derive_dims,
        initial_values_dsv32,
        lower_dsv32,
    )

    # dsv3 + DSA (lightning indexer, sparse mode) — ladders in
    # tests/dataflow_training/modules/test_dsa.py + tests/dataflow_training/models/test_dsv32.py
    return ModelFamily(
        name="dsv32",
        config_type=ShapedDsv32Config,
        derive_dims=derive_dims,
        lower=lower_dsv32,
        initial_values=initial_values_dsv32,
        build_resolver=build_dsv32_resolver,
        twin_module="reference_models.dsv32",
        bridge_module="dataflow_training.model_families.dsv32.bridge",
    )


def _gpt2() -> ModelFamily:
    from dataflow_training.blocks.layouts import gpt2_activation_layout, gpt2_weight_layout
    from dataflow_training.model_families.gpt2.blocks import (
        Gpt2BlockBwd,
        Gpt2BlockFwd,
        Gpt2BlockRecompute,
        build_gpt2_resolver,
    )
    from .gpt2 import ShapedGpt2Config, derive_dims, initial_values, lower_gpt2

    return ModelFamily(
        name="gpt2",
        config_type=ShapedGpt2Config,
        derive_dims=derive_dims,
        lower=lower_gpt2,
        initial_values=initial_values,
        build_resolver=build_gpt2_resolver,
        block_fwd=Gpt2BlockFwd,
        block_bwd=Gpt2BlockBwd,
        block_recompute=Gpt2BlockRecompute,
        weight_layout=gpt2_weight_layout,
        activation_layout=gpt2_activation_layout,
        twin_module="reference_models.gpt2",
        bridge_module="dataflow_training.model_families.gpt2.bridge",
    )


def _glm52() -> ModelFamily:
    from .glm52 import (
        ShapedGlm52Config,
        derive_dims,
        initial_values_glm52,
        lower_glm52,
    )

    from dataflow_training.model_families.glm52.blocks import build_glm52_resolver

    # IndexShare: cross-layer selection via M/dM objects — ladder in
    # tests/dataflow_training/models/test_glm52.py + tests/dataflow_training/models/test_glm52_lowering.py
    return ModelFamily(
        name="glm52",
        config_type=ShapedGlm52Config,
        derive_dims=derive_dims,
        lower=lower_glm52,
        initial_values=initial_values_glm52,
        build_resolver=build_glm52_resolver,
        twin_module="reference_models.glm52",
        bridge_module="dataflow_training.model_families.glm52.bridge",
    )


_FAMILIES: dict[str, Callable[[], ModelFamily]] = {
    "gpt2": _gpt2,
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
_cache: dict[str, ModelFamily] = {}


def register_family(name: str, thunk: Callable[[], ModelFamily]) -> None:
    """Register a model family from OUTSIDE the dataflow package.

    ``thunk`` is a zero-arg callable returning a ``ModelFamily`` (lazy, so
    registration is import-cheap). External families become visible to
    ``family()`` / ``resolve_family()`` and thereby to every tool
    (predict_step, measure_step, train_solo, verify_family). See
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
        dims = fam.derive_dims(cfg)
    except Exception as exc:
        return problems + [f"derive_dims raised: {exc!r}"]
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


def family(name: str) -> ModelFamily:
    if name not in _cache:
        _cache[name] = _FAMILIES[name]()
    return _cache[name]


def resolve_family(cfg) -> ModelFamily:
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

def build_init_program(fam, cfg, *, seed: int = 0,
                       object_sizes: dict | None = None,
                       tp_view: dict | None = None):
    """INIT IS A PROGRAM: one "family_init" task whose OUTPUTS are the
    training program's initial objects (backing-resident), filled by
    the family's seeded init inside the task — byte-identical to the
    in-process ``initial_values`` path by construction. Replaces the
    retired materialize_group service verb: register + run this
    program through the ordinary verbs, and the daemon's final-object
    capture persists every W_/O_/Aux_/data object into the store.

    ``object_sizes`` overrides per-object byte sizes (sharded-optimizer
    runs shrink O_*); ``tp_view`` selects a per-rank weight view for
    tensor-parallel fills (families that support it)."""
    import dataclasses as dc

    from dataflow.core.program import OutputSpec, Program, TaskSpec

    train = fam.lower(cfg)
    outs = []
    for spec in train.initial_objects:
        size = spec.size_bytes
        if object_sizes and spec.id in object_sizes:
            size = int(object_sizes[spec.id])
        outs.append(OutputSpec(id=spec.id, size_bytes=size,
                               location="backing", role=spec.role,
                               tensor=spec.tensor))
    params = {"seed": int(seed)}
    if tp_view is not None:
        params["tp_view"] = tp_view
    task = TaskSpec(
        id="family_init_0",
        outputs=tuple(outs),
        compute_block_key="family_init",
        block_params=params,
    )
    return Program(
        name=f"{getattr(train, 'name', fam.name)}-init",
        initial_objects=(),
        tasks=(task,),
        final_locations={o.id: "backing" for o in outs},
    )
