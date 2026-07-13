"""Program IR: objects, tasks, movement directives, recompute metadata.

A ``Program`` is a linear task chain (list order == execution order for the
current runtime) over named objects. Dependencies are nevertheless fully
derivable from producer/consumer relations, so a DAG scheduler can adopt the
same IR later without a format change.

The same schema represents both stages of a program's life:

- *bare*: tasks carry inputs/mutates/outputs and runtimes; all directive
  fields are empty. This is what lowering emits and what planners consume.
- *annotated*: a planner (e.g. PressureFit via ``dataflow_sim``) has filled
  ``releases_after`` / ``offload_after`` / ``prefetch_after`` and possibly
  adjusted initial object locations. This is what the runtime executes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .types import Location, Role, TensorMeta


@dataclass(frozen=True)
class ObjectSpec:
    """An object present before the chain starts (initial memory)."""

    id: str
    size_bytes: int
    location: Location = "backing"
    role: Role = "other"
    tensor: TensorMeta | None = None


@dataclass(frozen=True)
class OutputSpec:
    """An object created by a task. Output ids are always fresh."""

    id: str
    size_bytes: int
    location: Location = "fast"
    role: Role = "other"
    tensor: TensorMeta | None = None


@dataclass(frozen=True)
class TransferDirective:
    """One offload/prefetch trigger fired at the anchoring task's end.

    ``runtime_us`` overrides the bandwidth-derived transfer time for this one
    transfer (mirrors the simulator's per-trigger override).
    """

    object_id: str
    runtime_us: float | None = None


@dataclass(frozen=True)
class TaskSpec:
    """One compute task in the chain.

    ``inputs`` are read-only unless also listed in ``mutates`` (in-place
    update; the object's version advances and its backing copy goes stale).
    ``compute_block_key`` + ``block_params`` identify the executable that
    implements this task — resolution is by key, never by task id, so
    planner-inserted tasks (e.g. recompute) bind automatically.

    ``comm_groups`` maps a comm PURPOSE to the NAME of the peer group
    serving it: {"dp": name} — this task exchanges gradients over
    that group; {"tp": name} — tensor-parallel activation
    collectives.
    The name is the lookup key into the per-run ctx.groups table
    (create_peer_group named it; topology [groups.X]); a run without
    that group executes the task standalone. Empty means a pure-local
    task. Group addressing lives HERE; ``block_params`` stays
    geometry/math the block needs.
    """

    id: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[OutputSpec, ...] = ()
    mutates: tuple[str, ...] = ()
    runtime_us: float = 1.0
    group: str = "compute"
    compute_block_key: str | None = None
    block_params: Mapping[str, Any] = field(default_factory=dict)
    comm_groups: Mapping[str, str] = field(default_factory=dict)
    releases_after: tuple[str, ...] = ()
    offload_after: tuple[TransferDirective, ...] = ()
    prefetch_after: tuple[TransferDirective, ...] = ()
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_directives(
        self,
        *,
        releases_after: tuple[str, ...] | None = None,
        offload_after: tuple[TransferDirective, ...] | None = None,
        prefetch_after: tuple[TransferDirective, ...] | None = None,
    ) -> "TaskSpec":
        return replace(
            self,
            releases_after=self.releases_after if releases_after is None else releases_after,
            offload_after=self.offload_after if offload_after is None else offload_after,
            prefetch_after=self.prefetch_after if prefetch_after is None else prefetch_after,
        )


@dataclass(frozen=True)
class RecomputeOption:
    """One save-vs-recompute choice for a saved-context object.

    Level 0 means "save normally"; higher levels save fewer forward bytes and
    add recompute work before backward. Mirrors
    ``dataflow_sim.workloads.common.recompute.RecomputeOption`` field-for-field
    so conversion is lossless.
    """

    level: int
    saved_bytes: int
    recompute_us: float
    label: str = ""


@dataclass(frozen=True)
class RecomputeRewrite:
    """Planner-visible recompute choices for one saved-context object."""

    object_id: str
    f_task_id: str
    r_task_id: str
    options: tuple[RecomputeOption, ...]
    f_compute_block_key: str = ""
    r_compute_block_key: str = ""
    group_key: str = ""


SCHEMA_VERSION = "dataflow-rt/v1"


@dataclass(frozen=True)
class Program:
    """A complete dataflow program (bare or annotated).

    Time unit is microseconds throughout. Bandwidths are integer bytes per
    microsecond (matching the simulator's transfer model). ``final_locations``
    are terminal placement constraints keyed by object id; objects omitted are
    disposable after last use.
    """

    name: str
    initial_objects: tuple[ObjectSpec, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()
    final_locations: Mapping[str, Location] = field(default_factory=dict)
    fast_memory_capacity: int | None = None
    backing_memory_capacity: int | None = None
    bandwidth_from_slow: int | None = None
    bandwidth_to_slow: int | None = None
    recompute_rewrites: tuple[RecomputeRewrite, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # -- convenience lookups (computed, not stored) --------------------------

    def object_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {o.id: o.size_bytes for o in self.initial_objects}
        for t in self.tasks:
            for out in t.outputs:
                sizes[out.id] = out.size_bytes
        return sizes

    def producers(self) -> dict[str, str]:
        """object id -> producing task id (initial objects absent)."""
        prod: dict[str, str] = {}
        for t in self.tasks:
            for out in t.outputs:
                prod[out.id] = t.id
        return prod

    def task_by_id(self) -> dict[str, TaskSpec]:
        return {t.id: t for t in self.tasks}

    def is_annotated(self) -> bool:
        return any(
            t.releases_after or t.offload_after or t.prefetch_after for t in self.tasks
        )
