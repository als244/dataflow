"""Structural validation of a Program.

This checks chain-order structure: id uniqueness, reference existence,
mutation subsetting, exact tensor sizes, and directive sanity. It does NOT
check movement feasibility (capacity, object state at directive fire time) —
that is location-aware semantics owned by the simulator (for plans) and the
runtime (at execution); both raise their own diagnostics.
"""
from __future__ import annotations

from .program import Program, TaskSpec
from .types import DTYPE_BITS, LOCATIONS, ROLES

_MAX_ERRORS = 50


class ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        shown = "\n  - ".join(errors[:_MAX_ERRORS])
        suffix = "" if len(errors) <= _MAX_ERRORS else f"\n  … {len(errors) - _MAX_ERRORS} more"
        super().__init__(f"program validation failed ({len(errors)} error(s)):\n  - {shown}{suffix}")


def validate_program(program: Program) -> None:
    """Raise ValidationError listing every structural problem found."""
    errors: list[str] = []
    err = errors.append

    if not program.name:
        err("program name must be non-empty")

    # --- objects -------------------------------------------------------------
    # An initial object may legitimately appear once per location: a planner
    # can pre-place a fast copy of a backing-source object (the simulator's
    # pool is keyed by (id, location)). Ids must be unique per location and
    # consistent in size/role across locations.
    created_at: dict[str, int] = {}  # object id -> task index (-1 = initial)
    initial_seen: dict[tuple[str, str], int] = {}
    initial_by_id: dict[str, tuple[int, str]] = {}  # id -> (size, role)
    for obj in program.initial_objects:
        key = (obj.id, obj.location)
        if key in initial_seen:
            err(f"duplicate initial object {obj.id!r} for location {obj.location!r}")
        initial_seen[key] = 1
        prior = initial_by_id.get(obj.id)
        if prior is not None and prior != (obj.size_bytes, obj.role):
            err(
                f"initial object {obj.id!r} appears in both locations with inconsistent "
                f"size/role: {prior} vs {(obj.size_bytes, obj.role)}"
            )
        initial_by_id[obj.id] = (obj.size_bytes, obj.role)
        created_at[obj.id] = -1
        _check_object(err, f"initial object {obj.id!r}", obj.size_bytes, obj.location, obj.role, obj.tensor)

    # --- tasks (chain order) -------------------------------------------------
    task_ids: set[str] = set()
    for idx, task in enumerate(program.tasks):
        where = f"task {task.id!r} (index {idx})"
        if task.id in task_ids:
            err(f"duplicate task id {task.id!r}")
        task_ids.add(task.id)

        if task.runtime_us < 0:
            err(f"{where}: runtime_us must be >= 0, got {task.runtime_us}")

        for inp in task.inputs:
            if inp not in created_at:
                err(f"{where}: input {inp!r} does not exist at this point in the chain")
            elif created_at[inp] >= idx:
                err(f"{where}: input {inp!r} is not created until task index {created_at[inp]}")

        seen_inputs = set(task.inputs)
        if len(seen_inputs) != len(task.inputs):
            err(f"{where}: duplicate entries in inputs")
        for mut in task.mutates:
            if mut not in seen_inputs:
                err(f"{where}: mutates {mut!r} which is not one of its inputs")

        for out in task.outputs:
            if out.id in created_at:
                origin = "initial memory" if created_at[out.id] == -1 else f"task index {created_at[out.id]}"
                err(f"{where}: output {out.id!r} collides with object created in {origin}")
            else:
                created_at[out.id] = idx
            _check_object(err, f"{where} output {out.id!r}", out.size_bytes, out.location, out.role, out.tensor)

        _check_directives(err, where, idx, task, created_at)

    # --- final locations -----------------------------------------------------
    for obj_id, loc in program.final_locations.items():
        if obj_id not in created_at:
            err(f"final_locations references unknown object {obj_id!r}")
        if loc not in LOCATIONS:
            err(f"final_locations[{obj_id!r}]: unknown location {loc!r}")

    # --- capacities / bandwidths ----------------------------------------------
    for name in ("fast_memory_capacity", "backing_memory_capacity", "bandwidth_from_slow", "bandwidth_to_slow"):
        value = getattr(program, name)
        if value is not None and value <= 0:
            err(f"{name} must be positive when set, got {value}")

    needs_from_slow = any(
        trig.runtime_us is None for t in program.tasks for trig in t.prefetch_after
    )
    needs_to_slow = any(
        trig.runtime_us is None for t in program.tasks for trig in t.offload_after
    )
    if needs_from_slow and program.bandwidth_from_slow is None:
        err("prefetch directives without runtime overrides require bandwidth_from_slow")
    if needs_to_slow and program.bandwidth_to_slow is None:
        err("offload directives without runtime overrides require bandwidth_to_slow")

    # --- recompute rewrites (soft: variants materialize only one of f/r) -----
    for rw in program.recompute_rewrites:
        tag = f"recompute rewrite for {rw.object_id!r}"
        if not rw.options:
            err(f"{tag}: options must be non-empty")
            continue
        if rw.options[0].level != 0:
            err(f"{tag}: options[0].level must be 0 (save), got {rw.options[0].level}")
        levels = [o.level for o in rw.options]
        if levels != sorted(levels) or len(set(levels)) != len(levels):
            err(f"{tag}: option levels must be strictly ascending, got {levels}")
        if rw.f_task_id not in task_ids and rw.r_task_id not in task_ids:
            err(f"{tag}: neither f_task_id {rw.f_task_id!r} nor r_task_id {rw.r_task_id!r} exists")

    if errors:
        raise ValidationError(errors)


def _check_object(err, where: str, size_bytes: int, location: str, role: str, tensor) -> None:
    if size_bytes < 1:
        err(f"{where}: size_bytes must be >= 1, got {size_bytes}")
    if location not in LOCATIONS:
        err(f"{where}: unknown location {location!r}")
    if role not in ROLES:
        err(f"{where}: unknown role {role!r}")
    if tensor is not None:
        if tensor.dtype is not None and tensor.dtype not in DTYPE_BITS:
            err(f"{where}: unknown dtype {tensor.dtype!r}")
        expected = tensor.nbytes()
        if expected is not None and expected != size_bytes:
            err(
                f"{where}: size_bytes={size_bytes} does not match dense tensor size "
                f"{expected} for shape={tensor.shape} dtype={tensor.dtype}"
            )


def _check_directives(err, where: str, idx: int, task: TaskSpec, created_at: dict[str, int]) -> None:
    def known_by_now(obj_id: str) -> bool:
        return obj_id in created_at and created_at[obj_id] <= idx

    released = set()
    for obj_id in task.releases_after:
        if obj_id in released:
            err(f"{where}: releases {obj_id!r} more than once")
        released.add(obj_id)
        if not known_by_now(obj_id):
            err(f"{where}: releases unknown/future object {obj_id!r}")

    offloaded = set()
    for trig in task.offload_after:
        if trig.object_id in offloaded:
            err(f"{where}: offloads {trig.object_id!r} more than once")
        offloaded.add(trig.object_id)
        if not known_by_now(trig.object_id):
            err(f"{where}: offloads unknown/future object {trig.object_id!r}")
        if trig.runtime_us is not None and trig.runtime_us < 0:
            err(f"{where}: offload of {trig.object_id!r} has negative runtime override")

    prefetched = set()
    for trig in task.prefetch_after:
        if trig.object_id in prefetched:
            err(f"{where}: prefetches {trig.object_id!r} more than once")
        prefetched.add(trig.object_id)
        if not known_by_now(trig.object_id):
            err(f"{where}: prefetches unknown/future object {trig.object_id!r}")
        if trig.runtime_us is not None and trig.runtime_us < 0:
            err(f"{where}: prefetch of {trig.object_id!r} has negative runtime override")

    if released & offloaded:
        err(f"{where}: objects both released and offloaded: {sorted(released & offloaded)}")
    if released & prefetched:
        err(f"{where}: objects both released and prefetched: {sorted(released & prefetched)}")
