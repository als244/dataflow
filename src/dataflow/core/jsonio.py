"""JSON serialization for the program IR.

The wire format mirrors the dataclasses one-to-one, with optional/empty
fields omitted for compact, diff-friendly fixtures. ``schema_version`` is
checked on load.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .program import (
    SCHEMA_VERSION,
    ObjectSpec,
    OutputSpec,
    Program,
    RecomputeOption,
    RecomputeRewrite,
    TaskSpec,
    TransferDirective,
)
from .types import TensorMeta


# --- to dict -----------------------------------------------------------------

def _tensor_to_dict(t: TensorMeta | None) -> dict[str, Any] | None:
    if t is None:
        return None
    d: dict[str, Any] = {}
    if t.dtype is not None:
        d["dtype"] = t.dtype
    if t.shape is not None:
        d["shape"] = list(t.shape)
    if t.strides is not None:
        d["strides"] = list(t.strides)
    return d or None


def _obj_to_dict(o: ObjectSpec | OutputSpec) -> dict[str, Any]:
    d: dict[str, Any] = {"id": o.id, "size_bytes": o.size_bytes, "location": o.location, "role": o.role}
    tensor = _tensor_to_dict(o.tensor)
    if tensor is not None:
        d["tensor"] = tensor
    return d


def _trigger_to_dict(t: TransferDirective) -> dict[str, Any]:
    d: dict[str, Any] = {"object_id": t.object_id}
    if t.runtime_us is not None:
        d["runtime_us"] = t.runtime_us
    return d


def _task_to_dict(t: TaskSpec) -> dict[str, Any]:
    d: dict[str, Any] = {"id": t.id, "runtime_us": t.runtime_us}
    if t.inputs:
        d["inputs"] = list(t.inputs)
    if t.outputs:
        d["outputs"] = [_obj_to_dict(o) for o in t.outputs]
    if t.mutates:
        d["mutates"] = list(t.mutates)
    if t.group != "compute":
        d["group"] = t.group
    if t.compute_block_key is not None:
        d["compute_block_key"] = t.compute_block_key
    if t.block_params:
        d["block_params"] = dict(t.block_params)
    if t.comm_groups:
        d["comm_groups"] = dict(t.comm_groups)
    if t.releases_after:
        d["releases_after"] = list(t.releases_after)
    if t.offload_after:
        d["offload_after"] = [_trigger_to_dict(x) for x in t.offload_after]
    if t.prefetch_after:
        d["prefetch_after"] = [_trigger_to_dict(x) for x in t.prefetch_after]
    if t.label is not None:
        d["label"] = t.label
    if t.metadata:
        d["metadata"] = dict(t.metadata)
    return d


def _rewrite_to_dict(rw: RecomputeRewrite) -> dict[str, Any]:
    return {
        "object_id": rw.object_id,
        "f_task_id": rw.f_task_id,
        "r_task_id": rw.r_task_id,
        "options": [
            {"level": o.level, "saved_bytes": o.saved_bytes, "recompute_us": o.recompute_us, "label": o.label}
            for o in rw.options
        ],
        "f_compute_block_key": rw.f_compute_block_key,
        "r_compute_block_key": rw.r_compute_block_key,
        "group_key": rw.group_key,
    }


def program_to_dict(p: Program) -> dict[str, Any]:
    d: dict[str, Any] = {
        "schema_version": p.schema_version,
        "name": p.name,
        "initial_objects": [_obj_to_dict(o) for o in p.initial_objects],
        "tasks": [_task_to_dict(t) for t in p.tasks],
    }
    if p.final_locations:
        d["final_locations"] = dict(p.final_locations)
    if p.fast_memory_capacity is not None:
        d["fast_memory_capacity"] = p.fast_memory_capacity
    if p.backing_memory_capacity is not None:
        d["backing_memory_capacity"] = p.backing_memory_capacity
    if p.bandwidth_from_slow is not None:
        d["bandwidth_from_slow"] = p.bandwidth_from_slow
    if p.bandwidth_to_slow is not None:
        d["bandwidth_to_slow"] = p.bandwidth_to_slow
    if p.recompute_rewrites:
        d["recompute_rewrites"] = [_rewrite_to_dict(rw) for rw in p.recompute_rewrites]
    if p.metadata:
        d["metadata"] = dict(p.metadata)
    return d


# --- from dict ---------------------------------------------------------------

def _tensor_from_dict(d: dict[str, Any] | None) -> TensorMeta | None:
    if not d:
        return None
    return TensorMeta(
        dtype=d.get("dtype"),
        shape=tuple(d["shape"]) if d.get("shape") is not None else None,
        strides=tuple(d["strides"]) if d.get("strides") is not None else None,
    )


def _object_from_dict(d: dict[str, Any]) -> ObjectSpec:
    return ObjectSpec(
        id=d["id"],
        size_bytes=int(d["size_bytes"]),
        location=d.get("location", "backing"),
        role=d.get("role", "other"),
        tensor=_tensor_from_dict(d.get("tensor")),
    )


def _output_from_dict(d: dict[str, Any]) -> OutputSpec:
    return OutputSpec(
        id=d["id"],
        size_bytes=int(d["size_bytes"]),
        location=d.get("location", "fast"),
        role=d.get("role", "other"),
        tensor=_tensor_from_dict(d.get("tensor")),
    )


def _trigger_from_dict(d: dict[str, Any]) -> TransferDirective:
    return TransferDirective(object_id=d["object_id"], runtime_us=d.get("runtime_us"))


def _task_from_dict(d: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=d["id"],
        inputs=tuple(d.get("inputs", ())),
        outputs=tuple(_output_from_dict(o) for o in d.get("outputs", ())),
        mutates=tuple(d.get("mutates", ())),
        runtime_us=float(d.get("runtime_us", 1.0)),
        group=d.get("group", "compute"),
        compute_block_key=d.get("compute_block_key"),
        block_params=dict(d.get("block_params", {})),
        comm_groups=dict(d.get("comm_groups", {})),
        releases_after=tuple(d.get("releases_after", ())),
        offload_after=tuple(_trigger_from_dict(x) for x in d.get("offload_after", ())),
        prefetch_after=tuple(_trigger_from_dict(x) for x in d.get("prefetch_after", ())),
        label=d.get("label"),
        metadata=dict(d.get("metadata", {})),
    )


def _rewrite_from_dict(d: dict[str, Any]) -> RecomputeRewrite:
    return RecomputeRewrite(
        object_id=d["object_id"],
        f_task_id=d["f_task_id"],
        r_task_id=d["r_task_id"],
        options=tuple(
            RecomputeOption(
                level=int(o["level"]),
                saved_bytes=int(o["saved_bytes"]),
                recompute_us=float(o["recompute_us"]),
                label=o.get("label", ""),
            )
            for o in d["options"]
        ),
        f_compute_block_key=d.get("f_compute_block_key", ""),
        r_compute_block_key=d.get("r_compute_block_key", ""),
        group_key=d.get("group_key", ""),
    )


def program_from_dict(d: dict[str, Any]) -> Program:
    version = d.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
    return Program(
        name=d["name"],
        initial_objects=tuple(_object_from_dict(o) for o in d.get("initial_objects", ())),
        tasks=tuple(_task_from_dict(t) for t in d.get("tasks", ())),
        final_locations=dict(d.get("final_locations", {})),
        fast_memory_capacity=d.get("fast_memory_capacity"),
        backing_memory_capacity=d.get("backing_memory_capacity"),
        bandwidth_from_slow=d.get("bandwidth_from_slow"),
        bandwidth_to_slow=d.get("bandwidth_to_slow"),
        recompute_rewrites=tuple(_rewrite_from_dict(rw) for rw in d.get("recompute_rewrites", ())),
        metadata=dict(d.get("metadata", {})),
        schema_version=version,
    )


# --- files -------------------------------------------------------------------

def save_program(program: Program, path: str | Path) -> None:
    Path(path).write_text(json.dumps(program_to_dict(program), indent=2, sort_keys=False) + "\n")


def load_program(path: str | Path) -> Program:
    return program_from_dict(json.loads(Path(path).read_text()))
