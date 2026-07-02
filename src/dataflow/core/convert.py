"""Converters between the program IR and dataflow_sim's schemas.

Three interop points:

- ``to_sim_chain(program)``            -> ``dataflow_sim.core.schema.TaskChain``
  (bare or annotated; what policies, the recompute planner, and the simulator
  consume)
- ``apply_chain_annotations(program, chain)`` -> annotated ``Program``
  (joins a planner-annotated chain's directives + pre-placed initial fast
  copies back onto the IR, preserving tensor/binding metadata)
- ``to_webapp_program(program)``       -> DataflowProgram v1 JSON dict
  (the schema the webapp accepts at POST /api/simulate with source="schema")

``dataflow_sim`` is imported lazily inside each function so that importing
``dataflow.core`` stays dependency-free; only callers that actually convert
need the simulator installed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .program import ObjectSpec, OutputSpec, Program, TaskSpec, TransferDirective
from .types import Role

# Our roles -> simulator ObjectType bands (used by the sim UI's memory bands).
_ROLE_TO_SIM_TYPE: dict[str, str] = {
    "parameter": "weight",
    "gradient": "gradient",
    "optimizer_state": "optimizer",
    "activation": "activation",
    "input": "activation",
    "output": "activation",
    "temp": "other",
    "other": "other",
}

_SIM_TYPE_TO_ROLE: dict[str, Role] = {
    "weight": "parameter",
    "gradient": "gradient",
    "optimizer": "optimizer_state",
    "activation": "activation",
    "other": "other",
}


def role_to_sim_type(role: str) -> str:
    return _ROLE_TO_SIM_TYPE.get(role, "other")


# --- Program -> TaskChain ------------------------------------------------------

def to_sim_chain(program: Program) -> Any:
    from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain, TransferTrigger

    return TaskChain(
        initial_memory=[
            Object(
                id=o.id,
                size=int(o.size_bytes),
                location=o.location,
                type=role_to_sim_type(o.role),
            )
            for o in program.initial_objects
        ],
        tasks=[
            Task(
                id=t.id,
                inputs=list(t.inputs),
                outputs=[
                    OutputAlloc(
                        id=out.id,
                        size=int(out.size_bytes),
                        location=out.location,
                        type=role_to_sim_type(out.role),
                    )
                    for out in t.outputs
                ],
                runtime=float(t.runtime_us),
                releases_after=list(t.releases_after),
                offload_after=[
                    TransferTrigger(obj_id=x.object_id, runtime=x.runtime_us)
                    for x in t.offload_after
                ],
                prefetch_after=[
                    TransferTrigger(obj_id=x.object_id, runtime=x.runtime_us)
                    for x in t.prefetch_after
                ],
                mutates_inputs=list(t.mutates),
            )
            for t in program.tasks
        ],
        final_locations=dict(program.final_locations),
        fast_memory_capacity=program.fast_memory_capacity,
        backing_memory_capacity=program.backing_memory_capacity,
        bandwidth_from_slow=program.bandwidth_from_slow,
        bandwidth_to_slow=program.bandwidth_to_slow,
    )


# --- annotated TaskChain -> Program -------------------------------------------

def apply_chain_annotations(program: Program, chain: Any) -> Program:
    """Join a planner-annotated chain back onto the IR program.

    The chain must contain exactly the program's tasks, in order (planners
    annotate; they do not add/remove/reorder tasks — recompute variants are
    produced by re-lowering, not by chain surgery). Directives are copied
    onto each task; pre-placed initial fast copies (extra (id, "fast")
    initial-memory entries emitted by PressureFit) are appended to
    ``initial_objects`` with tensor metadata inherited from the backing entry.
    """
    chain_task_ids = [t.id for t in chain.tasks]
    program_task_ids = [t.id for t in program.tasks]
    if chain_task_ids != program_task_ids:
        raise ValueError(
            "annotated chain tasks do not match program tasks "
            f"(chain has {len(chain_task_ids)}, program has {len(program_task_ids)}; "
            "first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(chain_task_ids, program_task_ids)) if a != b), min(len(chain_task_ids), len(program_task_ids)))})"
        )

    new_tasks: list[TaskSpec] = []
    for spec, sim_task in zip(program.tasks, chain.tasks):
        new_tasks.append(
            spec.with_directives(
                releases_after=tuple(sim_task.releases_after),
                offload_after=tuple(
                    TransferDirective(object_id=x.obj_id, runtime_us=x.runtime)
                    for x in sim_task.offload_after
                ),
                prefetch_after=tuple(
                    TransferDirective(object_id=x.obj_id, runtime_us=x.runtime)
                    for x in sim_task.prefetch_after
                ),
            )
        )

    # Reconcile initial memory: the annotated chain may carry extra pre-placed
    # fast entries for backing-source objects.
    existing = {(o.id, o.location) for o in program.initial_objects}
    by_id = {o.id: o for o in program.initial_objects}
    new_initial = list(program.initial_objects)
    for sim_obj in chain.initial_memory:
        key = (sim_obj.id, sim_obj.location)
        if key in existing:
            continue
        src = by_id.get(sim_obj.id)
        if src is None:
            raise ValueError(
                f"annotated chain adds initial object {sim_obj.id!r} unknown to the program"
            )
        new_initial.append(
            ObjectSpec(
                id=src.id,
                size_bytes=src.size_bytes,
                location=sim_obj.location,
                role=src.role,
                tensor=src.tensor,
            )
        )
        existing.add(key)

    # The planner-annotated chain is authoritative for capacities/bandwidths
    # (policies may embed a capacity override into the chain they return).
    return replace(
        program,
        tasks=tuple(new_tasks),
        initial_objects=tuple(new_initial),
        final_locations=dict(chain.final_locations),
        fast_memory_capacity=chain.fast_memory_capacity,
        backing_memory_capacity=chain.backing_memory_capacity,
        bandwidth_from_slow=chain.bandwidth_from_slow,
        bandwidth_to_slow=chain.bandwidth_to_slow,
    )


# --- Program -> DataflowProgram v1 (webapp upload) -----------------------------

def to_webapp_program(
    program: Program,
    *,
    description: str = "",
    primary_unit: str | None = None,
    primary_count: float | None = None,
) -> dict[str, Any]:
    """Export as a DataflowProgram v1 JSON dict (webapp custom upload).

    Cost mapping: if a task's metadata carries ``cost_subops`` (a list of
    dicts with kind fixed/roofline fields), those become the task's compute
    block subops so the webapp can re-resolve runtimes for any hardware.
    Otherwise the task's ``runtime_us`` is exported as a fixed cost.

    Movement directives are not part of DataflowProgram v1 — the webapp
    applies its own policy. Export the *bare* program (annotations, if any,
    are simply dropped here).
    """
    from dataflow_sim.workloads.dataflow import (
        ComputeBlock,
        DataflowCost,
        DataflowMetrics,
        DataflowObject,
        DataflowOutput,
        DataflowProgram,
        DataflowTask,
    )

    def _cost_from_dict(d: dict[str, Any], default_name: str) -> DataflowCost:
        kind = d.get("kind", "fixed")
        if kind == "fixed":
            return DataflowCost(kind="fixed", name=d.get("name", default_name), runtime_us=d["runtime_us"])
        if kind == "roofline":
            return DataflowCost(
                kind="roofline",
                name=d.get("name", default_name),
                flops=int(d.get("flops", 0)),
                memory_bytes=int(d.get("memory_bytes", 0)),
                efficiency=d.get("efficiency", "memory"),
                count=int(d.get("count", 1)),
            )
        raise ValueError(f"unsupported cost_subops kind {kind!r} on task metadata")

    # One compute block per distinct compute_block_key; tasks without a key get
    # inline fixed costs (the sim normalizer wraps those into one-off blocks).
    blocks: dict[str, ComputeBlock] = {}
    tasks: list[DataflowTask] = []
    seen_fast_initial: set[str] = set()

    for t in program.tasks:
        subops_meta = t.metadata.get("cost_subops")
        block_key = t.compute_block_key
        cost = None
        if block_key is not None:
            if block_key not in blocks:
                if subops_meta:
                    subops = [_cost_from_dict(s, f"{block_key}:{i}") for i, s in enumerate(subops_meta)]
                else:
                    subops = [DataflowCost(kind="fixed", name=block_key, runtime_us=t.runtime_us)]
                blocks[block_key] = ComputeBlock(
                    key=block_key,
                    name=t.label or block_key,
                    category=t.group,
                    subops=subops,
                )
        else:
            cost = DataflowCost(kind="fixed", name=t.label or t.id, runtime_us=t.runtime_us)

        tasks.append(
            DataflowTask(
                id=t.id,
                label=t.label,
                group=t.group,
                compute_block_key=block_key,
                inputs=list(t.inputs),
                outputs=[
                    DataflowOutput(id=o.id, size_bytes=o.size_bytes, role=o.role, location=o.location)
                    for o in t.outputs
                ],
                mutates=list(t.mutates),
                cost=cost,
            )
        )

    objects = []
    for o in program.initial_objects:
        # DataflowProgram has one entry per object id; prefer the backing
        # source entry and skip planner-added pre-placed fast duplicates.
        if o.id in seen_fast_initial:
            continue
        seen_fast_initial.add(o.id)
        objects.append(
            DataflowObject(
                id=o.id,
                size_bytes=o.size_bytes,
                initial_location=o.location,
                role=o.role,
            )
        )

    metrics = None
    unit = primary_unit or program.metadata.get("primary_unit")
    count = primary_count if primary_count is not None else program.metadata.get("primary_count")
    if unit is not None and count is not None:
        metrics = DataflowMetrics(primary_unit=str(unit), primary_count=float(count))

    prog = DataflowProgram(
        name=program.name,
        description=description or str(program.metadata.get("description", "")),
        metadata={k: v for k, v in program.metadata.items() if k not in ("primary_unit", "primary_count", "description")},
        metrics=metrics,
        objects=objects,
        compute_blocks=list(blocks.values()),
        tasks=tasks,
        final_locations=dict(program.final_locations),
    )
    return prog.model_dump(mode="json")


# --- TaskChain -> Program (tests / adopting sim-built chains) -------------------

def from_sim_chain(chain: Any, *, name: str) -> Program:
    program = Program(
        name=name,
        initial_objects=tuple(
            ObjectSpec(
                id=o.id,
                size_bytes=int(o.size),
                location=o.location,
                role=_SIM_TYPE_TO_ROLE.get(o.type, "other"),
            )
            for o in chain.initial_memory
        ),
        tasks=tuple(
            TaskSpec(
                id=t.id,
                inputs=tuple(t.inputs),
                outputs=tuple(
                    OutputSpec(
                        id=o.id,
                        size_bytes=int(o.size),
                        location=o.location,
                        role=_SIM_TYPE_TO_ROLE.get(o.type, "other"),
                    )
                    for o in t.outputs
                ),
                mutates=tuple(t.mutates_inputs),
                runtime_us=float(t.runtime),
                releases_after=tuple(t.releases_after),
                offload_after=tuple(
                    TransferDirective(object_id=x.obj_id, runtime_us=x.runtime)
                    for x in t.offload_after
                ),
                prefetch_after=tuple(
                    TransferDirective(object_id=x.obj_id, runtime_us=x.runtime)
                    for x in t.prefetch_after
                ),
            )
            for t in chain.tasks
        ),
        final_locations=dict(chain.final_locations),
        fast_memory_capacity=chain.fast_memory_capacity,
        backing_memory_capacity=chain.backing_memory_capacity,
        bandwidth_from_slow=chain.bandwidth_from_slow,
        bandwidth_to_slow=chain.bandwidth_to_slow,
    )
    return program
