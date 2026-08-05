"""Planning integration: annotate core Programs via dataflow_sim.

This is the ONE module boundary through which policy/recompute planning
happens. The runtime never sees planners; lowering never sees the simulator's
internals. Swapping the policy later means swapping ``policy_fn`` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from dataflow.core import Program
from dataflow.core.convert import apply_chain_annotations, to_sim_chain

BuildVariant = Callable[[Mapping[str, int]], Program]
PhysicalExtent = Callable[[Program], int]
ReplanForCapacity = Callable[[int], "PlannedProgram"]

# Fixed program-wide allowance for CUDA context, loaded framework/kernel
# modules, persistent streams/handles, and small allocator effects. Geometry
# feedback is accounted separately through ``packing_reserve_bytes``.
DEFAULT_PROGRAM_LEEWAY_BYTES = 3 << 29  # 1.5 GiB


@dataclass(frozen=True)
class PlannedProgram:
    program: Program                 # annotated, ready for the runtime
    makespan_us: float               # simulator-verified makespan of the plan
    peak_fast_bytes: int
    recompute_levels: dict[str, int]
    peak_backing_bytes: int = 0      # host-side (backing) demand of the plan
    # per-direction PCIe summary of the simulated schedule:
    # {"from_slow"|"to_slow": {"bytes", "busy_us", "n"}} — utilization =
    # busy_us / makespan_us (the webapp's link panels, as numbers)
    transfer_stats: dict = field(default_factory=dict)
    diagnostics: Any = None          # policy diagnostics (policy-specific)
    recompute_result: Any = None     # sim RecomputePlanResult when recompute ran
    program_leeway_bytes: int = 0
    max_task_workspace_bytes: int = 0
    object_memory_capacity: int | None = None
    user_memory_capacity: int | None = None
    packing_reserve_bytes: int = 0
    static_extent_bytes: int | None = None
    static_extent_replans: int = 0


def transfer_summary(log: Any, program: Program) -> dict:
    """Schedule summary from the sim's intervals: per-direction PCIe
    totals (task_id "direction:OBJ[#n]"; each interval moves the whole
    object) plus the compute track's busy and recompute-task time —
    idle = makespan - compute_busy_us (the webapp's panels, as
    numbers)."""
    sizes = program.object_sizes()
    recompute_ids = {t.id for t in program.tasks if t.group == "recompute"}
    out: dict = {d: {"bytes": 0, "busy_us": 0.0, "n": 0}
                 for d in ("from_slow", "to_slow")}
    out["compute_busy_us"] = 0.0
    out["recompute_us"] = 0.0
    for iv in log.task_intervals:
        if iv.track == "compute":
            out["compute_busy_us"] += iv.end - iv.start
            if iv.task_id in recompute_ids:
                out["recompute_us"] += iv.end - iv.start
            continue
        if iv.track not in ("from_slow", "to_slow"):
            continue
        obj = iv.task_id.split(":", 1)[1].split("#", 1)[0]
        s = out[iv.track]
        s["bytes"] += sizes.get(obj, 0)
        s["busy_us"] += iv.end - iv.start
        s["n"] += 1
    return out


def _to_sim_rewrites(program: Program) -> list[Any]:
    from dataflow_sim.workloads.common.recompute import (
        RecomputeOption as SimOption,
        RecomputeRewrite as SimRewrite,
    )

    return [
        SimRewrite(
            object_id=rw.object_id,
            f_task_id=rw.f_task_id,
            r_task_id=rw.r_task_id,
            options=tuple(
                SimOption(
                    level=o.level,
                    saved_bytes=o.saved_bytes,
                    recompute_us=o.recompute_us,
                    label=o.label,
                )
                for o in rw.options
            ),
            f_compute_block_key=rw.f_compute_block_key,
            r_compute_block_key=rw.r_compute_block_key,
            group_key=rw.group_key,
        )
        for rw in program.recompute_rewrites
    ]


def strip_plan_annotations(program: Program) -> Program:
    """Recover the priced bare variant from one annotated plan.

    Replanning static geometry must preserve the selected task/recompute ABIs
    while removing only residency decisions. Initial fast copies duplicated
    from backing by PressureFit are likewise planner annotations.
    """
    backing_ids = {
        value.id for value in program.initial_objects if value.location == "backing"
    }
    initial = tuple(
        value
        for value in program.initial_objects
        if not (value.location == "fast" and value.id in backing_ids)
    )
    tasks = tuple(
        replace(
            task,
            releases_after=(),
            offload_after=(),
            prefetch_after=(),
        )
        for task in program.tasks
    )
    return replace(program, initial_objects=initial, tasks=tasks)


def fit_static_extent(
    planned: PlannedProgram,
    *,
    user_memory_capacity: int,
    physical_extent: PhysicalExtent,
    fixed_program_leeway_bytes: int,
    fallback_replan_for_capacity: ReplanForCapacity | None = None,
    maximum_replans: int = 8,
    minimum_reserve_step_bytes: int = 64 << 20,
    preplace: str = "task0",
) -> PlannedProgram:
    """Constrain a selected plan to one executor's static arena geometry.

    PressureFit bounds dynamic object residency. A static arena physicalizer
    may need a larger contiguous extent. Measure that exact extent and convert
    any excess into a monotonically increasing packing reserve. The common
    path reruns only PressureFit over the already selected/priced task variant.
    If that variant cannot fit the reduced capacity and
    ``fallback_replan_for_capacity`` is supplied, the caller's complete
    planning search is retried. Profiling is never repeated.
    """
    if user_memory_capacity <= 0:
        raise ValueError("user_memory_capacity must be positive")
    if fixed_program_leeway_bytes < 0:
        raise ValueError("fixed_program_leeway_bytes must be nonnegative")
    if maximum_replans < 0 or minimum_reserve_step_bytes <= 0:
        raise ValueError("static-extent retry bounds must be positive")

    from dataflow_sim.core.validate import ValidationError
    from dataflow_sim.engine.errors import SimulationCapacityError

    current = planned
    selected_levels = dict(planned.recompute_levels)
    selected_recompute_result = planned.recompute_result
    packing_reserve = 0
    for attempt in range(maximum_replans + 1):
        max_workspace = max(
            (task.workspace_bytes for task in current.program.tasks),
            default=0,
        )
        extent = int(physical_extent(current.program))
        if extent < 0:
            raise ValueError("physical extent must be nonnegative")
        required = extent + max_workspace + fixed_program_leeway_bytes
        if required <= user_memory_capacity:
            return replace(
                current,
                recompute_levels=selected_levels,
                recompute_result=selected_recompute_result,
                program_leeway_bytes=fixed_program_leeway_bytes,
                max_task_workspace_bytes=max_workspace,
                user_memory_capacity=user_memory_capacity,
                packing_reserve_bytes=packing_reserve,
                static_extent_bytes=extent,
                static_extent_replans=attempt,
            )
        if attempt == maximum_replans:
            break
        excess = required - user_memory_capacity
        packing_reserve += max(excess, minimum_reserve_step_bytes)
        planning_capacity = user_memory_capacity - packing_reserve
        if planning_capacity <= fixed_program_leeway_bytes + max_workspace:
            break
        bare = strip_plan_annotations(current.program)
        try:
            current = plan_program(
                bare,
                fast_memory_capacity=planning_capacity,
                backing_capacity=bare.backing_memory_capacity,
                recompute=False,
                preplace=preplace,
                program_leeway_bytes=fixed_program_leeway_bytes,
            )
            # The task ABIs remain fixed, but the prior recompute search's
            # annotated chain is no longer the plan being returned.
            selected_recompute_result = None
        except (SimulationCapacityError, ValidationError) as error:
            if isinstance(error, ValidationError) and not str(error).startswith(
                "forced-footprint-exceeds-fast_memory_capacity:"
            ):
                raise
            if fallback_replan_for_capacity is None:
                raise
            current = fallback_replan_for_capacity(planning_capacity)
            selected_levels = dict(current.recompute_levels)
            selected_recompute_result = current.recompute_result

    raise ValueError(
        "static-extent admission is infeasible after "
        f"{maximum_replans} replans: user_budget={user_memory_capacity}, "
        f"fixed_leeway={fixed_program_leeway_bytes}, "
        f"max_task_workspace={max_workspace}, "
        f"packing_reserve={packing_reserve}"
    )


def simulate_program(program: Program, *, snapshots: bool = False, memory_trace: bool = False) -> Any:
    """Run the simulator on an (annotated) program; returns the sim EventLog."""
    from dataflow_sim.engine.simulator import run

    return run(to_sim_chain(program), snapshots=snapshots, memory_trace=memory_trace)


def plan_program(
    program: Program,
    *,
    fast_memory_capacity: int | None = None,
    backing_capacity: int | None = None,
    recompute: bool = False,
    build_variant: BuildVariant | None = None,
    max_iters: int = 8,
    pressurefit_prefetch_rules: tuple[str, ...] | None = None,
    max_wall_s: float | None = None,
    preplace: str = "task0",
    program_leeway_bytes: int = 0,
) -> PlannedProgram:
    """Annotate a bare program with PressureFit (+ optional recompute planning).

    ``fast_memory_capacity`` overrides the program's own capacity when given.
    With ``recompute=True``, ``build_variant(levels) -> Program`` must re-lower
    the program for a recompute-level assignment (the program's
    ``recompute_rewrites`` supply the options); the returned PlannedProgram
    carries the chosen levels and the re-lowered annotated program.

    ``preplace`` defaults to ``"task0"`` here — unlike the simulator's own
    ``"greedy"`` default — because the runtime REALIZES initial fast
    placement with synchronous uploads before the chain's clock starts:
    every pre-placed byte beyond task 0's needs is exposed wall time the
    simulator never charged (measured 0.32-0.42 s/step at 8B scale). With
    ``"task0"`` those bytes travel as planned prefetches instead, charged
    by the sim and overlapped with early compute. Pass ``"greedy"`` to
    reproduce legacy plans.
    """
    from dataclasses import replace as dc_replace

    from dataflow_sim.engine.simulator import run
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    cap = fast_memory_capacity if fast_memory_capacity is not None else program.fast_memory_capacity
    if program_leeway_bytes < 0:
        raise ValueError("program_leeway_bytes must be nonnegative")

    def to_chain(prog: Program) -> Any:
        """Program -> sim chain, carrying the backing ceiling when set —
        the SIM enforces it (over-backing schedules fail verification;
        no graceful stall mechanism, by design)."""
        chain = to_sim_chain(prog)
        if backing_capacity is not None:
            chain = dc_replace(chain, backing_memory_capacity=backing_capacity)
        return chain

    def policy_fn(chain: Any) -> Any:
        return apply_pressurefit_policy(
            chain, fast_memory_capacity=cap, preplace=preplace,
            prefetch_rules=pressurefit_prefetch_rules,
            program_leeway_bytes=program_leeway_bytes,
        )

    def memory_terms(prog: Program) -> tuple[int, int | None]:
        maximum_workspace = max(
            (task.workspace_bytes for task in prog.tasks),
            default=0,
        )
        object_capacity = (
            None
            if cap is None
            else cap - maximum_workspace - program_leeway_bytes
        )
        return maximum_workspace, object_capacity

    if recompute:
        if build_variant is None:
            raise ValueError("recompute=True requires build_variant(levels) -> Program")
        if not program.recompute_rewrites:
            raise ValueError("recompute=True but program.recompute_rewrites is empty")
        from dataflow_sim.planning.recompute import plan_with_recompute

        variants: dict[tuple[tuple[str, int], ...], Program] = {}

        def build_variant_chain(levels: Mapping[str, int]) -> Any:
            variant = build_variant(levels)
            variants[tuple(sorted(levels.items()))] = variant
            return to_chain(variant)

        result = plan_with_recompute(
            build_variant_chain,
            _to_sim_rewrites(program),
            policy_fn,
            max_iters=max_iters,
            max_wall_s=max_wall_s,
        )
        chosen = variants.get(tuple(sorted(result.levels.items()))) or build_variant(result.levels)
        annotated = apply_chain_annotations(chosen, result.chain)
        log = run(result.chain, snapshots=False)
        maximum_workspace, object_capacity = memory_terms(chosen)
        return PlannedProgram(
            program=annotated,
            makespan_us=result.makespan_us,
            peak_fast_bytes=log.peak_fast_memory_bytes,
            peak_backing_bytes=log.peak_backing_memory_bytes,
            transfer_stats=transfer_summary(log, annotated),
            recompute_levels=dict(result.levels),
            recompute_result=result,
            program_leeway_bytes=program_leeway_bytes,
            max_task_workspace_bytes=maximum_workspace,
            object_memory_capacity=object_capacity,
            user_memory_capacity=cap,
        )

    bare_chain = to_chain(program)
    annotated_chain = policy_fn(bare_chain)
    log = run(annotated_chain, snapshots=False)
    annotated = apply_chain_annotations(program, annotated_chain)
    makespan = max((iv.end for iv in log.task_intervals), default=0.0)
    maximum_workspace, object_capacity = memory_terms(program)
    return PlannedProgram(
        program=annotated,
        makespan_us=makespan,
        peak_fast_bytes=log.peak_fast_memory_bytes,
        peak_backing_bytes=log.peak_backing_memory_bytes,
        transfer_stats=transfer_summary(log, annotated),
        recompute_levels={},
        program_leeway_bytes=program_leeway_bytes,
        max_task_workspace_bytes=maximum_workspace,
        object_memory_capacity=object_capacity,
        user_memory_capacity=cap,
    )
