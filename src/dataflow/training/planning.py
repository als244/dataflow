"""Planning integration: annotate core Programs via dataflow_sim.

This is the ONE module boundary through which policy/recompute planning
happens. The runtime never sees planners; lowering never sees the simulator's
internals. Swapping the policy later means swapping ``policy_fn`` here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from dataflow.core import Program
from dataflow.core.convert import apply_chain_annotations, to_sim_chain

BuildVariant = Callable[[Mapping[str, int]], Program]


@dataclass(frozen=True)
class PlannedProgram:
    program: Program                 # annotated, ready for the runtime
    makespan_us: float               # simulator-verified makespan of the plan
    peak_fast_bytes: int
    recompute_levels: dict[str, int]
    diagnostics: Any = None          # policy diagnostics (policy-specific)
    recompute_result: Any = None     # sim RecomputePlanResult when recompute ran


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


def simulate_program(program: Program, *, snapshots: bool = False, memory_trace: bool = False) -> Any:
    """Run the simulator on an (annotated) program; returns the sim EventLog."""
    from dataflow_sim.engine.simulator import run

    return run(to_sim_chain(program), snapshots=snapshots, memory_trace=memory_trace)


def plan_program(
    program: Program,
    *,
    fast_memory_capacity: int | None = None,
    recompute: bool = False,
    build_variant: BuildVariant | None = None,
    max_iters: int = 8,
    pressurefit_schedules: tuple[str, ...] | None = None,
    max_wall_s: float | None = None,
    preplace: str = "task0",
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
    from dataflow_sim.engine.simulator import run
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    cap = fast_memory_capacity if fast_memory_capacity is not None else program.fast_memory_capacity

    def policy_fn(chain: Any) -> Any:
        return apply_pressurefit_policy(
            chain, fast_memory_capacity=cap, preplace=preplace,
            schedules=pressurefit_schedules,
        )

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
            return to_sim_chain(variant)

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
        return PlannedProgram(
            program=annotated,
            makespan_us=result.makespan_us,
            peak_fast_bytes=log.peak_fast_memory_bytes,
            recompute_levels=dict(result.levels),
            recompute_result=result,
        )

    bare_chain = to_sim_chain(program)
    annotated_chain = policy_fn(bare_chain)
    log = run(annotated_chain, snapshots=False)
    annotated = apply_chain_annotations(program, annotated_chain)
    makespan = max((iv.end for iv in log.task_intervals), default=0.0)
    return PlannedProgram(
        program=annotated,
        makespan_us=makespan,
        peak_fast_bytes=log.peak_fast_memory_bytes,
        recompute_levels={},
    )
