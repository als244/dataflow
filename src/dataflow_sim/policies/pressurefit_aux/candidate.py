"""Build and verify one PressureFit heuristic candidate."""
from __future__ import annotations

from dataflow_sim.core.schema import TaskChain
from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.policies.pressurefit_aux.core import _Facts, _pool_size
from dataflow_sim.policies.pressurefit_aux.emit import _emit_chain
from dataflow_sim.policies.pressurefit_aux.prefetch_rules import _apply_prefetch_rule
from dataflow_sim.policies.pressurefit_aux.physical_repair import (
    _PHYSICAL_REPAIR_LIMIT,
    _apply_physical_repair,
)
from dataflow_sim.policies.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.policies.pressurefit_aux.residency_refinement import (
    _extend_inbound_lead_time,
)
from dataflow_sim.policies.pressurefit_aux.seeds import _copy_intervals
from dataflow_sim.policies.pressurefit_aux.transitions import _build_transitions
from dataflow_sim.policies.pressurefit_aux.types import (
    _IntervalSet,
    _PrefetchRuleSpec,
    _ResidencySpec,
)


def _refine_residency(
    bare: TaskChain,
    facts: _Facts,
    intervals: _IntervalSet,
    residency: _ResidencySpec,
    prefetch_rule: _PrefetchRuleSpec,
    extra_pressure: list[int],
) -> None:
    """Apply a rule's optional residency refinement before transitions."""
    if prefetch_rule.kind != "interval-entry":
        return
    _extend_inbound_lead_time(
        facts,
        intervals,
        bare.fast_memory_capacity,
        bare.bandwidth_from_slow,
        extra_pressure,
        prefetch_headroom=residency.prefetch_headroom,
        coalesce_clean_gaps=prefetch_rule.coalesce_clean_gaps,
    )


def _reduce_residency(
    bare: TaskChain,
    facts: _Facts,
    intervals: _IntervalSet,
    residency: _ResidencySpec,
    prefetch_rule: _PrefetchRuleSpec,
    extra_pressure: list[int],
) -> None:
    """Re-reduce residency after simulator-discovered pressure."""
    _reduce_to_fit(
        facts,
        intervals,
        bare.fast_memory_capacity,
        extra_pressure,
        cut_score=residency.cut_score,
        prefetch_headroom=residency.prefetch_headroom,
        continue_headroom_cuts=residency.continue_headroom_cuts,
    )
    _refine_residency(
        bare,
        facts,
        intervals,
        residency,
        prefetch_rule,
        extra_pressure,
    )


def _realize_plan(
    bare: TaskChain,
    facts: _Facts,
    intervals: _IntervalSet,
    residency: _ResidencySpec,
    prefetch_rule: _PrefetchRuleSpec,
    extra_pressure: list[int],
) -> TaskChain:
    """Realize ``Transitions -> PrefetchRule -> Emit`` for one candidate."""
    transitions = _build_transitions(
        intervals,
        facts,
        coalesce_clean_gaps=prefetch_rule.coalesce_clean_gaps,
    )
    pool = (
        _pool_size(
            facts,
            intervals,
            prefetch_headroom=residency.prefetch_headroom,
        )
        if (
            prefetch_rule.kind == "packed-fit"
            and bare.fast_memory_capacity is not None
        )
        else None
    )
    prefetches = _apply_prefetch_rule(
        transitions,
        facts,
        bare.bandwidth_from_slow,
        prefetch_rule.kind,
        pool=pool,
        cap=bare.fast_memory_capacity,
        extra_pressure=extra_pressure,
        prefetch_headroom=residency.prefetch_headroom,
    )
    return _emit_chain(
        bare,
        transitions,
        prefetches,
        coalesce_clean_gaps=prefetch_rule.coalesce_clean_gaps,
    )


def _simulated_makespan_us(annotated: TaskChain) -> float:
    log = simulator_run(annotated, snapshots=False)
    return max(interval.end for interval in log.task_intervals)


def _verify_candidate(
    bare: TaskChain,
    facts: _Facts,
    base_fit: _IntervalSet,
    residency: _ResidencySpec,
    prefetch_rule: _PrefetchRuleSpec,
) -> tuple[float, TaskChain]:
    """Realize, simulate, and physically repair one heuristic candidate."""
    intervals = _copy_intervals(base_fit)
    extra_pressure = [0] * (facts.n + 1)
    _refine_residency(
        bare,
        facts,
        intervals,
        residency,
        prefetch_rule,
        extra_pressure,
    )

    for _ in range(_PHYSICAL_REPAIR_LIMIT):
        annotated = _realize_plan(
            bare,
            facts,
            intervals,
            residency,
            prefetch_rule,
            extra_pressure,
        )
        try:
            return _simulated_makespan_us(annotated), annotated
        except ValueError as error:
            if not _apply_physical_repair(
                error,
                bare,
                facts,
                intervals,
                extra_pressure,
            ):
                raise
            _reduce_residency(
                bare,
                facts,
                intervals,
                residency,
                prefetch_rule,
                extra_pressure,
            )

    annotated = _realize_plan(
        bare,
        facts,
        intervals,
        residency,
        prefetch_rule,
        extra_pressure,
    )
    return _simulated_makespan_us(annotated), annotated
