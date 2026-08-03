"""PressureFit policy.

Standalone planner built around deterministic greedy pressure reduction:
  1. derive schema-level facts from the TaskChain;
  2. choose initial residency and build one seed interval set from liveness
     anchors;
  3. reduce copies under a small portfolio of pressure views and deterministic
     cut scores;
  4. emit release/offload/prefetch triggers from each reduced interval set
     under four prefetch rules, with ordinary and clean-gap-
     coalesced forms;
  5. verify each annotated chain with the simulator, translating bounded
     capacity contradictions back into boundary pressure and re-reducing;
  6. return the fastest valid annotated chain.

The bounded portfolio exists because conservative prefetch headroom,
capacity-tight feasibility, transfer work, and overlap do not dominate one
another across all chains. Packed-fifo coordinates transfers backward from
their deadlines, packed-fit adds a pressure clamp, interval-entry extends
entries earlier where pressure permits, and latest-safe places transfers
independently. Simulator replay remains the physical arbiter.

The policy is name-agnostic. It uses object source availability, size, uses,
producer, and explicit mutation metadata.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from dataflow_sim.policies._common import (
    _compute_ideal_starts,
    _object_sizes,
    _object_uses_by_task_idx,
)
from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _build_facts,
)
from dataflow_sim.policies.pressurefit_aux.candidate import _verify_candidate
from dataflow_sim.policies.pressurefit_aux.diagnostics import (
    PressureFitCandidateDiagnostic,
    PressureFitDiagnostics,
    _candidate_diagnostic,
)
from dataflow_sim.policies.pressurefit_aux.seeds import (
    _copy_intervals,
    _initial_residency,
    _pressure_initial_placement,
)
from dataflow_sim.policies.pressurefit_aux.types import (
    _IntervalSet,
    _ResidencySpec,
    _PrefetchRuleSpec,
)
from dataflow_sim.policies.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.core.schema import TaskChain

# Prefetch rules in tie-break priority order. The second half repeats the four
# boundary-selection rules while allowing clean zero-width gaps to coalesce.
_PREFETCH_RULES: tuple[_PrefetchRuleSpec, ...] = (
    _PrefetchRuleSpec("packed-fifo", "packed-fifo"),
    _PrefetchRuleSpec("packed-fit", "packed-fit"),
    _PrefetchRuleSpec("interval-entry", "interval-entry"),
    _PrefetchRuleSpec("latest-safe", "latest-safe"),
    _PrefetchRuleSpec("packed-fifo-coalesced", "packed-fifo", True),
    _PrefetchRuleSpec("packed-fit-coalesced", "packed-fit", True),
    _PrefetchRuleSpec("interval-entry-coalesced", "interval-entry", True),
    _PrefetchRuleSpec("latest-safe-coalesced", "latest-safe", True),
)

# PressureFit deliberately races a conservative view that reserves one
# prefetch-destination boundary against a capacity-tight view that charges
# only required residency.  Each is reduced with two deterministic greedy
# cut scores; simulator replay chooses the physically fastest valid result.
_RESIDENCY_SPECS: tuple[_ResidencySpec, ...] = (
    _ResidencySpec("headroom-stall", True, "min-stall"),
    _ResidencySpec("headroom-transfer", True, "min-transfer"),
    _ResidencySpec("tight-stall", False, "min-stall"),
    _ResidencySpec("tight-transfer", False, "min-transfer"),
    _ResidencySpec("relaxed-stall", False, "min-stall", False),
)


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    makespan_us: float
    chain: TaskChain
    name: str


def pressurefit(
    bare: TaskChain,
    *,
    fast_memory_capacity: int | None = None,
    preplace: str = "greedy",
    prefetch_rules: tuple[str, ...] | None = None,
) -> tuple[TaskChain, PressureFitDiagnostics]:
    """Run the PressureFit algorithm and return its plan plus diagnostics.

    This is the implementation's algorithm-shaped spine: analyze the bare
    workload, construct seed residency, reduce one copy per residency
    strategy, realize and simulate every selected PrefetchRule, and return the
    valid candidate with minimum makespan. Physical repair remains inside
    candidate verification.

    ``preplace="greedy"`` fills spare initial fast memory; ``"task0"``
    preplaces only task 0 inputs and realizes later objects through charged
    prefetches. ``prefetch_rules`` optionally restricts evaluation to exact
    names from the built-in portfolio.
    """
    planning_start = time.perf_counter()
    if fast_memory_capacity is not None:
        bare = replace(bare, fast_memory_capacity=fast_memory_capacity)
    if preplace not in ("greedy", "task0"):
        raise ValueError(f"preplace must be 'greedy' or 'task0', got {preplace!r}")

    # Analyze(W, H).
    ideal = _compute_ideal_starts(bare)
    compute_lower_bound_us = (
        ideal[bare.tasks[-1].id] + bare.tasks[-1].runtime
        if bare.tasks
        else 0.0
    )
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)
    initial_compute = _pressure_initial_placement(
        bare, bare.fast_memory_capacity, sizes, uses_by_task, mode=preplace,
    )
    facts = _build_facts(bare)

    # Build the seed residency P0 and select the requested heuristic family.
    seed = _initial_residency(facts, initial_compute)
    prefetch_rule_specs = _PREFETCH_RULES
    if prefetch_rules is not None:
        # Planner trial mode: evaluate a subset of the named prefetch-rule
        # candidates (the full race only matters for plans that are kept).
        prefetch_rule_specs = tuple(
            rule for rule in _PREFETCH_RULES if rule.name in prefetch_rules
        )
        if not prefetch_rule_specs:
            raise ValueError(f"no known prefetch rules in {prefetch_rules!r}")
    results: list[_CandidateResult] = []
    candidate_diagnostics: list[PressureFitCandidateDiagnostic] = []
    first_error: Exception | None = None

    # For each configuration: reduce residency, realize transitions and
    # prefetches, emit a plan, and validate it with Simulate.
    for residency in _RESIDENCY_SPECS:
        base_fit = _copy_intervals(seed)
        try:
            _reduce_to_fit(
                facts,
                base_fit,
                bare.fast_memory_capacity,
                cut_score=residency.cut_score,
                prefetch_headroom=residency.prefetch_headroom,
                continue_headroom_cuts=residency.continue_headroom_cuts,
            )
        except Exception as error:
            if first_error is None:
                first_error = error
            continue

        (
            variant_results,
            variant_diagnostics,
            variant_error,
            reached_lower_bound,
        ) = _evaluate_prefetch_rules(
            bare,
            facts,
            base_fit,
            residency,
            prefetch_rule_specs,
            compute_lower_bound_us=compute_lower_bound_us,
        )
        results.extend(variant_results)
        candidate_diagnostics.extend(variant_diagnostics)
        if first_error is None and variant_error is not None:
            first_error = variant_error
        # Compute tasks are serialized, so their transfer-free completion time
        # is a global makespan lower bound. Once a verified candidate reaches
        # it, evaluating any remaining portfolio member cannot improve the
        # selected makespan or the stable first-candidate tie break.
        if reached_lower_bound:
            break

    if not results:
        assert first_error is not None
        raise first_error

    # Return the simulator-valid plan with minimum makespan.
    results.sort(key=lambda x: x.makespan_us)
    selected = results[0]
    selected_candidates = [
        replace(diag, selected=diag.name == selected.name)
        for diag in candidate_diagnostics
    ]
    diagnostics = PressureFitDiagnostics(
        planning_time_s=time.perf_counter() - planning_start,
        task_count=facts.n,
        object_count=len(facts.sizes),
        fast_memory_capacity=bare.fast_memory_capacity,
        candidate_count=len(selected_candidates),
        valid_candidate_count=sum(
            1 for diag in selected_candidates if diag.status == "valid"
        ),
        selected_candidate=selected.name,
        selected_makespan_us=selected.makespan_us,
        candidates=selected_candidates,
    )
    return selected.chain, diagnostics


def apply_pressurefit_policy(
    bare: TaskChain,
    *,
    fast_memory_capacity: int | None = None,
    preplace: str = "greedy",
    prefetch_rules: tuple[str, ...] | None = None,
) -> TaskChain:
    """Return only the annotated chain selected by :func:`pressurefit`."""
    chain, _diagnostics = pressurefit(
        bare,
        fast_memory_capacity=fast_memory_capacity,
        preplace=preplace,
        prefetch_rules=prefetch_rules,
    )
    return chain


def plan_pressurefit_policy(
    bare: TaskChain,
    *,
    fast_memory_capacity: int | None = None,
    preplace: str = "greedy",
    prefetch_rules: tuple[str, ...] | None = None,
) -> tuple[TaskChain, PressureFitDiagnostics]:
    """Return PressureFit's annotated chain and candidate diagnostics."""
    return pressurefit(
        bare,
        fast_memory_capacity=fast_memory_capacity,
        preplace=preplace,
        prefetch_rules=prefetch_rules,
    )


def _evaluate_prefetch_rules(
    bare: TaskChain,
    facts: _Facts,
    base_fit: _IntervalSet,
    residency: _ResidencySpec,
    prefetch_rules: tuple[_PrefetchRuleSpec, ...],
    *,
    compute_lower_bound_us: float,
) -> tuple[
    list[_CandidateResult],
    list[PressureFitCandidateDiagnostic],
    Exception | None,
    bool,
]:
    results: list[_CandidateResult] = []
    diagnostics: list[PressureFitCandidateDiagnostic] = []
    first_error: Exception | None = None
    reached_lower_bound = False

    for prefetch_rule in prefetch_rules:
        t0 = time.perf_counter()
        candidate_name = f"{residency.name}/{prefetch_rule.name}"
        try:
            makespan, annotated = _verify_candidate(
                bare,
                facts,
                base_fit,
                residency,
                prefetch_rule,
            )
            wall = time.perf_counter() - t0
            results.append(
                _CandidateResult(makespan, annotated, candidate_name),
            )
            diagnostics.append(_candidate_diagnostic(
                candidate_name,
                status="valid",
                wall_time_s=wall,
                makespan_us=makespan,
            ))
            if makespan == compute_lower_bound_us:
                reached_lower_bound = True
                break
        except Exception as e:
            wall = time.perf_counter() - t0
            if first_error is None:
                first_error = e
            diagnostics.append(_candidate_diagnostic(
                candidate_name,
                status="error",
                wall_time_s=wall,
                error=f"{type(e).__name__}: {e}",
            ))

    return results, diagnostics, first_error, reached_lower_bound
