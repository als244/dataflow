"""Evidence-directed recompute selection layered above a residency policy.

Recompute and offload are competing regeneration channels for the same
activation bytes: offload spends tier-link, recompute spends the compute stream.
Which is cheaper for a given activation depends on the whole schedule (what
else is offloaded, weight traffic, stream backlogs), so no static ranking
works. This planner starts from "offload everything the policy wants"
(levels all 0), reads the simulator's stall/backlog report for the plan the
policy actually produced, and greedily converts the most-blamed activations
to recomputation — accepting each batch only if the simulated makespan
improves, and keeping the best plan seen.

The loop is workload-agnostic: it sees a chain-variant builder, a rewrite
table (object id -> discrete recompute options), and a policy function. It
never interprets model semantics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

from dataflow_sim.core.schema import TaskChain
from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.engine.stall_report import StallReport, build_stall_report
from dataflow_sim.workloads.common.recompute import RecomputeRewrite


@dataclass(frozen=True)
class RecomputeStep:
    """One accepted or rejected iteration, for diagnostics."""
    converted: tuple[str, ...]
    makespan_us: float
    accepted: bool


@dataclass(frozen=True)
class RecomputePlanResult:
    levels: dict[str, int]          # chosen option level per rewrite object
    chain: TaskChain                # annotated chain of the best plan
    makespan_us: float
    baseline_makespan_us: float     # levels all zero
    planning_time_s: float
    history: tuple[RecomputeStep, ...]
    report: StallReport             # report of the best plan


def plan_with_recompute(
    build_variant: Callable[[Mapping[str, int]], TaskChain],
    rewrites: list[RecomputeRewrite],
    policy_fn: Callable[[TaskChain], TaskChain],
    *,
    max_iters: int = 8,
    max_wall_s: float | None = None,
) -> RecomputePlanResult:
    """Greedy, simulator-verified recompute selection.

    Structure: evaluate a small fixed seed family (none / all / every-other,
    derived from the rewrite table) so the result never loses to a trivial
    choice, then run the evidence loop from the all-saved plan. Each
    iteration: blame activations for observed stalls and backlog, estimate
    net benefit of recomputing them (blame minus added compute), convert the
    best half of the positive-net candidates, replan, and accept only if the
    simulated makespan improves. Rejected batches are halved down to a
    single conversion before giving up. The returned plan is the best seen
    anywhere.

    Known limitation (deliberate, v1): blame is transfer-based. An
    activation that stays resident produces no stall or backlog evidence
    even when recomputing it would free pool headroom for other traffic —
    the seed family is what covers that regime today.
    """
    t0 = time.perf_counter()
    rewrites_by_obj = {rw.object_id: rw for rw in rewrites}

    def evaluate(levels: Mapping[str, int]):
        annotated = policy_fn(build_variant(levels))
        log = simulator_run(annotated, snapshots=False)
        report = build_stall_report(annotated, log)
        return annotated, report

    # The all-saved baseline can itself be infeasible (a save-all working set
    # that no annotation fits under the cap — common for large batch/grad-
    # accum programs). It must not be fatal: fall back to the seed family and
    # only raise if EVERY variant is infeasible.
    levels: dict[str, int] = {rw.object_id: 0 for rw in rewrites}
    history: list[RecomputeStep] = []
    baseline_error: Exception | None = None
    best: tuple[dict[str, int], TaskChain, object] | None = None
    try:
        chain, report = evaluate(levels)
        baseline = report.makespan_us
        best = (dict(levels), chain, report)
    except Exception as err:
        baseline_error = err
        baseline = float("inf")
        history.append(RecomputeStep(
            converted=("<baseline:infeasible>",), makespan_us=float("inf"), accepted=False,
        ))

    seeds = {
        "all": {rw.object_id: rw.options[-1].level for rw in rewrites},
        "half": {
            rw.object_id: (rw.options[-1].level if n % 2 == 0 else 0)
            for n, rw in enumerate(rewrites)
        },
    }
    greedy_start = best
    for name, seed_levels in seeds.items():
        try:
            seed_chain, seed_report = evaluate(seed_levels)
        except Exception:
            continue
        accepted = best is None or seed_report.makespan_us < best[2].makespan_us
        history.append(RecomputeStep(
            converted=(f"<seed:{name}>",),
            makespan_us=seed_report.makespan_us,
            accepted=accepted,
        ))
        if accepted:
            best = (dict(seed_levels), seed_chain, seed_report)

    if best is None:
        raise baseline_error if baseline_error is not None else RuntimeError(
            "recompute planning found no feasible variant"
        )

    def over_budget() -> bool:
        return max_wall_s is not None and (time.perf_counter() - t0) > max_wall_s

    # The evidence loop refines from the all-saved plan, where transfer
    # blame is most informative; when that baseline was infeasible, refine
    # from the best feasible seed instead. `best` keeps whatever won so far.
    current = greedy_start if greedy_start is not None else best
    for _ in range(max_iters):
        if over_budget():
            break
        candidates = _ranked_candidates(current[0], rewrites_by_obj, current[2])
        if not candidates:
            break
        batch = max(1, len(candidates) // 2)
        accepted = False
        while batch >= 1 and not over_budget():
            chosen = candidates[:batch]
            trial_levels = dict(current[0])
            for obj_id, _net in chosen:
                trial_levels[obj_id] = _next_level(rewrites_by_obj[obj_id], trial_levels[obj_id])
            try:
                trial_chain, trial_report = evaluate(trial_levels)
            except Exception:
                batch //= 2  # infeasible trial: shrink the batch and retry
                continue
            step = RecomputeStep(
                converted=tuple(obj for obj, _ in chosen),
                makespan_us=trial_report.makespan_us,
                accepted=trial_report.makespan_us < current[2].makespan_us,
            )
            history.append(step)
            if step.accepted:
                current = (trial_levels, trial_chain, trial_report)
                if trial_report.makespan_us < best[2].makespan_us:
                    best = current
                accepted = True
                break
            batch //= 2
        if not accepted:
            break

    levels, chain, report = best
    return RecomputePlanResult(
        levels=levels,
        chain=chain,
        makespan_us=report.makespan_us,
        baseline_makespan_us=baseline,
        planning_time_s=time.perf_counter() - t0,
        history=tuple(history),
        report=report,
    )


def _next_level(rewrite: RecomputeRewrite, current: int) -> int:
    for option in rewrite.options:
        if option.level > current:
            return option.level
    return current


def _ranked_candidates(
    levels: Mapping[str, int],
    rewrites_by_obj: Mapping[str, RecomputeRewrite],
    report: StallReport,
) -> list[tuple[str, float]]:
    """Objects worth converting, ranked by estimated net benefit (us)."""
    out: list[tuple[str, float]] = []
    for obj_id, rewrite in rewrites_by_obj.items():
        current = levels.get(obj_id, 0)
        nxt = _next_level(rewrite, current)
        if nxt == current:
            continue
        cur_opt = _option(rewrite, current)
        nxt_opt = _option(rewrite, nxt)
        benefit = (
            report.stall_by_object.get(obj_id, 0)
            + report.transfer_backlog_overlap.get(obj_id, 0)
        )
        cost = nxt_opt.recompute_us - cur_opt.recompute_us
        net = benefit - cost
        if net > 0:
            out.append((obj_id, net))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def _option(rewrite: RecomputeRewrite, level: int):
    for option in rewrite.options:
        if option.level == level:
            return option
    raise ValueError(f"rewrite {rewrite.object_id!r} has no level {level}")
