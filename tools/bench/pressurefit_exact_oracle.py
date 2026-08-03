#!/usr/bin/env python3
"""Exhaustively solve tiny fixed-initial-placement TaskChains.

This development oracle is intentionally independent of PressureFit. It
enumerates one of ``none/release/offload/prefetch`` for every object after
every task, rejects invalid annotations through the public validator, and
selects the fastest chain accepted by the simulator. The search is exponential
and is therefore suitable only for small correctness and approximation-gap
studies.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
from typing import Literal

from dataflow_sim.core.schema import TaskChain, TransferTrigger
from dataflow_sim.core.validate import validate_chain
from dataflow_sim.engine.simulator import run


_Action = Literal["none", "release", "offload", "prefetch"]
_ACTIONS: tuple[_Action, ...] = ("none", "release", "offload", "prefetch")


@dataclass(frozen=True, slots=True)
class ExactPlanResult:
    """Complete exhaustive-search result for one fixed bare chain."""

    assignment_count: int
    valid_plan_count: int
    best_makespan_us: float
    peak_fast_memory_bytes: int
    peak_backing_memory_bytes: int
    best_chain: TaskChain


def _object_ids(chain: TaskChain) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for obj in chain.initial_memory:
        ordered.setdefault(obj.id, None)
    for task in chain.tasks:
        for output in task.outputs:
            ordered.setdefault(output.id, None)
    return tuple(ordered)


def _assert_bare(chain: TaskChain) -> None:
    if any(
        task.releases_after or task.offload_after or task.prefetch_after
        for task in chain.tasks
    ):
        raise ValueError("exact oracle requires a bare chain without directives")


def _annotate(
    bare: TaskChain,
    object_ids: tuple[str, ...],
    assignment: tuple[_Action, ...],
) -> TaskChain:
    width = len(object_ids)
    tasks = []
    for task_index, task in enumerate(bare.tasks):
        releases: list[str] = []
        offloads: list[TransferTrigger] = []
        prefetches: list[TransferTrigger] = []
        offset = task_index * width
        for object_index, object_id in enumerate(object_ids):
            action = assignment[offset + object_index]
            if action == "release":
                releases.append(object_id)
            elif action == "offload":
                offloads.append(TransferTrigger(object_id))
            elif action == "prefetch":
                prefetches.append(TransferTrigger(object_id))
        tasks.append(replace(
            task,
            releases_after=releases,
            offload_after=offloads,
            prefetch_after=prefetches,
        ))
    return replace(bare, tasks=tasks)


def _makespan_us(chain: TaskChain) -> tuple[float, int, int]:
    log = run(chain, snapshots=False)
    makespan = max((interval.end for interval in log.task_intervals), default=0.0)
    return (
        makespan,
        log.peak_fast_memory_bytes,
        log.peak_backing_memory_bytes,
    )


def find_exact_plan(
    bare: TaskChain,
    *,
    max_assignments: int = 1_000_000,
) -> ExactPlanResult:
    """Return the simulator-optimal annotation for a tiny bare chain.

    Initial placement is fixed exactly as supplied. The oracle does not add
    free/preplaced copies, so comparisons are meaningful only when another
    planner starts from the same physical initial-memory declaration.
    """
    _assert_bare(bare)
    object_ids = _object_ids(bare)
    slot_count = len(object_ids) * len(bare.tasks)
    assignment_count = len(_ACTIONS) ** slot_count
    if assignment_count > max_assignments:
        raise ValueError(
            "exact oracle assignment space exceeds limit: "
            f"4**{slot_count}={assignment_count:,} > {max_assignments:,}"
        )

    best_chain: TaskChain | None = None
    best_makespan = float("inf")
    best_peak_fast = 0
    best_peak_backing = 0
    valid_plan_count = 0
    for assignment in product(_ACTIONS, repeat=slot_count):
        candidate = _annotate(bare, object_ids, assignment)
        try:
            validate_chain(candidate)
            makespan, peak_fast, peak_backing = _makespan_us(candidate)
        except ValueError:
            continue
        valid_plan_count += 1
        if makespan < best_makespan:
            best_chain = candidate
            best_makespan = makespan
            best_peak_fast = peak_fast
            best_peak_backing = peak_backing

    if best_chain is None:
        raise ValueError(
            f"infeasible: no valid plan among {assignment_count:,} assignments"
        )
    return ExactPlanResult(
        assignment_count=assignment_count,
        valid_plan_count=valid_plan_count,
        best_makespan_us=best_makespan,
        peak_fast_memory_bytes=best_peak_fast,
        peak_backing_memory_bytes=best_peak_backing,
        best_chain=best_chain,
    )


def _json_payload(result: ExactPlanResult) -> dict:
    payload = asdict(result)
    payload["best_chain"] = asdict(result.best_chain)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively plan a tiny fixed-placement TaskChain",
    )
    parser.add_argument("chain", type=Path, help="bare TaskChain JSON file")
    parser.add_argument(
        "--max-assignments",
        type=int,
        default=1_000_000,
        help="reject a larger exponential search space (default: 1,000,000)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = find_exact_plan(
        TaskChain.load(args.chain),
        max_assignments=args.max_assignments,
    )
    rendered = json.dumps(_json_payload(result), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
