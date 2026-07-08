"""Golden path: shaped program -> PressureFit (+ recompute) -> sim -> exports.

Produces, under --out (default examples/):
    <name>.program.json            bare program (core IR)
    <name>.annotated.json          planner-annotated program (core IR)
    <name>.webapp.json             DataflowProgram v1 (upload at the webapp)
    <name>.summary.json            makespan / peak-fast / chosen recompute levels

Usage:
    python tools/golden_path.py --config tiny --fast-gib 0.0006
    python tools/golden_path.py --config 8b --fast-gib 16 --recompute
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataflow.core import save_program, validate_program
from dataflow.core.jsonio import program_to_dict
from dataflow.core.convert import to_webapp_program
from dataflow.training.planning import plan_program, simulate_program
from dataflow.training.models.llama3 import ShapedLlamaConfig, build_shaped_llama3

GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["tiny", "8b"], default="tiny")
    parser.add_argument("--fast-gib", type=float, required=True)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum-rounds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("examples"))
    args = parser.parse_args()

    if args.config == "tiny":
        cfg = ShapedLlamaConfig.tiny()
    else:
        cfg = ShapedLlamaConfig.llama3_8b(
            seq_len=args.seq_len, batch=args.batch, grad_accum_rounds=args.grad_accum_rounds
        )

    cap = int(args.fast_gib * GIB)
    program = build_shaped_llama3(cfg)
    validate_program(program)

    t0 = time.perf_counter()
    planned = plan_program(
        program,
        fast_memory_capacity=cap,
        recompute=args.recompute,
        build_variant=(lambda levels: build_shaped_llama3(cfg, recompute_levels=levels))
        if args.recompute
        else None,
    )
    plan_s = time.perf_counter() - t0
    validate_program(planned.program)

    log = simulate_program(planned.program, memory_trace=True)
    makespan = max(iv.end for iv in log.task_intervals)
    assert makespan == planned.makespan_us, (makespan, planned.makespan_us)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = planned.program.name + (f"-{args.fast_gib:g}gib" + ("-recompute" if args.recompute else ""))
    save_program(program, args.out / f"{stem}.program.json")
    save_program(planned.program, args.out / f"{stem}.annotated.json")
    (args.out / f"{stem}.webapp.json").write_text(json.dumps(to_webapp_program(program), indent=2) + "\n")

    recompute_count = sum(1 for v in planned.recompute_levels.values() if v > 0)
    summary = {
        "name": planned.program.name,
        "fast_memory_capacity_bytes": cap,
        "task_count": len(planned.program.tasks),
        "object_count": len(program.object_sizes()),
        "makespan_us": planned.makespan_us,
        "peak_fast_bytes": planned.peak_fast_bytes,
        "peak_fast_gib": planned.peak_fast_bytes / GIB,
        "recompute_levels_chosen": recompute_count,
        "recompute_total_options": len(planned.recompute_levels),
        "planning_time_s": plan_s,
        "tokens_per_second": (
            float(program.metadata["primary_count"]) / (planned.makespan_us / 1e6)
        ),
    }
    (args.out / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/{stem}.{{program,annotated,webapp,summary}}.json")


if __name__ == "__main__":
    main()
