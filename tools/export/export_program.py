"""Golden path: shaped program -> PressureFit (+ recompute) -> sim -> exports.

Works for ANY preset (`run.presets.resolve_preset` name — the full
table is docs/builtin_models.md). Produces, under --out (default
examples/):
    <name>.program.json            bare program (core IR)
    <name>.annotated.json          planner-annotated program (core IR)
    <name>.webapp.json             DataflowProgram v1 (upload at the webapp)
    <name>.summary.json            makespan / peak-fast / chosen recompute levels

Usage:
    python tools/export/export_program.py --preset llama3:tiny --fast-gib 0.0006
    python tools/export/export_program.py --preset llama3_8b --fast-gib 16 --recompute
    python tools/export/export_program.py --preset gpt2_124m --fast-gib 4 --recompute
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import replace
from pathlib import Path

from dataflow.core import save_program, validate_program
from dataflow.core.convert import to_webapp_program
from dataflow_training.lowering.planning import plan_program, simulate_program
from dataflow_training.model_families.families import load_plugins, resolve_family
from dataflow_training.run.presets import resolve_preset

GIB = 1024**3


def lower_variant(fam, cfg, levels):
    return fam.lower(cfg, recompute_levels=levels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="llama3:tiny",
                        help="any resolve_preset name (docs/builtin_models.md)")
    parser.add_argument("--plugin", action="append", default=None,
                        help="external-family plugin module(s) to load")
    parser.add_argument("--fast-gib", type=float, required=True)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad-accum-rounds", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("examples"))
    args = parser.parse_args()

    if args.plugin:
        load_plugins(explicit=args.plugin)
    cfg = resolve_preset(args.preset)
    overrides = {k: v for k, v in (("seq_len", args.seq_len),
                                   ("batch", args.batch),
                                   ("grad_accum_rounds", args.grad_accum_rounds))
                 if v is not None}
    if overrides:
        cfg = replace(cfg, **overrides)
    fam = resolve_family(cfg)

    cap = int(args.fast_gib * GIB)
    program = fam.lower(cfg)
    validate_program(program)

    t0 = time.perf_counter()
    planned = plan_program(
        program,
        fast_memory_capacity=cap,
        recompute=args.recompute,
        build_variant=functools.partial(lower_variant, fam, cfg)
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
