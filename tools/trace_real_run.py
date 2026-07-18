#!/usr/bin/env python
"""Sim/engine integrity check: a few REAL training steps through the
daemon -> the full webapp bundle, measured vs simulated side by side.

The non-invasive twin of the nsys rig: every run already records a
RunTrace inside the engine; the run verb returns it on request
(trace=True). This drives K steps of a preset on a real daemon, keeps
the LAST step's trace (steady state — warm kernels, settled pools),
and writes under --out:

    <stem>.program.json     bare program (core IR)
    <stem>.annotated.json   planner-annotated program
    <stem>.webapp.json      DataflowProgram v1 (webapp upload)
    <stem>.measured.json    dataflow-measured-run/v1: measured
                            EventLog + memory trace + summary, and the
                            sim's EventLog/summary for the same plan

plus a one-line real-vs-sim parity summary (task coverage + makespan).

    python tools/trace_real_run.py --preset smoke --steps 3 --out examples/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def wire_intervals_to_dicts(intervals: list) -> list[dict]:
    """[[task_id, track, start, end], ...] -> webapp EventLog entries."""
    return [{"task_id": tid, "start": start, "end": end, "track": track}
            for tid, track, start, end in intervals]


def measured_log_from_trace(trace: dict) -> dict:
    return {
        "task_intervals": wire_intervals_to_dicts(trace["intervals"]),
        "events": [],
        "peak_fast_memory_bytes": trace["peak_fast_bytes"],
        "memory_trace": [{"t": t, "fast_bytes_by_band": {"other": used}}
                         for t, used in trace["memory_trace"]],
    }


def sim_log_for(program) -> dict:
    from dataflow.core.convert import to_sim_chain
    from dataflow_sim.engine.simulator import run as sim_run

    log = sim_run(to_sim_chain(program), snapshots=False,
                  memory_trace=True)
    return {
        "task_intervals": [
            {"task_id": iv.task_id, "start": iv.start, "end": iv.end,
             "track": iv.track} for iv in log.task_intervals],
        "events": [],
        "peak_fast_memory_bytes": log.peak_fast_memory_bytes,
        "memory_trace": [{"t": p.t,
                          "fast_bytes_by_band": dict(p.fast_bytes_by_band)}
                         for p in log.memory_trace],
    }


def log_makespan_us(log: dict) -> float:
    return max((iv["end"] for iv in log["task_intervals"]), default=0.0)


def parity_line(measured: dict, sim: dict) -> str:
    real_ids = {iv["task_id"] for iv in measured["task_intervals"]}
    sim_ids = {iv["task_id"] for iv in sim["task_intervals"]}
    return (f"tasks real {len(real_ids)} / sim {len(sim_ids)} "
            f"(missing-in-sim {len(real_ids - sim_ids)}, "
            f"missing-in-real {len(sim_ids - real_ids)}); "
            f"makespan real {log_makespan_us(measured):.0f}us vs "
            f"sim {log_makespan_us(sim):.0f}us")


def put_step_rounds(client, feed, rounds: int, step: int) -> int:
    """Put one step's token/target rounds; returns the step's valid
    count (the global denominator)."""
    valid = 0
    for r in range(rounds):
        tok, tgt = feed(step * rounds + r)
        valid += int((tgt >= 0).sum())
        client.put_object(f"tokens_0_{r}", tok.numpy().tobytes())
        client.put_object(f"targets_0_{r}", tgt.numpy().tobytes())
    return valid


def capture_run(client, cfg, recipe, feed, steps: int, *,
                budget_gib: float, seed: int = 11, log=print) -> dict:
    """K real steps with trace capture; returns {program, annotated,
    trace (last step), losses}."""
    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.run.presets import cfg_dict, resolver_family
    from dataflow_training.run.driver import init_model
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    fam = resolve_family(cfg)
    bare = fam.lower(cfg)
    planned = plan_program(bare,
                           fast_memory_capacity=int(budget_gib * 1024 ** 3))
    cd = cfg_dict(cfg)
    init_model(client, resolver_family(cfg), cd, seed=seed)
    R = cfg.grad_accum_rounds
    valid0 = put_step_rounds(client, feed, R, 0)
    reg = client.register_program(
        program_to_dict(planned.program),
        resolver={"kind": "model_family",
                  "family": resolver_family(cfg), "cfg": cd,
                  "hyper": recipe.hyper_spec()})
    missing = reg["bindings"]["missing_inputs"]
    if missing:
        raise RuntimeError(f"unbound inputs: {missing}")
    last_trace = None
    losses = []
    fetch = [f"loss_0_{r}" for r in range(R)]
    for step in range(steps):
        valid = valid0 if step == 0 else put_step_rounds(client, feed,
                                                         R, step)
        out = client.run(reg["prog_id"],
                         args={"step": step, "valid_rows": valid},
                         fetch=fetch, trace=(step == steps - 1))
        if out.get("state") != "done":
            raise RuntimeError(f"step {step}: {out}")
        losses.append(sum(out["fetched"][k] for k in fetch))
        if "trace" in out:
            last_trace = out["trace"]
        log(f"[trace_real_run] step {step}: loss {losses[-1]:.4f}")
    if last_trace is None:
        raise RuntimeError("no trace returned — daemon predates the "
                           "trace run option?")
    return {"program": bare, "annotated": planned.program,
            "trace": last_trace, "losses": losses}


def main() -> int:
    from dataflow.core.convert import to_webapp_program
    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.run import presets as P
    from dataflow_training.run.driver import daemon_client
    from dataflow_training.data.fineweb import make_feed
    from dataflow_training.run.recipe import Recipe

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--slab", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=Path("examples"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = (P.smoke_preset() if args.preset == "smoke"
           else P.resolve_preset(args.preset))
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=1,
                    total_steps=args.steps)
    with daemon_client(slab_gib=args.slab, log=print) as client:
        cap = capture_run(client, cfg, recipe, make_feed(cfg.tokens),
                          args.steps, budget_gib=args.budget)

    measured = measured_log_from_trace(cap["trace"])
    sim = sim_log_for(cap["annotated"])
    stem = args.name or f"trace-{args.preset}"
    args.out.mkdir(parents=True, exist_ok=True)
    outs = {
        f"{stem}.program.json": program_to_dict(cap["program"]),
        f"{stem}.annotated.json": program_to_dict(cap["annotated"]),
        f"{stem}.webapp.json": to_webapp_program(cap["annotated"]),
        f"{stem}.measured.json": {
            "format": "dataflow-measured-run/v1",
            "meta": {"preset": args.preset, "steps": args.steps,
                     "budget_gib": args.budget,
                     "losses": cap["losses"],
                     "traced_step": args.steps - 1},
            "log": measured,
            "sim_log": sim,
        },
    }
    for fname, payload in outs.items():
        (args.out / fname).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}/{stem}.{{program,annotated,webapp,measured}}.json")
    print(f"[parity] {parity_line(measured, sim)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
