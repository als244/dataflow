"""nsys-trace ONE campaign cell executing its OWN saved plan.

    python trace_cell.py --results <results-dir> --opt adamw \
        --seq 8192 --t-round 32768 --t-step 131072 --budget 12 \
        --steps 5 --warmup 2 --out traces/

Why this exists. Comparing "what the simulator scheduled" against "what the
engine did" is only meaningful when BOTH describe the same program. The
campaign's own traces do not exist, and the traces in
poor_perf/updated_prof/ are from a different revision running a DIFFERENT
plan — comparing their schedules to a campaign plan compares two different
schedules and supports no conclusion (learned the hard way; see
FA3_CORRECTED_VS_FA3.md).

So this loads the plan artifact measure executed, hands it to run_engine with
``expect_prog_id`` (which refuses if the registered program hashes to anything
else), and brackets a profiler window around steady-state steps. The trace and
the plan are then the same program by construction, and
``pred_s`` comes from the artifact rather than being recomputed.

Run it UNDER nsys so the in-process daemon is captured:

    nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
         --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o <out>/<cell> python trace_cell.py ...

The capture range matters for more than trace size: per-task NVTX
annotation is OFF outside a profiled window (SwitchableAnnotator), and
without those ranges a trace cannot be attributed back to tasks at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stages"))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="campaign results dir (holds data/plans/<opt>/)")
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--preset", default="llama3_8b")
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--t-round", type=int, required=True)
    ap.add_argument("--t-step", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--backing", type=float, default=128.0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2,
                    help="steps before the capture window opens")
    ap.add_argument("--data", default=None,
                    help="data spec; None = the campaign default "
                         "(measure passes nothing)")
    ap.add_argument("--out", default=None, help="write a json summary here")
    a = ap.parse_args()

    import sweep as SW
    import measure_step as MS
    from dataflow_training.run.driver import engine_client, run_engine
    from dataflow_training.run.recipe import Recipe

    plans = os.path.join(a.results, "data", "plans")
    path = SW.plan_artifact_path(plans, a.opt, a.seq, a.t_round, a.t_step,
                                 a.budget, a.backing)
    if not os.path.exists(path):
        raise SystemExit(f"no plan artifact at {path} — measure never ran this cell")
    art, planned = SW.load_plan_artifact(path)
    cfg = SW.cell_config(SW.base_cfg(a.preset, a.opt), a.seq, a.t_round, a.t_step)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2, total_steps=a.steps)

    print(f"cell s{a.seq} tr{a.t_round} ts{a.t_step} b{a.budget:g}", flush=True)
    print(f"  plan {os.path.basename(path)}", flush=True)
    print(f"  prog_id {art['prog_id']}  pred_s {art['pred_s']:.3f}  "
          f"device {art.get('device')}", flush=True)
    print(f"  capture: steps {a.warmup}..{a.steps - 1} of {a.steps}", flush=True)

    t0 = time.perf_counter()
    with engine_client(backing_gib=a.backing, log=MS.quiet_log) as client:
        res = run_engine(client, cfg, recipe, MS.cell_pipeline(cfg, a.data),
                         a.steps, budget_gib=a.budget, backing_gib=a.backing,
                         seed=11, planned=planned,
                         expect_prog_id=art["prog_id"],
                         profile={"start": a.warmup, "stop": a.steps - 1},
                         log=MS.quiet_log)
    wall = time.perf_counter() - t0

    # steady-state only: the capture window's steps, not the warm-ups
    steady = res.step_wall_s[a.warmup:]
    meas = sorted(steady)[len(steady) // 2] if steady else float("nan")
    out = {"seq": a.seq, "t_round": a.t_round, "t_step": a.t_step,
           "budget": a.budget, "backing": a.backing, "opt": a.opt,
           "prog_id": art["prog_id"], "pred_s": art["pred_s"],
           "meas_s": meas, "ratio": meas / art["pred_s"],
           "captured_steps": [a.warmup, a.steps - 1],
           "step_wall_s": list(res.step_wall_s), "wall_s": wall}
    print(f"  pred {art['pred_s']:.3f}s  meas {meas:.3f}s  "
          f"ratio {meas / art['pred_s']:.3f}   (wall {wall:.0f}s)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  wrote {a.out}", flush=True)


main()
