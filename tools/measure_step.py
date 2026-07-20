#!/usr/bin/env python
"""REAL throughput sweeps — the measured twin of tools/predict_step.py.

Same grid interface (T_round-speaking geometry, budgets, seq_lens); each
cell RUNS the engine for --steps steps through one shared daemon
(programs unregistered and the store wiped between cells, so device
extents don't accumulate) and reports the warmed measurement next to
the simulator's prediction for the same plan:

    python tools/measure_step.py --preset gpt2_124m --hw 3090 \
        --t-rounds 8192,32768,65536 --tokens-step 524288 \
        --budgets 16,4 --steps 12 --data doc

Per row: predicted s/step (the plan's simulator-verified makespan),
measured s/step (mean over the warmed tail), their ratio, tok/s, and
effective/hardware TFLOPs/s from the measured wall time. One daemon
serves every cell (the store is wiped between cells); a cell that fails
to plan or run reports as a row, not a crash. ``--measured-plan`` makes
the prediction column use PROFILED task costs (disk-cached) instead of
roofline.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

WARMUP_STEPS = 3      # excluded from the measured tail


def quiet_log(*args, **kwargs) -> None:
    pass


def cell_feed(cfg, data_mode: str):
    from dataflow_training.data.fineweb import make_doc_feed, make_feed

    if data_mode == "doc":
        return make_doc_feed(cfg.tokens, cfg.seq_len)
    return make_feed(cfg.tokens)


def run_cell(client, cfg, budget: float, steps: int, data_mode: str,
             recipe, measured_plan: bool) -> dict:
    from dataflow_training.lowering.flops import flop_report
    from dataflow_training.run.driver import plan_at_budget, run_engine

    planned = plan_at_budget(cfg, budget, measured=measured_plan)
    rep = flop_report(cfg, planned.program)
    eff, hwf = rep.per_step()
    res = run_engine(client, cfg, recipe, cell_feed(cfg, data_mode),
                     steps, budget_gib=budget, seed=11, log=quiet_log)
    tail = res.step_wall_s[WARMUP_STEPS:] or res.step_wall_s
    meas_s = sum(tail) / len(tail)
    tokens_step = cfg.tokens * cfg.grad_accum_rounds
    pred_s = planned.makespan_us / 1e6
    return {
        "seq": cfg.seq_len, "t_round": cfg.tokens,
        "ga": cfg.grad_accum_rounds, "tokens_step": tokens_step,
        "budget": budget, "pred_s": pred_s, "meas_s": meas_s,
        "ratio": meas_s / pred_s if pred_s else float("nan"),
        "tok_s": tokens_step / meas_s,
        "eff_tfs": eff / meas_s / 1e12,
        "hw_tfs": hwf / meas_s / 1e12,
        "recompute": sum(1 for v in (planned.recompute_levels or {})
                         .values() if v),
    }


def main() -> int:
    from dataflow_training.run import presets as P
    from dataflow_training.run.driver import daemon_client
    from dataflow_training.run.recipe import Recipe

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--plugin", action="append", default=None,
                    help="external-family plugin module(s) to load")
    ap.add_argument("--opt", choices=["adamw", "muon"], default=None)
    ap.add_argument("--t-round", type=int, default=None)
    ap.add_argument("--t-rounds", default=None)
    ap.add_argument("--tokens-step", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--seq-lens", default=None)
    ap.add_argument("--budget", type=float, default=14.0)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--steps", type=int, default=12,
                    help="steps per cell (first 3 excluded as warmup)")
    ap.add_argument("--data", choices=["block", "doc"], default="block")
    ap.add_argument("--slab", type=float, default=16.0)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--measured-plan", action="store_true",
                    help="prediction column from PROFILED task costs")
    ap.add_argument("--hw", default=None,
                    help="display only (the run measures the real box)")
    args = ap.parse_args()

    if args.plugin:
        from dataflow_training.model_families.families import load_plugins
        load_plugins(args.plugin)
    base = P.resolve_preset(args.preset)
    if args.opt:
        base = replace(base, opt_policy=args.opt)
    budgets = ([float(x) for x in args.budgets.split(",")]
               if args.budgets else [args.budget])
    seqs = ([int(x) for x in args.seq_lens.split(",")] if args.seq_lens
            else [args.seq_len or base.seq_len])
    t_rounds = ([int(x) for x in args.t_rounds.split(",")]
                if args.t_rounds
                else [args.t_round] if args.t_round else [base.tokens])
    tokens_step = args.tokens_step or (base.tokens
                                       * base.grad_accum_rounds)
    recipe = Recipe(peak_lr=args.peak_lr, min_lr=args.peak_lr / 10,
                    warmup_steps=max(1, args.steps // 3),
                    total_steps=args.steps)

    print(f"preset {args.preset}  data {args.data}  steps/cell "
          f"{args.steps} (warmup {WARMUP_STEPS})  prediction "
          f"{'MEASURED (profiled)' if args.measured_plan else 'roofline'}")
    hdr = (f"{'seq':>5} {'T_round':>8} {'ga':>4} {'tok/step':>9} "
           f"{'budget':>6} {'pred_s':>7} {'meas_s':>7} {'ratio':>6} "
           f"{'tok/s':>8} {'effTF/s':>8} {'hwTF/s':>7} {'recomp':>6}")
    print(hdr)
    with daemon_client(slab_gib=args.slab, log=quiet_log) as client:
        for seq in seqs:
            for tr in t_rounds:
                if tr % seq or tokens_step % tr:
                    print(f"{seq:>5} {tr:>8,}   SKIP: t_round must "
                          f"divide seq_len and tokens-step")
                    continue
                cfg = replace(base, seq_len=seq,
                              grad_accum_rounds=tokens_step // tr,
                              batch=tr // seq)
                for budget in budgets:
                    try:
                        r = run_cell(client, cfg, budget, args.steps,
                                     args.data, recipe,
                                     args.measured_plan)
                        print(f"{r['seq']:>5} {r['t_round']:>8,} "
                              f"{r['ga']:>4} {r['tokens_step']:>9,} "
                              f"{r['budget']:>6g} {r['pred_s']:>7.2f} "
                              f"{r['meas_s']:>7.2f} {r['ratio']:>6.2f} "
                              f"{r['tok_s']:>8,.0f} {r['eff_tfs']:>8.1f} "
                              f"{r['hw_tfs']:>7.1f} {r['recompute']:>6}",
                              flush=True)
                    except Exception as exc:
                        print(f"{seq:>5} {tr:>8,} {tokens_step // tr:>4} "
                              f"{tokens_step:>9,} {budget:>6g}   "
                              f"FAILED: {str(exc).splitlines()[0][:60]}",
                              flush=True)
                    for entry in client.list_programs():
                        client.unregister_program(entry["prog_id"])
                    client.wipe("all", force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
