#!/usr/bin/env python
"""Throughput + sim-fidelity sweep orchestrator (portable across boxes).

Reuses the shipped bench tools' OWN per-cell functions so every number is
identical to `tools/bench/predict_step.py` / `measure_step.py`, but emits one
structured JSONL record per cell (+ a combined CSV) instead of a printed table.

Modes:
  predict          roofline sim (CPU, instant)  -> combo_row(measured=False)
  predict-measured sim on H100-PROFILED costs    -> combo_row(measured=True)   [needs GPU]
  measure          real engine run per cell       -> run_cell(measured_plan=True) [needs GPU]

predict/predict-measured sweep the cross-product of --seq/--t-round/--t-step/--budget.
measure takes an explicit --cells JSON list [{seq,t_round,t_step,budget}, ...]
so the prioritized subset is exact and one daemon serves the whole run.

Geometry contract (from the tools): seq | t_round (batch=t_round/seq) and
t_round | t_step (ga=t_step/t_round); violating cells are recorded as skips.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import replace
from itertools import product

def _find_root(start):
    """Walk up until the repo root (has src/dataflow_training + tools/bench),
    so this script runs unchanged wherever it lives under the repo."""
    d = start
    while d != os.path.dirname(d):
        if (os.path.isdir(os.path.join(d, "src", "dataflow_training"))
                and os.path.isdir(os.path.join(d, "tools", "bench"))):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found from " + start)


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "tools", "bench"),   # predict_step / measure_step
          os.path.join(ROOT, "src"),              # dataflow_training
          ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import predict_step as PS                               # noqa: E402
import measure_step as MS                               # noqa: E402
from dataflow_training.run import presets as P          # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402


def geom_ok(seq, t_round, t_step):
    return t_round % seq == 0 and t_step % t_round == 0


def base_cfg(preset, opt):
    base = P.resolve_preset(preset)
    if opt:
        base = replace(base, opt_policy=opt)
    return base


def cfg_for(base, seq, t_round, t_step):
    return replace(base, seq_len=seq, grad_accum_rounds=t_step // t_round,
                   batch=t_round // seq)


def emit(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def run_predict(args, measured):
    base = base_cfg(args.preset, args.opt)
    fam = resolve_family(base)
    hw = PS.HW_PROFILES[args.hw]
    ov = {}
    if args.tflops:
        ov["peak_bf16_tflops"] = args.tflops
    if args.bw:
        ov["mem_bw_gbs"] = args.bw
    if args.pcie:
        ov["pcie_gbs"] = args.pcie
    if ov:
        hw = replace(hw, **ov)
    profile_cache: dict = {}
    seqs = [int(x) for x in args.seq.split(",")]
    trs = [int(x) for x in args.t_round.split(",")]
    tss = [int(x) for x in args.t_step.split(",")]
    buds = [float(x) for x in args.budget.split(",")]
    # "unlimited" plans with no host ceiling, so each row reports what the plan
    # actually WANTS in peak backing — the demand the allowance policy scales
    backs = [None if x.strip().lower() in ("unlimited", "none", "")
             else float(x) for x in str(args.backing_gib).split(",")]
    mode = "predict-measured" if measured else "predict"
    n = 0
    with open(args.out, "a") as fh:
        for seq, tr, ts, bud, back in product(seqs, trs, tss, buds, backs):
            meta = dict(mode=mode, opt=args.opt, preset=args.preset,
                        seq=seq, t_round=tr, t_step=ts, budget=bud, hw=args.hw,
                        backing=back, ts_epoch=time.time())
            if not geom_ok(seq, tr, ts):
                emit(fh, {**meta, "skip": "geometry: need seq|t_round|t_step"})
                continue
            cfg = cfg_for(base, seq, tr, ts)
            t0 = time.time()
            try:
                row = PS.combo_row(fam, cfg, hw, bud, measured=measured,
                                   recompute=True, profile_cache=profile_cache,
                                   backing_gib=back)
                # The allowance is set by policy, so what matters is not how
                # much the plan "wanted" (that is only defined when host memory
                # is free) but whether the ceiling BINDS here, and what relief
                # would buy. Re-planning once with more room gives the local
                # slope of throughput against host memory — a shadow price at
                # this operating point rather than an assumed level.
                if back:
                    row["binding"] = bool(row["backing_gib"] >= back * 0.999)
                    try:
                        more = PS.combo_row(fam, cfg, hw, bud, measured=measured,
                                            recompute=True,
                                            profile_cache=profile_cache,
                                            backing_gib=back * HOST_PROBE)
                        row["host_marginal_gain"] = round(
                            (more["tok_s"] - row["tok_s"]) / row["tok_s"], 4)
                    except (ValueError, KeyError):
                        row["host_marginal_gain"] = None
                emit(fh, {**meta, **row, "wall_s": round(time.time() - t0, 3)})
            except Exception as exc:  # infeasible / plan failure = a result
                emit(fh, {**meta, "infeasible": str(exc).splitlines()[0][:120],
                          "wall_s": round(time.time() - t0, 3)})
            n += 1
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            print(f"[{mode} {args.opt}] {n} seq{seq} tr{tr} ts{ts} b{bud:g} "
                  f"k{'inf' if back is None else format(back, 'g')}"
                  f"  peakRSS={rss}MB", flush=True)
    print(f"DONE {mode} {args.opt}: {n} cells -> {args.out}")


def run_cell_backed(client, cfg, budget, backing, steps, data_mode, recipe):
    """One measured cell. Unlike the shipped helper this plans WITH the host
    allowance the run will actually have, so predicted and executed plans agree
    (a plan blind to the slab keeps contexts the slab cannot hold)."""
    from dataflow_training.lowering.flops import flop_report
    from dataflow_training.run.driver import plan_at_budget, run_engine

    planned = plan_at_budget(cfg, budget, measured=True, backing_gib=backing)
    eff, hwf = flop_report(cfg, planned.program).per_step()
    res = run_engine(client, cfg, recipe, MS.cell_pipeline(cfg, data_mode), steps,
                     budget_gib=budget, backing_gib=backing, seed=11,
                     log=MS.quiet_log)
    tail = res.step_wall_s[MS.WARMUP_STEPS:] or res.step_wall_s
    meas_s = sum(tail) / len(tail)
    tokens_step = cfg.max_tokens * cfg.grad_accum_rounds
    pred_s = planned.makespan_us / 1e6
    levels = planned.recompute_levels or {}
    return {"seq": cfg.seq_len, "t_round": cfg.max_tokens,
            "ga": cfg.grad_accum_rounds, "tokens_step": tokens_step,
            "budget": budget, "backing": backing,
            "pred_s": pred_s, "meas_s": meas_s,
            "ratio": meas_s / pred_s if pred_s else float("nan"),
            "tok_s": tokens_step / meas_s,
            "eff_tfs": eff / meas_s / 1e12, "hw_tfs": hwf / meas_s / 1e12,
            "recompute": sum(1 for v in levels.values() if v),
            "rewritable": len(levels),
            "peak_backing_gib": planned.peak_backing_bytes / 1024 ** 3}


def run_measure(args):
    from dataflow_training.run.driver import engine_client
    from dataflow_training.run.recipe import Recipe
    base = base_cfg(args.preset, args.opt)
    cells = json.load(open(args.cells))
    default_back = float(str(args.backing_gib).split(",")[0])
    # the slab is fixed when the client boots, so cells are grouped by the host
    # allowance they were selected at and each group gets its own server
    groups = {}
    for c in cells:
        groups.setdefault(float(c.get("backing", default_back)), []).append(c)
    recipe = Recipe(peak_lr=args.peak_lr, min_lr=args.peak_lr / 10,
                    warmup_steps=max(1, args.steps // 3), total_steps=args.steps)
    n = 0
    with open(args.out, "a") as fh:
        for backing in sorted(groups):
            print(f"[measure {args.opt}] backing {backing:g} GiB "
                  f"({len(groups[backing])} cells)", flush=True)
            with engine_client(backing_gib=backing, log=MS.quiet_log) as client:
                for c in groups[backing]:
                    seq, tr, ts = c["seq"], c["t_round"], c["t_step"]
                    bud = float(c["budget"])
                    meta = dict(mode="measure", opt=args.opt, preset=args.preset,
                                seq=seq, t_round=tr, t_step=ts, budget=bud,
                                backing=backing, steps=args.steps,
                                ts_epoch=time.time())
                    if c.get("spines"):
                        meta["spines"] = c["spines"]
                    if not geom_ok(seq, tr, ts):
                        emit(fh, {**meta, "skip": "geometry"})
                        continue
                    cfg = cfg_for(base, seq, tr, ts)
                    t0 = time.time()
                    try:
                        row = run_cell_backed(client, cfg, bud, backing,
                                              args.steps, args.data, recipe)
                        emit(fh, {**meta, **row,
                                  "wall_s": round(time.time() - t0, 3)})
                        print(f"[measure {args.opt}] {n+1} seq{seq} tr{tr} ts{ts} "
                              f"b{bud:g} k{backing:g}  meas {row['meas_s']:.2f}s "
                              f"pred {row['pred_s']:.2f}s ratio {row['ratio']:.2f}  "
                              f"{row['eff_tfs']:.0f}effTF {row['tok_s']:,.0f}tok/s",
                              flush=True)
                    except Exception as exc:
                        emit(fh, {**meta, "failed": str(exc).splitlines()[0][:120],
                                  "wall_s": round(time.time() - t0, 3)})
                        print(f"[measure {args.opt}] {n+1} seq{seq} tr{tr} ts{ts} "
                              f"b{bud:g} k{backing:g}  FAILED: "
                              f"{str(exc).splitlines()[0][:70]}", flush=True)
                    for entry in client.list_programs():
                        client.unregister_program(entry["prog_id"])
                    client.wipe("all", force=True)
                    n += 1
    print(f"DONE measure {args.opt}: {n} cells -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["predict", "predict-measured", "measure"])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--seq", default="1024,2048,4096,8192")
    ap.add_argument("--t-round", dest="t_round", default="8192,16384,32768,65536")
    ap.add_argument("--t-step", dest="t_step", default="65536,131072,262144")
    ap.add_argument("--budget", default="4,8,16,32,64")
    ap.add_argument("--hw", default="5090")            # roofline seed base
    ap.add_argument("--tflops", type=float, default=None, help="roofline peak bf16 TF override")
    ap.add_argument("--bw", type=float, default=None, help="roofline mem bw GB/s override")
    ap.add_argument("--pcie", type=float, default=None, help="roofline host link GB/s override")
    ap.add_argument("--backing-gib", dest="backing_gib", default="130",
                    help="host allowance in GiB; comma-separated to sweep, or "
                         "'unlimited' to let each plan report what it wants")
    ap.add_argument("--cells", default=None, help="measure: JSON list of cells")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--data", default=None)
    ap.add_argument("--peak-lr", dest="peak_lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "measure":
        assert args.cells, "--cells required for measure mode"
        run_measure(args)
    else:
        run_predict(args, measured=(args.mode == "predict-measured"))


if __name__ == "__main__":
    sys.exit(main())
