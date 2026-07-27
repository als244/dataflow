#!/usr/bin/env python
"""Throughput + sim-fidelity sweep orchestrator (portable across boxes).

Reuses the shipped bench tools' OWN per-cell functions so every number is
identical to `tools/bench/predict_step.py` / `measure_step.py`, but emits one
structured JSONL record per cell (+ a combined CSV) instead of a printed table.

Modes:
  predict          roofline sim (CPU, instant)  -> combo_row(measured=False)
  predict-measured sim on GPU-PROFILED costs; each feasible row's OWN plan
                   (annotated program + prog_id + pred_s) is also saved as a
                   gzipped artifact under --plans — predict is the ONLY
                   stage that plans                                [needs GPU]
  measure          run each cell's SAVED plan on the real engine — measure
                   performs NO planning; the registered prog_id must equal
                   the artifact's, and pred_s is reported from the artifact

predict/predict-measured sweep the cross-product of --seq/--t-round/--t-step/--budget.
measure takes an explicit --cells JSON list [{seq,t_round,t_step,budget}, ...]
so the prioritized subset is exact and one daemon serves the whole run.

Geometry contract (from the tools): seq | t_round (batch=t_round/seq) and
t_round | t_step (ga=t_step/t_round); violating cells are recorded as skips.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import sys
import threading
import time

import torch
from dataclasses import replace
from itertools import product

# how much extra host memory the counterfactual plan is offered when pricing
# what more RAM would be worth. Deliberately NOT capped at what this box has:
# the question is whether a bigger machine would help, and capping it at the
# current size can only ever answer "no".
HOST_PROBE = 1.25


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


def cell_config(base, seq, t_round, t_step):
    # num_steps=1: cells are ONE-STEP programs, the exact shape run_engine
    # registers (one step slot per client.run) — so a predicted plan IS
    # runnable as-is
    return replace(base, seq_len=seq, grad_accum_rounds=t_step // t_round,
                   batch=t_round // seq, num_steps=1)


def emit(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


# --------------------------------------------------------- plan artifacts

def plan_artifact_path(plans_dir, opt, seq, tr, ts, budget, backing):
    return os.path.join(plans_dir, opt,
                        f"s{seq}_tr{tr}_ts{ts}_b{budget:g}_k{backing:g}"
                        f".json.gz")


def save_plan_artifact(path, planned):
    """Write one selected cell's plan as the artifact measure executes:
    the ANNOTATED program dict, its content-hash prog_id (the same
    function registration uses), and the prediction that describes it.
    Always overwrites — a stale artifact from older code would carry an
    old program, and measure would silently time the wrong plan. The
    device name is stamped in because a plan is priced from THIS box's
    profiled costs: executing another box's plan would produce a
    fidelity ratio that describes nothing (measure refuses)."""
    import socket

    from dataflow.core.jsonio import program_to_dict
    from dataflow.service.wire import program_content_id

    pd = program_to_dict(planned.program)
    art = {"prog_id": program_content_id(pd),
           "device": torch.cuda.get_device_name(0),
           "host": socket.gethostname(),
           "pred_s": planned.makespan_us / 1e6,
           "makespan_us": planned.makespan_us,
           "peak_fast_bytes": planned.peak_fast_bytes,
           "peak_backing_bytes": planned.peak_backing_bytes,
           "recompute_levels": dict(planned.recompute_levels or {}),
           "transfer_stats": planned.transfer_stats or {},
           "program": pd}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(art, fh)
    return art


def load_plan_artifact(path):
    """(artifact dict, reconstructed PlannedProgram) — the program comes
    back through the same jsonio the wire uses, so registering it hashes
    to the artifact's prog_id (the gate asserts exactly that)."""
    from dataflow.core.jsonio import program_from_dict
    from dataflow_training.lowering.planning import PlannedProgram

    with gzip.open(path, "rt") as fh:
        art = json.load(fh)
    planned = PlannedProgram(
        program=program_from_dict(art["program"]),
        makespan_us=art["makespan_us"],
        peak_fast_bytes=art["peak_fast_bytes"],
        recompute_levels=art["recompute_levels"],
        peak_backing_bytes=art["peak_backing_bytes"],
        transfer_stats=art["transfer_stats"])
    return art, planned


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
            cfg = cell_config(base, seq, tr, ts)
            t0 = time.time()
            try:
                planned = PS.plan_combo(fam, cfg, hw, bud, measured=measured,
                                        recompute=True,
                                        profile_cache=profile_cache,
                                        backing_gib=back)
                row = PS.combo_row_from_plan(cfg, bud, planned)
                if measured and back is not None:
                    # this row's OWN plan is the artifact measure executes,
                    # saved by the same call that priced the row — predict
                    # is the only stage that plans. (Unlimited-allowance
                    # rows aren't measurable: no slab to run them in.)
                    save_plan_artifact(
                        plan_artifact_path(args.plans, args.opt, seq, tr,
                                           ts, bud, back), planned)
                # The allowance is set by policy, so what matters is not how
                # much the plan "wanted" (that is only defined when host memory
                # is free) but whether the ceiling BINDS here, and what relief
                # would buy. Re-planning once with more room gives the local
                # slope of throughput against host memory — a shadow price at
                # this operating point rather than an assumed level.
                if back:
                    row["binding"] = bool(row["backing_gib"] >= back * 0.999)
                if back and not row["binding"]:
                    # the ceiling never bound, so a plan given more room is the
                    # same plan — the gain is zero without planning to find out
                    row["host_marginal_gain"] = 0.0
                elif back:
                    try:
                        more = PS.combo_row(fam, cfg, hw, bud, measured=measured,
                                            recompute=True,
                                            profile_cache=profile_cache,
                                            backing_gib=back * HOST_PROBE)
                        row["host_marginal_gain"] = round(
                            (more["tok_s"] - row["tok_s"]) / row["tok_s"], 4)
                    except (ValueError, KeyError):
                        row["host_marginal_gain"] = None
                wall = round(time.time() - t0, 3)
                emit(fh, {**meta, **row, "wall_s": wall})
            except ValueError as exc:
                # the planner cannot fit this cell — that is a result
                emit(fh, {**meta, "infeasible": str(exc).splitlines()[0][:120],
                          "wall_s": round(time.time() - t0, 3)})
            except Exception as exc:
                # anything else is a fault in this harness, not a property of
                # the cell; record it as an error so it cannot be read as a
                # feasibility boundary
                emit(fh, {**meta, "error": f"{type(exc).__name__}: {exc}"[:160],
                          "wall_s": round(time.time() - t0, 3)})
                print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            n += 1
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            print(f"[{mode} {args.opt}] {n} seq{seq} tr{tr} ts{ts} b{bud:g} "
                  f"k{'inf' if back is None else format(back, 'g')}"
                  f"  wall {time.time() - t0:.1f}s  peakRSS={rss}MB",
                  flush=True)
    print(f"DONE {mode} {args.opt}: {n} cells -> {args.out}", flush=True)


class DevicePeak:
    """True peak device memory while a run is in flight.

    The engine's placed extent only covers what the planner reserved. A step
    also costs a CUDA context, cuBLAS and triton workspaces held by torch's
    caching allocator, and whatever else the process allocates along the way --
    none of which the engine sees. The driver's own free/total accounting sees
    all of it, so sample that instead and keep the high-water mark.

    Reports the absolute peak and the rise above the pre-run baseline; on a
    shared device the absolute figure includes other tenants, and the delta is
    the honest attribution."""

    def __init__(self, hz: float = 20.0):
        import torch
        self.torch = torch
        self.interval = 1.0 / hz
        self.baseline = self.used()
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def used(self) -> int:
        free, total = self.torch.cuda.mem_get_info()
        return total - free

    def watch(self) -> None:
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.used())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, self.used())


def run_cell_backed(client, cfg, budget, backing, steps, data_mode, recipe,
                    artifact, planned):
    """One measured cell, executing EXACTLY the plan the emit-plans stage
    wrote: measure performs NO planning. run_engine registers the loaded
    program and refuses unless the registered prog_id equals the
    artifact's, and pred_s is REPORTED from the artifact rather than
    recomputed — the measurement and its prediction can only ever
    describe the same program."""
    from dataflow_training.lowering.flops import flop_report
    from dataflow_training.run.driver import run_engine

    eff, hwf = flop_report(cfg, planned.program).per_step()
    with DevicePeak() as devmem:
        res = run_engine(client, cfg, recipe, MS.cell_pipeline(cfg, data_mode),
                         steps, budget_gib=budget, backing_gib=backing, seed=11,
                         planned=planned, expect_prog_id=artifact["prog_id"],
                         log=MS.quiet_log)
    # What the device actually gave this program, so the plan's memory model can
    # be checked rather than trusted: a run is only "within budget" if the
    # engine's own reserved extent says so.
    pools = client.engine_status().get("program_pools", [])
    reserved = max((p.get("fast_extent_bytes", 0) for p in pools), default=0)
    # what torch's caching allocator held for this cell: kernel workspaces and
    # scratch the engine never sees, and the bulk of the gap between its extent
    # and the device's own peak
    torch_reserved = torch.cuda.max_memory_reserved()
    tail = res.step_wall_s[MS.WARMUP_STEPS:] or res.step_wall_s
    meas_s = sum(tail) / len(tail)
    tokens_step = cfg.max_tokens * cfg.grad_accum_rounds
    pred_s = artifact["pred_s"]
    levels = artifact["recompute_levels"] or {}
    return {"seq": cfg.seq_len, "t_round": cfg.max_tokens,
            "ga": cfg.grad_accum_rounds, "tokens_step": tokens_step,
            "budget": budget, "backing": backing,
            "prog_id": artifact["prog_id"],
            "pred_s": pred_s, "meas_s": meas_s,
            "ratio": meas_s / pred_s if pred_s else float("nan"),
            "tok_s": tokens_step / meas_s,
            "eff_tfs": eff / meas_s / 1e12, "hw_tfs": hwf / meas_s / 1e12,
            "recompute": sum(1 for v in levels.values() if v),
            "rewritable": len(levels),
            "peak_backing_gib": artifact["peak_backing_bytes"] / 1024 ** 3,
            "planned_fast_gib": artifact["peak_fast_bytes"] / 1024 ** 3,
            "engine_extent_gib": reserved / 1024 ** 3,
            "device_peak_gib": devmem.peak / 1024 ** 3,
            "device_baseline_gib": devmem.baseline / 1024 ** 3,
            "device_peak_delta_gib": (devmem.peak - devmem.baseline) / 1024 ** 3,
            "torch_reserved_gib": torch_reserved / 1024 ** 3,
            "within_budget": bool(reserved <= budget * 1024 ** 3)}


def run_measure(args):
    from dataflow_training.run.driver import engine_client
    from dataflow_training.run.recipe import Recipe
    base = base_cfg(args.preset, args.opt)
    cells = json.load(open(args.cells))
    default_back = float(str(args.backing_gib).split(",")[0])
    # measure executes ONLY saved plans: every cell's artifact must
    # already exist AND have been priced on THIS device — a missing or
    # foreign artifact is a harness error to fix before any GPU time is
    # spent, not a per-cell "failed" row
    here = torch.cuda.get_device_name(0)
    missing, foreign = [], []
    for c in cells:
        if not geom_ok(c["seq"], c["t_round"], c["t_step"]):
            continue
        path = plan_artifact_path(args.plans, args.opt, c["seq"],
                                  c["t_round"], c["t_step"],
                                  float(c["budget"]),
                                  float(c.get("backing", default_back)))
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue
        art, planned_unused = load_plan_artifact(path)
        if art.get("device", here) != here:
            foreign.append(f"{os.path.basename(path)} ({art['device']})")
    if missing:
        raise SystemExit(
            f"measure: {len(missing)} cell(s) have no plan artifact under "
            f"{args.plans} (e.g. {missing[:3]}) — re-run the predict stage "
            f"(it saves each feasible row's plan); measure does not plan")
    if foreign:
        raise SystemExit(
            f"measure: {len(foreign)} plan(s) were priced on a DIFFERENT "
            f"device than this box's {here} (e.g. {foreign[:2]}) — a "
            f"fidelity ratio against another box's costs describes "
            f"nothing; re-run the predict stage here")
    # the slab is fixed when the client boots, so cells are grouped by the host
    # allowance they were selected at and each group gets its own server
    groups = {}
    for c in cells:
        groups.setdefault(float(c.get("backing", default_back)), []).append(c)
    recipe = Recipe(peak_lr=args.peak_lr, min_lr=args.peak_lr / 10,
                    warmup_steps=max(1, args.steps // 3), total_steps=args.steps)
    # RESUME: rows are appended and flushed per cell, so a run that was killed
    # part way has kept everything it finished. Without this the restart
    # re-measures those cells and appends a SECOND row for each, which reads as
    # data rather than as a repeat -- expensive on a frontier pass that takes
    # hours, and silently wrong afterwards.
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if "meas_s" in r or "failed" in r:
                done.add((r["seq"], r["t_round"], r["t_step"], r["budget"]))
    if done:
        print(f"[measure {args.opt}] resuming: {len(done)} cells already "
              f"recorded, skipping them", flush=True)
    n = 0
    with open(args.out, "a") as fh:
        for backing in sorted(groups):
            print(f"[measure {args.opt}] backing {backing:g} GiB "
                  f"({len(groups[backing])} cells)", flush=True)
            with engine_client(backing_gib=backing, log=MS.quiet_log) as client:
                for c in groups[backing]:
                    seq, tr, ts = c["seq"], c["t_round"], c["t_step"]
                    bud = float(c["budget"])
                    if (seq, tr, ts, bud) in done:
                        continue
                    meta = dict(mode="measure", opt=args.opt, preset=args.preset,
                                seq=seq, t_round=tr, t_step=ts, budget=bud,
                                backing=backing, steps=args.steps,
                                ts_epoch=time.time())
                    if c.get("spines"):
                        meta["spines"] = c["spines"]
                    if not geom_ok(seq, tr, ts):
                        emit(fh, {**meta, "skip": "geometry"})
                        continue
                    cfg = cell_config(base, seq, tr, ts)
                    t0 = time.time()
                    try:
                        art, planned = load_plan_artifact(
                            plan_artifact_path(args.plans, args.opt, seq,
                                               tr, ts, bud, backing))
                        row = run_cell_backed(client, cfg, bud, backing,
                                              args.steps, args.data, recipe,
                                              art, planned)
                        emit(fh, {**meta, **row,
                                  "wall_s": round(time.time() - t0, 3)})
                        print(f"[measure {args.opt}] {n+1} seq{seq} tr{tr} ts{ts} "
                              f"b{bud:g} k{backing:g}  meas {row['meas_s']:.2f}s "
                              f"pred {row['pred_s']:.2f}s ratio {row['ratio']:.2f}  "
                              f"{row['eff_tfs']:.0f}effTF {row['tok_s']:,.0f}tok/s  "
                              f"wall {time.time() - t0:.0f}s",
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
                    # the engine runs in this process, so its kernel workspaces
                    # sit in torch's caching allocator and outlive the cell that
                    # created them. Returning them to the driver keeps each
                    # cell's measured peak its own, and keeps a long sweep from
                    # accumulating scratch it will never reuse.
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    n += 1
    print(f"DONE measure {args.opt}: {n} cells -> {args.out}", flush=True)


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
    ap.add_argument("--plans", default=None,
                    help="plan-artifact directory (default: plans/ beside "
                         "--out; predict-measured writes it, measure reads it)")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--data", default=None)
    ap.add_argument("--peak-lr", dest="peak_lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.plans is None:
        args.plans = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "plans")
    if args.mode == "measure":
        assert args.cells, "--cells required for measure mode"
        run_measure(args)
    else:
        run_predict(args, measured=(args.mode == "predict-measured"))


if __name__ == "__main__":
    sys.exit(main())
