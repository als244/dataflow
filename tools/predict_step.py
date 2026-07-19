#!/usr/bin/env python
"""Simulator throughput expectations — the FIRST LINE OF ATTACK for
"how fast should this train?" before any long run.

Single point:

    python tools/predict_step.py --preset gpt2_124m --ga-rounds 64 \
        --budget 14 --hw 3090 --steps 10000

Sweep (a table row per budget x geometry combination):

    python tools/predict_step.py --preset gpt2_124m --hw 3090 \
        --budgets 16,8,4,2 --ga-batch 64x8,16x32,8x64 --measured

Per row: the plan's simulator-verified makespan (predicted s/step),
tok/s, EFFECTIVE and HARDWARE TFLOPs/s (lowering/flops.py — the same
subops the sim prices), peak fast bytes, and the planner's recompute
choice. ``--measured`` swaps roofline cost seeds for PROFILED task
costs (load_or_profile; disk-cached per geometry+kernel-set+device;
needs the GPU — ~1.3%-exact at ample budget on the calibration runs,
optimistic under tight-budget transfer pressure). Roofline mode is
CPU-only and instant. The REFERENCE (pure-torch twin) leg has no
simulator model; this predicts the engine.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataflow_training.lowering.shaped_program import ShapedHardware

# roofline profiles for the home fleet (bf16 dense peak, HBM/GDDR bw,
# host link); efficiencies stay the calibrated defaults
HW_PROFILES = {
    "5090": ShapedHardware(),
    "3090": ShapedHardware(peak_bf16_tflops=71.0, mem_bw_gbs=936.0,
                           pcie_gbs=25.0),
}


def lower_variant(fam, cfg, hw, levels):
    return fam.lower(cfg, hw=hw, recompute_levels=levels)


def measured_variant(fam, cfg, profiles, resolver, levels):
    from dataflow_training.run.profiling import apply_measured_costs

    return apply_measured_costs(fam.lower(cfg, recompute_levels=levels),
                                profiles, resolver)


def plan_combo(fam, cfg, hw, budget_gib: float, *, measured: bool,
               recompute: bool, profile_cache: dict):
    """One (cfg, budget) plan. ``profile_cache`` memoizes profiles per
    geometry (budget changes never re-profile; geometry changes do —
    task shapes differ)."""
    from dataflow_training.lowering.planning import plan_program

    cap = int(budget_gib * 1024 ** 3)
    if not measured:
        return plan_program(
            fam.lower(cfg, hw=hw), fast_memory_capacity=cap,
            recompute=recompute,
            build_variant=(partial(lower_variant, fam, cfg, hw)
                           if recompute else None))
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.run.profiling import apply_measured_costs, load_or_profile

    key = (cfg.grad_accum_rounds, cfg.batch, cfg.seq_len)
    if key not in profile_cache:
        backend = profile_cache.setdefault("_backend", CudaBackend())
        dims = fam.derive_dims(cfg)
        resolver = fam.build_resolver(dims)
        profile_cache[key] = (load_or_profile(fam.lower(cfg), resolver,
                                              backend), resolver)
    profiles, resolver = profile_cache[key]
    return plan_program(
        apply_measured_costs(fam.lower(cfg), profiles, resolver),
        fast_memory_capacity=cap, recompute=recompute,
        build_variant=(partial(measured_variant, fam, cfg, profiles,
                               resolver) if recompute else None))


def combo_row(fam, cfg, hw, budget: float, *, measured: bool,
              recompute: bool, profile_cache: dict) -> dict:
    from dataflow_training.lowering.flops import flop_report

    planned = plan_combo(fam, cfg, hw, budget, measured=measured,
                         recompute=recompute, profile_cache=profile_cache)
    rep = flop_report(cfg, planned.program)
    eff, hwf, allin = rep.per_step()
    step_s = planned.makespan_us / 1e6
    tokens_step = cfg.tokens * cfg.grad_accum_rounds
    levels = planned.recompute_levels or {}
    return {
        "ga": cfg.grad_accum_rounds, "batch": cfg.batch,
        "t_round": cfg.tokens, "tokens_step": tokens_step,
        "budget": budget, "step_s": step_s,
        "tok_s": tokens_step / step_s,
        "eff_tfs": eff / planned.makespan_us / 1e6,
        "hw_tfs": hwf / planned.makespan_us / 1e6,
        "all_tfs": allin / planned.makespan_us / 1e6,
        "peak_gib": planned.peak_fast_bytes / 1024 ** 3,
        "recompute": sum(1 for v in levels.values() if v),
        "rewritable": len(levels),
        "eff_flops": eff, "hw_flops": hwf,
    }


def print_table(rows, *, steps: int | None) -> None:
    hdr = (f"{'ga':>4} {'batch':>5} {'T_round':>8} {'tok/step':>9} "
           f"{'budget':>6} {'s/step':>7} {'tok/s':>8} {'effTF/s':>8} "
           f"{'hwTF/s':>7} {'peakGiB':>8} {'recomp':>7}")
    if steps:
        hdr += f" {'ETA_h':>6}"
    print(hdr)
    for r in rows:
        if "infeasible" in r:
            print(f"{r['ga']:>4} {r['batch']:>5} {r['t_round']:>8,} "
                  f"{r['tokens_step']:>9,} {r['budget']:>6g} "
                  f"  INFEASIBLE: {r['infeasible']}")
            continue
        line = (f"{r['ga']:>4} {r['batch']:>5} {r['t_round']:>8,} "
                f"{r['tokens_step']:>9,} {r['budget']:>6g} "
                f"{r['step_s']:>7.2f} {r['tok_s']:>8,.0f} "
                f"{r['eff_tfs']:>8.1f} {r['hw_tfs']:>7.1f} "
                f"{r['peak_gib']:>8.2f} "
                f"{str(r['recompute']) + '/' + str(r['rewritable']):>7}")
        if steps:
            line += f" {steps * r['step_s'] / 3600:>6.1f}"
        print(line)


def main() -> int:
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run import presets as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--ga-rounds", type=int, default=None,
                    help="single-point ga override")
    ap.add_argument("--batch", type=int, default=None,
                    help="single-point batch override (T_round scales)")
    ap.add_argument("--budget", type=float, default=14.0,
                    help="single-point device budget (GiB)")
    ap.add_argument("--budgets", default=None,
                    help="SWEEP: comma list of budgets, e.g. 16,8,4,2")
    ap.add_argument("--ga-batch", default=None,
                    help="SWEEP: comma list of gaXbatch combos, e.g. "
                         "64x8,16x32,8x64 (each row keeps its own "
                         "tokens/step = ga*batch*seq_len)")
    ap.add_argument("--hw", choices=sorted(HW_PROFILES), default="5090")
    ap.add_argument("--tflops", type=float, default=None,
                    help="override peak bf16 TFLOPs")
    ap.add_argument("--bw", type=float, default=None,
                    help="override memory bandwidth (GB/s)")
    ap.add_argument("--pcie", type=float, default=None,
                    help="override host link (GB/s)")
    ap.add_argument("--measured", action="store_true",
                    help="profiled task costs instead of roofline "
                         "(needs the GPU; disk-cached per geometry)")
    ap.add_argument("--steps", type=int, default=None,
                    help="also print the ETA column for this many steps")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--top", type=int, default=5,
                    help="single-point mode: most expensive tasks shown")
    args = ap.parse_args()

    base = P.resolve_preset(args.preset)
    hw = HW_PROFILES[args.hw]
    overrides = {}
    if args.tflops is not None:
        overrides["peak_bf16_tflops"] = args.tflops
    if args.bw is not None:
        overrides["mem_bw_gbs"] = args.bw
    if args.pcie is not None:
        overrides["pcie_gbs"] = args.pcie
    if overrides:
        hw = replace(hw, **overrides)
    fam = resolve_family(base)
    recompute = not args.no_recompute
    profile_cache: dict = {}

    combos = []
    if args.ga_batch:
        for tok in args.ga_batch.split(","):
            ga_s, b_s = tok.lower().split("x")
            combos.append((int(ga_s), int(b_s)))
    else:
        ga = args.ga_rounds or base.grad_accum_rounds
        b = args.batch or base.batch
        combos.append((ga, b))
    budgets = ([float(x) for x in args.budgets.split(",")]
               if args.budgets else [args.budget])

    sweep = bool(args.ga_batch or args.budgets)
    print(f"preset {args.preset}  family {fam.name}  hw {args.hw}: "
          f"{hw.peak_bf16_tflops:.0f} TF bf16 x {hw.matmul_eff:.2f}, "
          f"{hw.mem_bw_gbs:.0f} GB/s, pcie {hw.pcie_gbs:.0f} GB/s  "
          f"costs={'MEASURED (profiled)' if args.measured else 'roofline'}")
    rows = []
    for ga, b in combos:
        cfg = replace(base, grad_accum_rounds=ga, batch=b)
        for budget in budgets:
            try:
                rows.append(combo_row(fam, cfg, hw, budget,
                                      measured=args.measured,
                                      recompute=recompute,
                                      profile_cache=profile_cache))
            except ValueError as exc:
                # a combo the planner cannot fit is a RESULT, not a crash
                rows.append({"ga": ga, "batch": b,
                             "t_round": cfg.tokens,
                             "tokens_step": cfg.tokens * ga,
                             "budget": budget,
                             "infeasible": str(exc).splitlines()[0][:60]})
    print_table(rows, steps=args.steps)

    if not sweep:
        r = rows[0]
        planned = plan_combo(fam, replace(base,
                                          grad_accum_rounds=r["ga"],
                                          batch=r["batch"]),
                             hw, r["budget"], measured=args.measured,
                             recompute=recompute,
                             profile_cache=profile_cache)
        tasks = list(planned.program.tasks)
        by_group: dict[str, float] = {}
        for t in tasks:
            by_group[t.group] = by_group.get(t.group, 0.0) + t.runtime_us
        parts = "  ".join(f"{g} {v / 1e6:.2f}s" for g, v in
                          sorted(by_group.items(), key=lambda kv: -kv[1]))
        print(f"compute sums: {parts}   ({len(tasks)} tasks)")
        print(f"model flops/step: eff {r['eff_flops'] / 1e12:.1f} TF  "
              f"hw {r['hw_flops'] / 1e12:.1f} TF")
        for t in sorted(tasks, key=lambda t: -t.runtime_us)[:args.top]:
            print(f"  top task {t.runtime_us / 1e3:8.2f} ms  {t.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
