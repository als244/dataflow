#!/usr/bin/env python
"""Simulator expectations for an ENGINE run: what the planner's
simulator-verified schedule says one training step should take, before
any GPU touches it.

    python tools/predict_step.py --preset gpt2_124m --ga-rounds 64 \
        --budget 14 --hw 5090 --steps 10000

Prints the plan's makespan (predicted s/step and tok/s), the per-group
compute sums against it (how much overlap the schedule found), peak fast
memory, recompute decisions, the most expensive tasks, and the ETA for
--steps. Roofline seeds come from ShapedHardware — pick a profile with
--hw or override single knobs. The REFERENCE (pure-torch twin) leg has
no simulator model; this predicts the engine only.
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


def main() -> int:
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run import presets as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--ga-rounds", type=int, default=None)
    ap.add_argument("--budget", type=float, default=14.0,
                    help="device fast-memory budget (GiB)")
    ap.add_argument("--hw", choices=sorted(HW_PROFILES), default="5090")
    ap.add_argument("--tflops", type=float, default=None,
                    help="override peak bf16 TFLOPs")
    ap.add_argument("--bw", type=float, default=None,
                    help="override memory bandwidth (GB/s)")
    ap.add_argument("--pcie", type=float, default=None,
                    help="override host link (GB/s)")
    ap.add_argument("--steps", type=int, default=None,
                    help="also print the ETA for this many steps")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = P.resolve_preset(args.preset)
    if args.ga_rounds:
        cfg = replace(cfg, grad_accum_rounds=args.ga_rounds)
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

    fam = resolve_family(cfg)
    recompute = not args.no_recompute
    planned = plan_program(
        fam.lower(cfg, hw=hw),
        fast_memory_capacity=int(args.budget * 1024 ** 3),
        recompute=recompute,
        build_variant=(partial(lower_variant, fam, cfg, hw)
                       if recompute else None),
    )

    tokens_step = cfg.tokens * cfg.grad_accum_rounds
    step_s = planned.makespan_us / 1e6
    tasks = list(planned.program.tasks)
    by_group: dict[str, float] = {}
    for t in tasks:
        by_group[t.group] = by_group.get(t.group, 0.0) + t.runtime_us
    compute_sum = sum(by_group.values()) / 1e6
    levels = planned.recompute_levels or {}
    n_recompute = sum(1 for v in levels.values() if v)

    print(f"preset {args.preset}  family {fam.name}  ga "
          f"{cfg.grad_accum_rounds}  tokens/step {tokens_step:,}")
    print(f"hw {args.hw}: {hw.peak_bf16_tflops:.0f} TF bf16 x "
          f"{hw.matmul_eff:.2f} eff, {hw.mem_bw_gbs:.0f} GB/s, "
          f"pcie {hw.pcie_gbs:.0f} GB/s   budget {args.budget:g} GiB")
    print(f"predicted step: {step_s:.2f} s   -> "
          f"{tokens_step / step_s:,.0f} tok/s   ({len(tasks)} tasks)")
    parts = "  ".join(f"{g} {v / 1e6:.2f}s" for g, v in
                      sorted(by_group.items(), key=lambda kv: -kv[1]))
    print(f"compute sums: {parts}   (serial {compute_sum:.2f}s; "
          f"overlap+stall factor x{step_s / max(compute_sum, 1e-9):.2f})")
    print(f"peak fast {planned.peak_fast_bytes / 1024**3:.2f} GiB   "
          f"recompute {n_recompute}/{len(levels) or 0} rewritable "
          f"activations")
    from dataflow_training.lowering.flops import flop_report

    rep = flop_report(cfg, planned.program)
    eff, hwf, allin = rep.per_step()
    print(f"model flops/step: eff {eff / 1e12:.1f} TF  hw {hwf / 1e12:.1f} TF"
          f"  opt {rep.optimizer / 1e12:.2f} TF  all-in {allin / 1e12:.1f} TF")
    print(f"expected throughput: eff {eff / planned.makespan_us / 1e6:.1f}"
          f" TF/s  hw {hwf / planned.makespan_us / 1e6:.1f} TF/s  "
          f"all-in {allin / planned.makespan_us / 1e6:.1f} TF/s")
    worst = sorted(tasks, key=lambda t: -t.runtime_us)[:args.top]
    for t in worst:
        print(f"  top task {t.runtime_us / 1e3:8.2f} ms  {t.id}")
    if args.steps:
        total_h = args.steps * step_s / 3600
        print(f"ETA {args.steps} steps: {total_h:.1f} h "
              f"({total_h / 24:.2f} days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
