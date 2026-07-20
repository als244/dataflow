#!/usr/bin/env python
"""Simulator throughput expectations — the FIRST LINE OF ATTACK for
"how fast should this train?" before any long run.

Single point:

    python tools/bench/predict_step.py --preset gpt2_124m --ga-rounds 64 \
        --budget 14 --hw 3090 --steps 10000

Sweep (a table row per budget x geometry combination; geometry speaks
T_round — the round token budget — with ga derived from --tokens-step;
"batch" is internal arithmetic under varlen packing, never an input):

    python tools/bench/predict_step.py --preset gpt2_124m --hw 3090 \
        --budgets 16,8,4,2 --t-rounds 8192,32768,65536 \
        --tokens-step 524288 --measured

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

_ROOT = Path(__file__).resolve().parents[2]
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
               recompute: bool, profile_cache: dict,
               backing_gib: float | None = None):
    """One (cfg, budget) plan. ``profile_cache`` memoizes profiles per
    geometry (budget changes never re-profile; geometry changes do —
    task shapes differ). ``backing_gib`` sets the sim's host-side
    ceiling — over-backing plans fail verification (INFEASIBLE row)."""
    from dataflow_training.lowering.planning import plan_program

    cap = int(budget_gib * 1024 ** 3)
    bk = int(backing_gib * 1024 ** 3) if backing_gib else None
    if not measured:
        return plan_program(
            fam.lower(cfg, hw=hw), fast_memory_capacity=cap,
            backing_capacity=bk, recompute=recompute,
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
        fast_memory_capacity=cap, backing_capacity=bk, recompute=recompute,
        build_variant=(partial(measured_variant, fam, cfg, profiles,
                               resolver) if recompute else None))


def combo_row(fam, cfg, hw, budget: float, *, measured: bool,
              recompute: bool, profile_cache: dict,
              backing_gib: float | None = None) -> dict:
    from dataflow_training.lowering.flops import flop_report

    planned = plan_combo(fam, cfg, hw, budget, measured=measured,
                         recompute=recompute, profile_cache=profile_cache,
                         backing_gib=backing_gib)
    rep = flop_report(cfg, planned.program)
    eff, hwf = rep.per_step()
    step_s = planned.makespan_us / 1e6
    tokens_step = cfg.tokens * cfg.grad_accum_rounds
    levels = planned.recompute_levels or {}
    return {
        "seq": cfg.seq_len,
        "ga": cfg.grad_accum_rounds, "batch": cfg.batch,
        "t_round": cfg.tokens, "tokens_step": tokens_step,
        "budget": budget, "step_s": step_s,
        "tok_s": tokens_step / step_s,
        "eff_tfs": eff / planned.makespan_us / 1e6,
        "hw_tfs": hwf / planned.makespan_us / 1e6,
        "peak_gib": planned.peak_fast_bytes / 1024 ** 3,
        "backing_gib": planned.peak_backing_bytes / 1024 ** 3,
        "h2d_gb": (planned.transfer_stats.get("from_slow", {})
                   .get("bytes", 0) / 1e9),
        "h2d_pct": (planned.transfer_stats.get("from_slow", {})
                    .get("busy_us", 0.0) / planned.makespan_us * 100),
        "d2h_gb": (planned.transfer_stats.get("to_slow", {})
                   .get("bytes", 0) / 1e9),
        "d2h_pct": (planned.transfer_stats.get("to_slow", {})
                    .get("busy_us", 0.0) / planned.makespan_us * 100),
        "rc_pct": (planned.transfer_stats.get("recompute_us", 0.0)
                   / planned.makespan_us * 100),
        "idle_pct": max(0.0, 100 * (1 - planned.transfer_stats
                                    .get("compute_busy_us", 0.0)
                                    / planned.makespan_us)),
        "recompute": sum(1 for v in levels.values() if v),
        "rewritable": len(levels),
        "eff_flops": eff, "hw_flops": hwf,
    }


def print_table(rows, *, steps: int | None) -> None:
    hdr = (f"{'seq':>5} {'T_round':>8} {'ga':>4} "
           f"{'tok/step':>9} {'budget':>6} {'s/step':>7} {'tok/s':>8} "
           f"{'effTF/s':>8} {'hwTF/s':>7} "
           f"{'fastGiB':>8} {'bkGiB':>6} "
           f"{'h2dGB':>6} {'h2d%':>5} {'d2hGB':>6} {'d2h%':>5} "
           f"{'rc%':>4} {'idle%':>5} {'recomp':>7}")
    if steps:
        hdr += f" {'ETA_h':>6}"
    print(hdr)
    for r in rows:
        head = (f"{r['seq']:>5} {r['t_round']:>8,} {r['ga']:>4} "
                f"{r['tokens_step']:>9,} "
                f"{r['budget']:>6g} ")
        if "infeasible" in r:
            print(head + f"  INFEASIBLE: {r['infeasible']}")
            continue
        line = (head +
                f"{r['step_s']:>7.2f} {r['tok_s']:>8,.0f} "
                f"{r['eff_tfs']:>8.1f} {r['hw_tfs']:>7.1f} "
                f"{r['peak_gib']:>8.2f} {r['backing_gib']:>6.2f} "
                f"{r['h2d_gb']:>6.1f} {r['h2d_pct']:>4.0f}% "
                f"{r['d2h_gb']:>6.1f} {r['d2h_pct']:>4.0f}% "
                f"{r['rc_pct']:>3.0f}% {r['idle_pct']:>4.0f}% "
                f"{str(r['recompute']) + '/' + str(r['rewritable']):>7}")
        if steps:
            line += f" {steps * r['step_s'] / 3600:>6.1f}"
        print(line)


def main() -> int:
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run import presets as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--plugin", action="append", default=None,
                    help="external-family plugin module(s) to load")
    ap.add_argument("--opt", choices=["adamw", "muon"], default=None,
                    help="override the preset's opt_policy. Sizes O (and "
                         "so backing) correctly — muon matrices carry m "
                         "only — and the all-in TF/s column counts NS "
                         "work; CAVEAT: the roofline optimizer_us seed is "
                         "adamw-shaped, so the MAKESPAN under-charges "
                         "muon's NS matmul time (~0.3-0.5 s/step at 1B)")
    ap.add_argument("--t-round", type=int, default=None,
                    help="single-point round token budget (must be a "
                         "multiple of seq_len)")
    ap.add_argument("--t-rounds", default=None,
                    help="SWEEP: comma list of round token budgets, e.g. "
                         "8192,32768,65536 — ga derives from --tokens-step")
    ap.add_argument("--tokens-step", type=int, default=None,
                    help="tokens per optimizer step (default: the "
                         "preset's); ga = tokens-step / t_round per row")
    ap.add_argument("--ga-rounds", type=int, default=None,
                    help="single-point ga override (alternative to "
                         "--t-round; preset round budget)")
    ap.add_argument("--budget", type=float, default=14.0,
                    help="single-point device budget (GiB)")
    ap.add_argument("--budgets", default=None,
                    help="SWEEP: comma list of budgets, e.g. 16,8,4,2")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override cfg.seq_len (T_round and tokens/step "
                         "scale; families with learned positions grow "
                         "their table when n_ctx follows seq_len)")
    ap.add_argument("--seq-lens", default=None,
                    help="SWEEP: comma list of seq_lens — a third axis "
                         "over the ga-batch x budgets grid")
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
    ap.add_argument("--backing", type=float, default=None,
                    help="host-side (slab) capacity ceiling in GiB — "
                         "plans whose backing demand exceeds it fail "
                         "verification and report INFEASIBLE")
    ap.add_argument("--steps", type=int, default=None,
                    help="also print the ETA column for this many steps")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--top", type=int, default=5,
                    help="single-point mode: most expensive tasks shown")
    args = ap.parse_args()

    if args.plugin:
        from dataflow_training.model_families.families import load_plugins
        load_plugins(args.plugin)
    base = P.resolve_preset(args.preset)
    if args.opt:
        base = replace(base, opt_policy=args.opt)
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
    if args.opt == "muon" and not args.measured:
        probe = fam.lower(base)
        charged = any(op.get("name") == "muon_ns"
                      for t in probe.tasks if t.group == "optimizer"
                      for op in (t.metadata or {}).get("cost_subops", []))
        if not charged:
            print("NOTE: this family's roofline seeds do not yet charge "
                  "muon NS time (makespan optimistic ~0.3-0.5 s/step at "
                  "1B); --measured profiles the real optimizer and is "
                  "muon-exact")

    budgets = ([float(x) for x in args.budgets.split(",")]
               if args.budgets else [args.budget])
    seqs = ([int(x) for x in args.seq_lens.split(",")] if args.seq_lens
            else [args.seq_len or base.seq_len])
    t_rounds = ([int(x) for x in args.t_rounds.split(",")]
                if args.t_rounds
                else [args.t_round] if args.t_round else None)

    def geometry(seq: int) -> list[tuple[int, int]]:
        """(ga, batch) rows for one seq_len — batch is INTERNAL
        arithmetic (T_round / seq_len); the interface speaks T_round."""
        tokens_step = args.tokens_step or (base.tokens
                                           * base.grad_accum_rounds)
        if t_rounds is None:
            ga = args.ga_rounds or base.grad_accum_rounds
            return [(ga, tokens_step // (ga * seq))] \
                if args.tokens_step or args.ga_rounds else \
                [(base.grad_accum_rounds, base.batch)]
        out = []
        for tr in t_rounds:
            if tr % seq:
                raise SystemExit(f"--t-round {tr} not a multiple of "
                                 f"seq_len {seq}")
            if tokens_step % tr:
                raise SystemExit(f"--tokens-step {tokens_step} not a "
                                 f"multiple of t_round {tr}")
            out.append((tokens_step // tr, tr // seq))
        return out

    sweep = bool(args.t_rounds or args.budgets or args.seq_lens)
    print(f"preset {args.preset}  family {fam.name}  hw {args.hw}: "
          f"{hw.peak_bf16_tflops:.0f} TF bf16 x {hw.matmul_eff:.2f}, "
          f"{hw.mem_bw_gbs:.0f} GB/s, pcie {hw.pcie_gbs:.0f} GB/s  "
          f"costs={'MEASURED (profiled)' if args.measured else 'roofline'}")
    rows = []
    for seq in seqs:
        for ga, b in geometry(seq):
            cfg = replace(base, seq_len=seq, grad_accum_rounds=ga, batch=b)
            for budget in budgets:
                try:
                    rows.append(combo_row(fam, cfg, hw, budget,
                                          measured=args.measured,
                                          recompute=recompute,
                                          profile_cache=profile_cache,
                                          backing_gib=args.backing))
                except ValueError as exc:
                    # a combo the planner cannot fit is a RESULT, not a crash
                    rows.append({"seq": seq, "ga": ga,
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
