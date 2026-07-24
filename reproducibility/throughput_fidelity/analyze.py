#!/usr/bin/env python
"""Summarize the sweep: predict feasibility, measure success/fidelity, and — for
failed measure cells — the predicted backing demand (so we can tell backing-ceiling
failures from real infeasibility). Reads data/ in this experiment dir."""
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(name):
    p = os.path.join(D, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def host_pressure(rows):
    """Is the host allowance actually constraining anything, and what would
    relief be worth? Reported as a distribution because it varies by cell: a
    single "peak demand" number is only defined when host memory is free, and
    stops meaning anything once a ceiling is in place."""
    feas = [r for r in rows if "tok_s" in r]
    if not feas:
        return
    bind = [r for r in feas if r.get("binding")]
    gains = sorted(r["host_marginal_gain"] for r in feas
                   if r.get("host_marginal_gain") is not None)
    print(f"  host allowance binds on {len(bind)}/{len(feas)} cells")
    if gains:
        top = gains[-1]
        print(f"  value of 25% more host: median {gains[len(gains) // 2] * 100:+.1f}%"
              f"  best {top * 100:+.1f}%  "
              f"({sum(1 for g in gains if g > 0.02)} cells would gain >2%)")


for opt in ("adamw", "muon"):
    pm = load(f"predict_measured_{opt}.jsonl")
    feas = [r for r in pm if "eff_tfs" in r]
    print(f"predict_measured {opt}: {len(pm)} rows | {len(feas)} feasible | "
          f"{len(pm) - len(feas)} infeasible")
    host_pressure(pm)
    key = {(r["seq"], r["t_round"], r["t_step"], r["budget"],
            r.get("backing")): r for r in pm}
    ms = load(f"measure_{opt}.jsonl")
    ok = [r for r in ms if "meas_s" in r]
    bad = [r for r in ms if "failed" in r]
    print(f"measure {opt}: {len(ok)} ok | {len(bad)} FAILED")
    print(f"  {'seq':>5} {'tr':>6} {'ts':>7} {'bud':>5} {'back':>6} | {'meas_s':>7} {'pred_s':>7} "
          f"{'ratio':>5} {'tok/s':>8} {'effTF':>5} | predBacking")
    for r in ms:
        k = (r["seq"], r["t_round"], r["t_step"], r["budget"],
             r.get("backing"))
        pmr = key.get(k, {})
        bk = pmr.get("backing_gib")
        bkstr = f"{bk:.0f}GiB" if bk is not None else (pmr.get("infeasible", "?")[:16] if pmr else "?")
        if "meas_s" in r:
            print(f"  {r['seq']:>5} {r['t_round']:>6} {r['t_step']:>7} {r['budget']:>5g} {r.get('backing',0):>6g} | "
                  f"{r['meas_s']:>7.2f} {r['pred_s']:>7.2f} {r['ratio']:>5.2f} "
                  f"{r['tok_s']:>8.0f} {r['eff_tfs']:>5.0f} | {bkstr}")
        else:
            print(f"  {r['seq']:>5} {r['t_round']:>6} {r['t_step']:>7} {r['budget']:>5g} {r.get('backing',0):>6g} | "
                  f"{'FAILED':>7} {r.get('failed','')[:26]:<26} | predBacking={bkstr}")
    print()
