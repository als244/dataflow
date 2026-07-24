#!/usr/bin/env python
"""Choose WHICH cells are worth running on real hardware, and write cells.json.

The prediction pass covers the whole grid cheaply and marks what the planner
cannot fit. Measuring all survivors would waste hours on cells that behave
identically: at 8B most geometries land in the same regime, and a second cell
with the same recompute pressure, transfer duty and idle fraction teaches
nothing the first did not.

Tokens-per-round is not really an axis to report: it is a FREE knob the
operator picks. Nobody runs a 64K-token round on a 16 GiB budget — that plan
asks for 220 GiB of host memory and runs slower than a smaller round on the
same hardware. So for each (sequence, tokens/step, budget, allowance) only the
best round size is on the FRONTIER, and the rest are choices no one would make.
Real runs go to the frontier, plus a few dominated cells kept as controls so
the ranking itself is checked rather than trusted.

So the subset is chosen in BEHAVIOUR space, not by hand:

  0. reduce over tokens-per-round: keep the best-throughput round per
     (sequence, tokens/step, budget, allowance); the rest are dominated
  1. keep cells predicted feasible in every optimizer being swept
  2. describe each by what the plan actually does — recompute share, transfer
     duty each way, idle share, and normalised throughput
  3. cluster those descriptions and take the cell nearest each centre, so every
     distinct regime is measured once
  4. force-keep one BUDGET SPINE (a single geometry across all its feasible
     budgets), because the headline result is throughput vs budget and a curve
     needs its points to share a geometry

Result: a few dozen runs that cover the landscape instead of a few hundred that
re-measure it.
"""
import argparse
import json
import os
import sys

FEATURES = ("rc_pct", "h2d_pct", "d2h_pct", "idle_pct", "tok_s")


def load_feasible(data_dir, opts):
    per_opt = []
    for opt in opts:
        path = os.path.join(data_dir, f"predict_measured_{opt}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing predictions: {path}")
        rows = [json.loads(line) for line in open(path)]
        per_opt.append({(r["seq"], r["t_round"], r["t_step"], r["budget"],
                         r.get("backing", 0.0)): r
                        for r in rows if "tok_s" in r})
    keys = set(per_opt[0])
    for d in per_opt[1:]:
        keys &= set(d)
    return sorted(keys), per_opt[0]


def kmeans(points, k, iters=40):
    """Plain Lloyd's algorithm — k evenly spread seeds, no dependency beyond
    numpy, deterministic so a re-run picks the same cells."""
    import numpy as np

    n = len(points)
    k = max(1, min(k, n))
    step = max(1, n // k)
    centres = points[::step][:k].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            hit = points[labels == j]
            if len(hit):
                centres[j] = hit.mean(0)
    return labels, centres


def main():
    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--opts", default="adamw,muon")
    ap.add_argument("--target", type=int, default=18,
                    help="how many cells to measure in total")
    ap.add_argument("--out", default=os.path.join(here, "cells.json"))
    args = ap.parse_args()

    opts = args.opts.split(",")
    keys, rows = load_feasible(args.data, opts)
    print(f"{len(keys)} cells feasible in all of {opts}")
    if not keys:
        raise SystemExit("nothing feasible — check the prediction pass")

    # tokens-per-round is a free knob: reduce over it and keep the winner
    best, dominated = {}, []
    for key in keys:
        s_, tr, ts, b, k = key
        slot = (s_, ts, b, k)
        cur = best.get(slot)
        if cur is None or rows[key]["tok_s"] > rows[cur]["tok_s"]:
            if cur is not None:
                dominated.append(cur)
            best[slot] = key
        else:
            dominated.append(key)
    frontier = sorted(best.values())
    print(f"frontier: {len(frontier)} cells (best tokens/round per "
          f"sequence x tokens-step x budget x allowance); "
          f"{len(dominated)} dominated choices set aside")
    keys = frontier

    # budget spine: the geometry whose budget axis is best covered
    by_geo = {}
    for (s, tr, ts, b, k) in keys:
        by_geo.setdefault((s, tr, ts, k), []).append(b)
    spine_geo = max(by_geo, key=lambda g: (len(by_geo[g]), -g[0]))
    spine = [(spine_geo[0], spine_geo[1], spine_geo[2], b, spine_geo[3])
             for b in sorted(by_geo[spine_geo])]
    print(f"budget spine: seq{spine_geo[0]} tr{spine_geo[1]} ts{spine_geo[2]} "
          f"backing{spine_geo[3]} over budgets {sorted(by_geo[spine_geo])}")

    rest = [k for k in keys if k not in set(spine)]
    picks = list(spine)
    if rest and len(picks) < args.target:
        raw = np.array([[rows[k].get(f, 0.0) for f in FEATURES] for k in rest],
                       dtype=float)
        # throughput spans orders of magnitude; percentages already share a scale
        raw[:, -1] = np.log10(np.maximum(raw[:, -1], 1.0))
        span = raw.max(0) - raw.min(0)
        span[span == 0] = 1.0
        norm = (raw - raw.min(0)) / span
        k = min(args.target - len(picks), len(rest))
        labels, centres = kmeans(norm, k)
        for j in range(len(centres)):
            idx = np.where(labels == j)[0]
            if not len(idx):
                continue
            d = ((norm[idx] - centres[j]) ** 2).sum(1)
            picks.append(rest[int(idx[d.argmin()])])
        print(f"clustered {len(rest)} remaining cells into {k} regimes")

    # a couple of dominated cells ride along as controls: if the engine ranks
    # them differently from the simulator, the reduction above is unsafe
    controls = []
    if dominated:
        dominated.sort(key=lambda k: -rows[k]["tok_s"])
        controls = [dominated[0], dominated[len(dominated) // 2]]
        picks.extend(controls)

    spine_set, control_set = set(spine), set(controls)
    out = []
    for key in sorted(set(picks)):
        s_, tr, ts, b, k = key
        r = rows[key]
        out.append(dict(seq=s_, t_round=tr, t_step=ts, budget=b, backing=k,
                        spines=(["budget_spine"] if key in spine_set
                                else ["dominated_control"] if key in control_set
                                else ["frontier"]),
                        predicted_s=round(r.get("step_s", 0.0), 2)))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    est = sum(c["predicted_s"] for c in out) * 6 * len(opts) / 60
    print(f"selected {len(out)} cells x {len(opts)} optimizers "
          f"(~{est:.0f} min of stepping) -> {args.out}")
    for c in out:
        print(f"  seq{c['seq']:>5} tr{c['t_round']:>6} ts{c['t_step']:>7} "
              f"b{c['budget']:>5g} k{c['backing']:>6g}  {c['predicted_s']:>6.2f}s  "
              f"{','.join(c['spines'])}")


if __name__ == "__main__":
    main()
