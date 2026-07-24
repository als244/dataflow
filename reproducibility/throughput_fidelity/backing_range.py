#!/usr/bin/env python
"""Set the host-allowance ladder from what plans actually ask for.

Guessing a ceiling is the wrong way round. The simulator already answers the
question: plan the whole grid with NO host ceiling and every row reports the
peak backing its plan wanted. Those demands are the interesting range — below
the smallest, nothing can run; above the largest, more host memory buys
nothing. This reads that distribution and writes the ladder into env.json.
"""
import argparse
import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(here, "data"))
ap.add_argument("--env", default=os.path.join(here, "env.json"))
ap.add_argument("--rungs", type=int, default=3)
args = ap.parse_args()

env = json.load(open(args.env))
demands = []
for path in glob.glob(os.path.join(args.data, "predict_unlimited_*.jsonl")):
    for line in open(path):
        row = json.loads(line)
        if "backing_gib" in row and "tok_s" in row:
            demands.append(row["backing_gib"])
if not demands:
    raise SystemExit("no unlimited-backing predictions found")

demands.sort()
floor = env["persistent_gib"] * 1.15
host_cap = env["backing_gib"]          # what this host can spare
lo = max(floor, demands[0])
hi = min(host_cap, demands[-1])
print(f"{len(demands)} feasible plans; peak backing demand "
      f"min {demands[0]:.1f}  median {demands[len(demands)//2]:.1f}  "
      f"max {demands[-1]:.1f} GiB")
print(f"persistent floor {floor:.1f} GiB, host can spare {host_cap:.1f} GiB")

if hi <= lo * 1.05:
    ladder = [round(hi, 1)]
else:
    n = max(2, args.rungs)
    ladder = sorted({round(lo + (hi - lo) * i / (n - 1), 1) for i in range(n)})
# a rung above the largest demand teaches nothing; one at the floor squeezes
# every plan, which is the point of the axis
env["backings"] = ladder
env["backing_demand_gib"] = {"min": round(demands[0], 1),
                             "median": round(demands[len(demands) // 2], 1),
                             "max": round(demands[-1], 1)}
env["backing_demand_exceeds_host"] = bool(demands[-1] > host_cap)
json.dump(env, open(args.env, "w"), indent=2)
print(f"backings -> {ladder}")
if env["backing_demand_exceeds_host"]:
    print(f"NOTE: the greediest plan wants {demands[-1]:.1f} GiB, more than this "
          f"host can spare — the top rung is a host limit, not a plan choice")
