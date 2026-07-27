#!/usr/bin/env python
"""Per-task-group audit: the cost the SIM used (annotated program runtime_us)
vs what the task actually took in the real run (measured trace intervals).
Localizes WHICH task kinds are mispriced and by how much."""
import json
import statistics
import sys
from collections import defaultdict

ann = json.load(open(sys.argv[1]))       # *.annotated.json
run = json.load(open(sys.argv[2]))       # *.measured.json

cost, group = {}, {}
for t in ann["tasks"]:
    cost[t["id"]] = t.get("runtime_us", 0.0)
    group[t["id"]] = t.get("group", "?")

real = defaultdict(float)
for iv in run["log"]["task_intervals"]:
    real[iv["task_id"]] += iv["end"] - iv["start"]
sim = defaultdict(float)
for iv in run["sim_log"]["task_intervals"]:
    sim[iv["task_id"]] += iv["end"] - iv["start"]

bg = defaultdict(list)
for tid, r in real.items():
    if tid in cost and cost[tid] > 0:
        bg[group.get(tid, "?")].append((r, cost[tid], sim.get(tid, 0.0)))

print(f"{'group':<12} {'n':>4} {'sim cost(ms)':>12} {'sim iv(ms)':>11} {'real(ms)':>9} "
      f"{'real/cost':>9} {'total real':>11} {'total cost':>10}")
tr = tc = 0.0
for g, v in sorted(bg.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
    r = statistics.fmean(x[0] for x in v)
    c = statistics.fmean(x[1] for x in v)
    s = statistics.fmean(x[2] for x in v)
    sr = sum(x[0] for x in v)
    sc = sum(x[1] for x in v)
    tr += sr
    tc += sc
    print(f"{g:<12} {len(v):>4} {c/1e3:>12.2f} {s/1e3:>11.2f} {r/1e3:>9.2f} "
          f"{r/c:>9.2f} {sr/1e6:>10.2f}s {sc/1e6:>9.2f}s")
print(f"{'TOTAL':<12} {'':>4} {'':>12} {'':>11} {'':>9} {tr/tc:>9.2f} {tr/1e6:>10.2f}s {tc/1e6:>9.2f}s")
