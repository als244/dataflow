#!/usr/bin/env python
"""Plot block_bwd per-iteration time under sustained back-to-back load."""
import json
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open(sys.argv[1]))
b2b = [x / 1e3 for x in D["back_to_back_us"]]
sb = [x / 1e3 for x in D["sync_between_us"]]
run = [statistics.mean(b2b[max(0, i - 25):i + 1]) for i in range(len(b2b))]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(b2b) + 1), b2b, ",", alpha=0.25, color="tab:blue")
ax.plot(range(1, len(b2b) + 1), run, "-", color="tab:blue", lw=2,
        label="back-to-back (25-iter running avg)")
ax.axhline(statistics.median(sb), color="tab:green", ls="--", lw=2,
           label=f"profiler sync-between = {statistics.median(sb):.1f} ms")
ax.axhline(60, color="tab:red", ls=":", lw=2, label="real-pipeline block_bwd ≈ 60 ms")
ax.set_xlabel("iteration (sustained back-to-back, no sync)")
ax.set_ylabel("block_bwd GPU time (ms)")
ax.set_title(f"block_bwd sustained-load ramp\n{D['task']}  {D['cell']}")
ax.legend()
ax.grid(alpha=0.3)
out = sys.argv[1].replace(".json", ".png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
print(f"first20={statistics.mean(b2b[:20]):.1f}ms  last200={statistics.mean(b2b[-200:]):.1f}ms  "
      f"sync-between={statistics.median(sb):.1f}ms")
