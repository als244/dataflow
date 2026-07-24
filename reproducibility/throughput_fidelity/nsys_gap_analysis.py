#!/usr/bin/env python
"""Is the in-pipeline slowdown INSIDE kernels or BETWEEN them?

Queries an nsys sqlite export: per-kernel intervals on the GPU, total busy
(union) vs span, and the gap distribution. If kernels are back-to-back
(gaps ~0) the dilation is inside kernels (clock/memory); if there are large
inter-kernel gaps, the host launch path is starving the GPU.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
tabs = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
ktab = next((t for t in ("CUPTI_ACTIVITY_KIND_KERNEL",) if t in tabs), None)
if not ktab:
    print("tables:", sorted(t for t in tabs if "KERNEL" in t.upper() or "CUPTI" in t.upper())[:20])
    sys.exit("no kernel table")

cols = {r[1] for r in cur.execute(f"PRAGMA table_info({ktab})")}
namecol = "demangledName" if "demangledName" in cols else "shortName"
rows = list(cur.execute(
    f"SELECT k.start, k.end, s.value FROM {ktab} k "
    f"LEFT JOIN StringIds s ON k.{namecol}=s.id ORDER BY k.start"))
print(f"{len(rows)} kernels")

# Segment at idle gaps > 500ms: separates the profiling pass / init from the
# actual training steps, so we analyze steady state, not process lifetime.
SEG_GAP = 500_000_000
segs = []
cur_rows = [rows[0]]
for prev, r in zip(rows, rows[1:]):
    if r[0] - prev[1] > SEG_GAP:
        segs.append(cur_rows)
        cur_rows = []
    cur_rows.append(r)
segs.append(cur_rows)
segs = [s for s in segs if len(s) > 50]
print(f"{len(segs)} activity segments (split at >500ms idle):")
for i, s in enumerate(segs):
    sp = max(r[1] for r in s) - s[0][0]
    bz = 0
    pe = None
    for a, b, _ in s:
        if pe is None or a > pe:
            bz += b - a
            pe = b
        elif b > pe:
            bz += b - pe
            pe = b
    print(f"  seg{i}: {len(s):6d} kernels  span {sp/1e9:7.3f}s  busy {bz/1e9:7.3f}s ({100*bz/sp:5.1f}%)")
# analyze the LONGEST segment = the training steps
rows = max(segs, key=lambda s: max(r[1] for r in s) - s[0][0])
print(f"\n--- analyzing longest segment: {len(rows)} kernels ---")
t0, t1 = rows[0][0], max(r[1] for r in rows)
span = t1 - t0
busy = 0
prev_end = None
gaps = []
for s, e, name in rows:
    if prev_end is None or s > prev_end:
        busy += e - s
        if prev_end is not None and s > prev_end:
            gaps.append((s - prev_end, name))
        prev_end = e
    elif e > prev_end:
        busy += e - prev_end
        prev_end = e
print(f"span      {span/1e9:8.3f} s")
print(f"kernel busy(union) {busy/1e9:8.3f} s  = {100*busy/span:.1f}% of span")
print(f"gap total {(span-busy)/1e9:8.3f} s  = {100*(span-busy)/span:.1f}%  ({len(gaps)} gaps)")
gaps.sort(reverse=True)
print("\nlargest inter-kernel gaps (us, kernel that followed):")
for g, n in gaps[:10]:
    print(f"  {g/1e3:9.1f}us  before {str(n)[:70]}")
tot = sum(g for g, _ in gaps)
big = sum(g for g, _ in gaps if g > 100_000)
print(f"\ngap time total {tot/1e6:.1f}ms | in gaps >100us: {big/1e6:.1f}ms")
buckets = defaultdict(float)
for g, _ in gaps:
    b = "<10us" if g < 10_000 else "10-100us" if g < 100_000 else "100us-1ms" if g < 1e6 else ">1ms"
    buckets[b] += g
for b in ("<10us", "10-100us", "100us-1ms", ">1ms"):
    print(f"  {b:>10}: {buckets[b]/1e6:8.2f}ms")
