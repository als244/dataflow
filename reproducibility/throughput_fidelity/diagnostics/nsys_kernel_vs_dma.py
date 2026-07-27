#!/usr/bin/env python
"""Mechanical test of the fidelity gap: does a kernel's duration depend on how
much concurrent host<->device DMA overlaps it?

For every instance of a chosen kernel (default: the flash backward), computes
  duration  vs  bytes of MEMCPY overlapping its execution window
and reports duration binned by overlap. Also reports duration vs wall-time
(ramp = thermal/power) so the two candidate mechanisms are separated:
  * duration rises with DMA overlap      -> memory/bus contention
  * duration rises with time, not overlap -> power/clock drift
"""
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
pat = sys.argv[2] if len(sys.argv) > 2 else "flash_bwd"
cur = db.cursor()
cols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
namecol = "demangledName" if "demangledName" in cols else "shortName"
ks = list(cur.execute(
    f"SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    f"LEFT JOIN StringIds s ON k.{namecol}=s.id WHERE s.value LIKE ? ORDER BY k.start",
    (f"%{pat}%",)))
print(f"{len(ks)} instances of '{pat}'")
if not ks:
    sys.exit(0)

mtabs = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%MEMCPY%'")]
mc = []
for t in mtabs:
    tc = {r[1] for r in cur.execute(f"PRAGMA table_info({t})")}
    if {"start", "end"} <= tc:
        byt = "bytes" if "bytes" in tc else None
        q = f"SELECT start,end,{byt if byt else '0'} FROM {t} ORDER BY start"
        mc += list(cur.execute(q))
mc.sort()
print(f"{len(mc)} memcpy records from {mtabs}")

t0 = ks[0][0]
rows = []
j = 0
for s, e in ks:
    ov_bytes = 0
    ov_time = 0
    for ms, me, mb in mc:
        if me < s:
            continue
        if ms > e:
            break
        lo, hi = max(s, ms), min(e, me)
        if hi > lo:
            ov_time += hi - lo
            span = max(1, me - ms)
            ov_bytes += mb * (hi - lo) / span
    rows.append(((e - s) / 1e6, ov_bytes / 1e9, ov_time / (e - s), (s - t0) / 1e9))

durs = [r[0] for r in rows]
print(f"duration ms: min {min(durs):.2f}  med {statistics.median(durs):.2f}  "
      f"mean {statistics.fmean(durs):.2f}  max {max(durs):.2f}  "
      f"spread {max(durs)/min(durs):.2f}x")

print("\n--- duration binned by CONCURRENT DMA (GB overlapping the kernel) ---")
rows.sort(key=lambda r: r[1])
n = max(1, len(rows) // 4)
for i in range(0, len(rows), n):
    ch = rows[i:i + n]
    if not ch:
        continue
    print(f"  dma {statistics.fmean(c[1] for c in ch):6.3f} GB "
          f"(busy-frac {statistics.fmean(c[2] for c in ch):4.2f}) -> "
          f"dur {statistics.fmean(c[0] for c in ch):6.2f} ms   (n={len(ch)})")

print("\n--- duration vs WALL TIME (ramp => power/thermal) ---")
rows.sort(key=lambda r: r[3])
for i in range(0, len(rows), n):
    ch = rows[i:i + n]
    if not ch:
        continue
    print(f"  t={statistics.fmean(c[3] for c in ch):6.1f}s -> "
          f"dur {statistics.fmean(c[0] for c in ch):6.2f} ms   (n={len(ch)})")
