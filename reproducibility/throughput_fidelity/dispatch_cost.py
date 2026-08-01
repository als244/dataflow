"""What does the dispatcher cost per COMPUTE TASK vs per TRANSFER?

    python dispatch_cost.py trace.nsys-rep

The dispatcher issues both: compute tasks AND the prefetch_after/offload_after
triggers a plan carries. Plans in this campaign hold a median of 1.52 transfers
per compute task, so charging the dispatcher only for tasks under-counts its
work by ~2.5x -- but the two are not the same price, and campaign data cannot
separate them (tasks, transfers and tokens are collinear across cells, and a
least-squares fit returns a NEGATIVE per-task cost).

They are separable in a trace. Every issued item gets an NVTX range on the
dispatch thread -- `block_fwd_*` etc. for tasks, `from_slow:X` / `to_slow:X`
for transfers -- so the host-side gap between consecutive ranges IS the
dispatcher's own processing time, and the range that FOLLOWS a gap says which
kind of item that time was spent getting to.

Reports the gap distribution split by what came next, which is the number the
C-core work needs: if transfer dispatch dominates, that is what a native core
should shed first.
"""
import argparse
import bisect
import collections
import os
import shutil
import sqlite3
import statistics as st
import subprocess


def ensure_sqlite(path):
    if path.endswith(".sqlite"):
        return path
    out = path[:-len(".nsys-rep")] + ".sqlite"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    nsys = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    print(f"  exporting {os.path.basename(path)} ...", flush=True)
    r = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite",
                        "true", "-o", out, path], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"nsys export failed:\n{r.stderr[-400:]}")
    return out


def kind(name):
    if name.startswith("from_slow:"):
        return "transfer from_slow"
    if name.startswith("to_slow:"):
        return "transfer to_slow"
    for p in ("block_fwd", "block_bwd", "block_recompute", "optimizer",
              "head", "embed", "prologue"):
        if name.startswith(p):
            return f"task {p}"
    return f"task {name.split('_')[0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--max-gap-us", type=float, default=20000.0,
                    help="ignore gaps above this (step boundaries, stalls)")
    a = ap.parse_args()

    db = sqlite3.connect(ensure_sqlite(a.trace))
    c = db.cursor()
    c.execute("""SELECT text, start, end, globalTid FROM NVTX_EVENTS
                 WHERE text IS NOT NULL AND end IS NOT NULL ORDER BY start""")
    rows = c.fetchall()
    if not rows:
        raise SystemExit("no NVTX ranges — was the capture bracketed? "
                         "annotation is off outside a profiled window")
    # the dispatch thread is the one issuing the compute tasks
    tids = collections.Counter(r[3] for r in rows if not r[0].startswith(("from_slow", "to_slow")))
    tid = tids.most_common(1)[0][0]
    seq = [r for r in rows if r[3] == tid]
    print(f"{os.path.basename(a.trace)}: {len(seq)} NVTX items on dispatch tid {tid}")

    # A host-side gap between NVTX ranges is NOT dispatcher cost: the
    # dispatcher also WAITS there (dependencies, pacing), and waiting while
    # the GPU works is free. What costs throughput is GPU IDLE. So take the
    # kernel-union holes -- the same definition gap_breakdown uses -- and
    # attribute each to the item the dispatcher went on to issue.
    c.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    iv = c.fetchall()
    merged = []
    cs, ce = iv[0]
    for s0, e0 in iv[1:]:
        if s0 <= ce:
            ce = max(ce, e0)
        else:
            merged.append((cs, ce)); cs, ce = s0, e0
    merged.append((cs, ce))
    starts = [x[1] for x in seq]
    gaps = collections.defaultdict(list)
    for i in range(len(merged) - 1):
        g0, g1 = merged[i][1], merged[i + 1][0]
        g = (g1 - g0) / 1000.0
        if not (0 < g <= a.max_gap_us):
            continue
        j = bisect.bisect_left(starts, g0)              # next item issued
        if j < len(seq):
            gaps[kind(seq[j][0])].append(g)
    tot_items = sum(len(v) for v in gaps.values())
    tot_us = sum(sum(v) for v in gaps.values())
    print(f"  {'what the dispatcher went on to issue':<30}{'n':>7}{'median us':>11}"
          f"{'mean us':>10}{'total ms':>11}")
    for k, v in sorted(gaps.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {k:<30}{len(v):>7}{st.median(v):>11.1f}{st.mean(v):>10.1f}"
              f"{sum(v) / 1000:>11.1f}")
    task = [x for k, v in gaps.items() if k.startswith("task") for x in v]
    xfer = [x for k, v in gaps.items() if k.startswith("transfer") for x in v]
    print(f"\n  BEFORE A COMPUTE TASK : n={len(task):<6} median {st.median(task):7.1f} us"
          f"   total {sum(task)/1000:8.1f} ms" if task else "  no task gaps")
    print(f"  BEFORE A TRANSFER     : n={len(xfer):<6} median {st.median(xfer):7.1f} us"
          f"   total {sum(xfer)/1000:8.1f} ms" if xfer else "  no transfer gaps")
    if task and xfer:
        print(f"  transfers are {len(xfer)/len(task):.2f}x as numerous, "
              f"{st.median(xfer)/st.median(task):.2f}x the per-item cost, and "
              f"{100*sum(xfer)/(sum(task)+sum(xfer)):.0f}% of dispatch time")
    print(f"  TOTAL dispatch gap time: {tot_us/1000:.1f} ms over {tot_items} items")


main()
