"""Turn a --durations=0 run into the suite-duration reference document.

    python -m pytest -q --durations=0 > run.log
    python tools/test/suite_durations.py run.log --wall 804 -o tests/SUITE_DURATIONS_CHICAGO.md

    python tools/test/suite_durations.py run.log --compare tests/SUITE_DURATIONS_CHICAGO.md

The document this writes is the expectation baseline for how long suite tasks
take at a commit. Re-measure after structural suite changes, and only from a
serial run on an otherwise idle box -- concurrent GPU work invalidates the
comparison (contention reds are not regressions).

--compare parses an existing document's per-file table and reports the files
that moved most, which is how you tell "the suite got slower" from "one file
got slower".
"""
import argparse
import collections
import re
from pathlib import Path

# "1.23s call     tests/path/test_x.py::test_name[param]"
DUR = re.compile(r"^([\d.]+)s\s+(call|setup|teardown)\s+(\S+::\S+)\s*$")
TOTALS = re.compile(r"(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) deselected|(\d+) error")
# per-file rows of an existing document: | `path` | n | 12.3s | ...
FILEROW = re.compile(r"^\|\s*`([^`]+\.py)`\s*\|\s*(\d+)\s*\|\s*([\d.]+)s")


PROGRESS = re.compile(r"^([.sFExX]+)\s+\[\s*\d+%\]$", re.M)
OUTCOME = {".": "passed", "s": "skipped", "F": "failed",
           "E": "error", "x": "xfailed", "X": "xpassed"}


def counts_from_progress(text):
    """Outcome counts from the progress dots.

    pytest -q does not always leave a parseable "N passed" line in a
    redirected log, and a duration document with a blank summary is worse
    than useless -- the dots are always there, one character per test.
    """
    blob = "".join(PROGRESS.findall(text))
    tally = collections.Counter(blob)
    return ", ".join(f"{tally[k]} {OUTCOME[k]}" for k in OUTCOME if tally.get(k))


def parse(path):
    per_test = collections.defaultdict(lambda: collections.defaultdict(float))
    summary, wall = "", None
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
        m = DUR.match(line.strip())
        if m:
            per_test[m[3]][m[2]] += float(m[1])
            continue
        if " passed" in line or " failed" in line or " error" in line:
            if TOTALS.search(line):
                summary = line.strip().strip("=").strip()
        m = re.search(r"in ([\d.]+)s", line)
        if m and ("passed" in line or "failed" in line):
            wall = float(m[1])
    return per_test, summary or counts_from_progress(text), wall


def by_file(per_test):
    files = collections.defaultdict(lambda: {"tests": 0, "total": 0.0, "slowest": ("", 0.0)})
    for nodeid, phases in per_test.items():
        path = nodeid.split("::")[0]
        tot = sum(phases.values())
        f = files[path]
        f["tests"] += 1
        f["total"] += tot
        if tot > f["slowest"][1]:
            f["slowest"] = (nodeid.split("::", 1)[1], tot)
    return files


def read_doc_files(path):
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = FILEROW.match(line.strip())
        if m and m[1] not in out:
            out[m[1]] = float(m[3])
    return out


def render(per_test, files, summary, wall, args):
    tot_acc = sum(sum(p.values()) for p in per_test.values())
    order = sorted(files.items(), key=lambda kv: -kv[1]["total"])
    tests_order = sorted(per_test.items(), key=lambda kv: -sum(kv[1].values()))
    over60 = sum(1 for _, f in order if f["total"] >= 60)
    in1060 = sum(1 for _, f in order if 10 <= f["total"] < 60)
    in110 = sum(1 for _, f in order if 1 <= f["total"] < 10)
    sub1 = sum(1 for _, f in order if f["total"] < 1)
    top10 = sum(f["total"] for _, f in order[:10])

    L = [f"# Test-suite duration reference ({args.box})", ""]
    L.append(f"Measured at {args.at} on {args.box}, single serial run of the "
             f"canonical suite invocation (`python -m pytest -q --durations=0`, "
             f"`dataflow` conda env, box otherwise idle). Use it as the "
             f"expectation baseline for how long suite tasks take at this "
             f"point; re-measure after structural suite changes.")
    L += ["", f"Stack: {args.stack}.", "", "## Summary", ""]
    mm = f"{int(wall // 60)}:{int(wall % 60):02d}" if wall else "n/a"
    L.append(f"- **Wall time: {mm}** ({wall:.0f}s) — {summary}." if wall
             else f"- {summary}.")
    L.append(f"- Time accounted to individual tests below: {tot_acc:.0f}s"
             + (f" ({100 * tot_acc / wall:.0f}% of wall; the rest is collection, "
                f"session setup, and the many sub-5ms phases pytest omits)."
                if wall else "."))
    L.append(f"- {len(order)} test files with measurable time; distribution by "
             f"per-file total: {over60} files over 60s, {in1060} in 10-60s, "
             f"{in110} in 1-10s, {sub1} under 1s.")
    L.append(f"- Concentration: the top 10 files hold {top10:.0f}s "
             f"({100 * top10 / tot_acc:.0f}% of accounted time) — they are the "
             f"levers if the suite ever needs to get faster.")

    L += ["", "### Top 10 files", "",
          "| # | file | tests | total | share of accounted |",
          "|---|---|---|---|---|"]
    for i, (p, f) in enumerate(order[:10], 1):
        L.append(f"| {i} | `{p}` | {f['tests']} | {f['total']:.1f}s | "
                 f"{100 * f['total'] / tot_acc:.1f}% |")

    L += ["", "### Top 10 individual tests", "", "| # | test | total |", "|---|---|---|"]
    for i, (n, ph) in enumerate(tests_order[:10], 1):
        L.append(f"| {i} | `{n}` | {sum(ph.values()):.1f}s |")

    L += ["", "## Per-file breakdown", "",
          "Sorted by total attributed time (call + setup + teardown of every "
          "test in the file).", "",
          "| file | tests | total | slowest test | its total |",
          "|---|---|---|---|---|"]
    for p, f in order:
        L.append(f"| `{p}` | {f['tests']} | {f['total']:.1f}s | "
                 f"`{f['slowest'][0]}` | {f['slowest'][1]:.1f}s |")

    if args.per_test:
        L += ["", "## Per-test detail", ""]
        for p, f in order:
            if f["total"] < args.per_test:
                continue
            L += [f"### `{p}` — {f['total']:.1f}s total, {f['tests']} tests", "",
                  "| test | call | setup | teardown | total |", "|---|---|---|---|---|"]
            rows = [(n, ph) for n, ph in per_test.items() if n.startswith(p + "::")]
            for n, ph in sorted(rows, key=lambda kv: -sum(kv[1].values())):
                L.append(f"| `{n.split('::', 1)[1]}` | {ph['call']:.2f}s | "
                         f"{ph['setup']:.2f}s | {ph['teardown']:.2f}s | "
                         f"{sum(ph.values()):.2f}s |")
            L.append("")

    L += ["", "## Methodology and caveats", "",
          "- Source: `--durations=0` phase report; pytest omits phases under 5ms "
          "(`--durations-min` default), so per-file sums are floors and very "
          "cheap tests may be absent entirely.",
          "- Session/module-scoped fixture cost lands in the setup phase of the "
          "first test that triggers it — a file's first test can look "
          "artificially heavy.",
          "- Serial single-run measurement under the serial-battery rule: "
          "concurrent GPU work on the box invalidates comparisons (contention "
          "reds are not regressions).",
          "- The deselected count is the opt-in lanes (fleet); their cost is not "
          "in this document.",
          "- MEASURE ON A WARM PROFILE CACHE. `impl_fingerprint()` hashes "
          "`run/profiling.py`, so any edit to that file invalidates every "
          "cached profile and the next run re-measures every signature from "
          "cold. Measured immediately after such an edit this suite came in at "
          "28:33 against a true 14:16 \u2014 the first run after touching the "
          "profiling module times the cold path, not the suite.",
          "- Files that profile under `tmp_path` (test_profiling_memory, "
          "test_engine_stress) are cold BY DESIGN and pay real device work "
          "every run; a warm cache does not help them.", ""]
    return "\n".join(L)


def compare(files, doc_path, limit):
    old = read_doc_files(doc_path)
    rows = []
    for p, f in files.items():
        if p in old:
            rows.append((f["total"] - old[p], old[p], f["total"], p))
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"\n=== vs {doc_path}   ({len(rows)} files in both)")
    o, n = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"  accounted total {o:.0f}s -> {n:.0f}s   ({n - o:+.0f}s, {100 * (n / o - 1):+.1f}%)")
    print(f"\n  {'delta':>9}{'was':>9}{'now':>9}  file")
    for d, ov, nv, p in rows[:limit]:
        print(f"  {d:>+8.1f}s{ov:>8.1f}s{nv:>8.1f}s  {p}")
    gone = sorted(set(old) - {p for _, _, _, p in rows})
    new = sorted(set(files) - set(old))
    if gone:
        print(f"\n  in doc but not this run ({len(gone)}): {', '.join(gone[:5])}")
    if new:
        print(f"  new in this run ({len(new)}): {', '.join(new[:5])}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="output of pytest -q --durations=0")
    p.add_argument("-o", "--out", help="write the markdown document here")
    p.add_argument("--wall", type=float, help="wall seconds if not parseable from the log")
    p.add_argument("--box", default="chicago (RTX 5090)")
    p.add_argument("--at", default="the current working tree",
                   help="commit or tree description for the header")
    p.add_argument("--stack", default="torch 2.13.0+cu130 / triton 3.7.1")
    p.add_argument("--per-test", type=float, default=1.0,
                   help="emit per-test tables for files at/above this total (0 = off)")
    p.add_argument("--compare", metavar="DOC",
                   help="diff per-file totals against an existing document")
    p.add_argument("--limit", type=int, default=25, help="rows in --compare")
    a = p.parse_args()

    per_test, summary, wall = parse(a.log)
    if not per_test:
        raise SystemExit(f"no duration lines in {a.log} — was --durations=0 passed?")
    wall = a.wall or wall
    files = by_file(per_test)
    print(f"parsed {len(per_test)} tests across {len(files)} files; "
          f"accounted {sum(sum(v.values()) for v in per_test.values()):.0f}s"
          + (f"; wall {wall:.0f}s" if wall else ""))
    if a.compare:
        compare(files, a.compare, a.limit)
    if a.out:
        Path(a.out).write_text(render(per_test, files, summary, wall, a) + "\n")
        print(f"wrote {a.out}")


main()
