"""Campaign driver: families x envelopes x placement modes -> mode-pure tables.

Reproduces the DSA-round-5 results protocol end to end. One command runs
(or re-renders) a full benchmark matrix and emits the two mode-pure
markdown tables (static / vmm) with per-cell detail:

    wall tok/s (sim tok/s) . measured peak GiB . bsXgaY . rc%

plus a best-legal-per-cell summary table.

Protocol (what a row means):
- Every row is ENVELOPE-LEGAL: bench_train's auto-headroom closing loop
  measures the true device peak (fixed + placement extent + torch
  reserved high-water) after the run and, on a bust, shrinks the ledger
  by the measured overage and re-runs — no hand-tuned leeway constants.
  Rows additionally carry envelope_ok; this renderer refuses illegal
  rows unless --allow-illegal (they render with a warning flag).
- Shapes (bs/ga) are chosen per (family, envelope) and HELD FIXED across
  placement modes so static-vs-vmm isolates placement, not shape.
- rc% = recomputed layer-rounds / total layer-rounds (rc_chosen / (L*ga)).
- sim = the simulator's prediction for the chosen plan (same profiles).

Shape sources (--shapes):
- "cached" (default): scan existing summaries; per (family, dev) take
  the shape of the best LEGAL wall tok/s seen in any mode.
- "oracle": invoke tools/best_config.py per (family, dev) and take its
  winner (expensive: runs its own sweep).
- explicit map: "12:bs4ga4,16:bs8ga2,20:bs8ga2,24:bs16ga1,28:bs16ga1".

Usage (reproduce the dsa-round5 campaign):
    python tools/bench_campaign.py \
        --families dsv3-mini,dsv32-mini,glm52-mini \
        --seq-tag s4k --device-gib 12,16,20,24,28 \
        --placements static,vmm --steps 3 \
        --shapes 12:bs4ga4,16:bs8ga2,20:bs8ga2,24:bs16ga1,28:bs16ga1 \
        --run --pace-seconds 40 \
        --out results/bench/dsa-round5/TABLES.md

Render-only (rebuild tables from existing artifacts, no GPU):
    python tools/bench_campaign.py --families ... --device-gib ... \
        --placements static,vmm --render-only --out -

Cells are run in SUBPROCESSES (one bench_train invocation per family x
shape x mode, envelopes sharing a shape batched into one invocation to
amortize host pinning), paced by --pace-seconds to stay under
systemd-oomd pressure limits during pinning churn. Existing legal rows
are skipped unless --rerun. Config names follow the repo convention
"{family}-{seq_tag}-{shape}" and must exist in bench_train CONFIGS.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUMMARY_DIRS = [REPO / "artifacts/bench", REPO / "artifacts/m4"]


def load_cells(families: list[str], allow_illegal: bool) -> dict:
    """Newest legal row per (family, dev, mode) from all summary dirs."""
    best: dict[tuple[str, int, str], dict] = {}
    for d in SUMMARY_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.summary.json"):
            m = re.search(r"(" + "|".join(map(re.escape, families)) + r")-(s\d+k)-(bs\d+ga\d+)", f.name)
            if not m:
                continue
            fam, shape = m.group(1), m.group(3)
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            mt = f.stat().st_mtime
            for r in data.get("sweep", []):
                if r.get("status"):
                    continue
                dev = int(r.get("device_envelope_gib") or 0)
                if not dev:
                    continue
                mode = r.get("placement_mode", "static")
                if r.get("extent_budget"):
                    mode = "static"
                ok = bool(r.get("envelope_ok",
                                r.get("actual_device_peak_gib", 99) <= dev + 0.02))
                if not ok and not allow_illegal:
                    continue
                cand = dict(
                    wall=r["wall_tokens_per_s"], sim=r["sim_tokens_per_s"],
                    peak=r["actual_device_peak_gib"], shape=shape,
                    rc=r["recompute_chosen"], ok=ok, mt=mt,
                    layers=r.get("n_layers"),
                    row=r, config=data.get("config"), summary=str(f),
                )
                key = (fam, dev, mode)
                prev = best.get(key)
                if prev is None or (cand["ok"] and not prev["ok"]) or \
                        (cand["ok"] == prev["ok"] and mt > prev["mt"]):
                    best[key] = cand
    return best


def shapes_cached(cells: dict, families: list[str], devs: list[int]) -> dict:
    out: dict[tuple[str, int], str] = {}
    for fam in families:
        for dev in devs:
            rows = [c for (f, d, _m), c in cells.items()
                    if f == fam and d == dev and c["ok"]]
            if rows:
                out[(fam, dev)] = max(rows, key=lambda c: c["wall"])["shape"]
    return out


def shapes_oracle(families: list[str], devs: list[int], seq_tag: str,
                  seqs_per_step: int) -> dict:
    seq_len = int(seq_tag[1:-1]) * 1024
    out: dict[tuple[str, int], str] = {}
    for fam in families:
        res = subprocess.run(
            [sys.executable, str(REPO / "tools/best_config.py"),
             "--family", fam.split("-")[0], "--preset", fam,
             "--seq-len", str(seq_len), "--seqs-per-step", str(seqs_per_step),
             "--device-gib", ",".join(map(str, devs)), "--json", "-"],
            capture_output=True, text=True, check=True)
        winners = json.loads(res.stdout)
        for dev_s, w in winners.items():
            out[(fam, int(float(dev_s)))] = f"bs{w['batch']}ga{w['rounds']}"
    return out


def run_cells(families, devs, modes, shapes, seq_tag, steps, pace,
              rerun, cells, dry_run):
    """One bench_train subprocess per (family, shape, mode); envelopes
    sharing a shape batched into a single invocation."""
    for fam in families:
        for mode in modes:
            by_shape: dict[str, list[int]] = defaultdict(list)
            for dev in devs:
                shape = shapes.get((fam, dev))
                if shape is None:
                    print(f"[skip] no shape for {fam}@{dev}", file=sys.stderr)
                    continue
                if not rerun and (fam, dev, mode) in cells:
                    continue
                by_shape[shape].append(dev)
            for shape, ds in sorted(by_shape.items()):
                cmd = [sys.executable, str(REPO / "tools/bench_train.py"),
                       "--config", f"{fam}-{seq_tag}-{shape}",
                       "--device-gib", ",".join(map(str, ds)),
                       "--steps", str(steps)]
                if mode != "static":
                    cmd += ["--placement", mode]
                print("[run]" if not dry_run else "[dry]", " ".join(cmd),
                      file=sys.stderr)
                if dry_run:
                    continue
                subprocess.run(cmd, check=False)
                time.sleep(pace)


def emit_cells(cells, families, devs, modes, out_dir: Path) -> None:
    """Per cell: measured.json (the full summary row + provenance) and
    plan.json (the annotated program actually executed — from the row's
    plan_path when present, else matched by planned ledger)."""
    import shutil
    for (fam, dev, mode), c in sorted(cells.items()):
        if fam not in families or dev not in devs or mode not in modes:
            continue
        d = out_dir / f"{fam}-{dev}gib-{mode}"
        d.mkdir(parents=True, exist_ok=True)
        measured = dict(c["row"])
        measured.update(family=fam, device_envelope_gib=dev,
                        placement_mode=mode, config=c["config"],
                        source_summary=c["summary"],
                        generated_by="tools/bench_campaign.py")
        (d / "measured.json").write_text(json.dumps(measured, indent=2) + "\n")
        plan = c["row"].get("plan_path")
        if not plan:  # legacy rows: match by planned ledger value
            led = c["row"].get("planned_budget_gib")
            if led is not None:
                hits = [q for sd in SUMMARY_DIRS for q in
                        sd.glob(f"*-{led:g}gib.annotated.json")]
                plan = str(hits[0]) if len(hits) == 1 else None
        if plan and Path(plan).exists():
            shutil.copy(plan, d / "plan.json")
        else:
            (d / "plan.MISSING").write_text(
                "no unambiguous annotated plan for this row (legacy "
                "summary without plan_path)\n")


def render(cells, families, devs, modes, labels) -> str:
    def cell(fam, dev, mode):
        c = cells.get((fam, dev, mode))
        if not c:
            return "—"
        ga = int(re.search(r"ga(\d+)", c["shape"]).group(1))
        layers = c.get("layers") or 18
        pct = 100.0 * c["rc"] / (layers * ga)
        flag = "" if c["ok"] else " ⚠ILLEGAL"
        return (f"{c['wall']:,.0f} (sim {c['sim']:,.0f}) · "
                f"{c['peak']:.2f} GiB · {c['shape']} · rc {pct:.0f}%{flag}")

    lines = []
    for mode in modes:
        lines += [f"\n### {mode.upper()} placement\n",
                  "| dev GiB | " + " | ".join(labels.get(f, f) for f in families) + " |",
                  "|" + "---|" * (len(families) + 1)]
        for dev in devs:
            row = " | ".join(cell(f, dev, mode) for f in families)
            lines.append(f"| {dev} | {row} |")
    lines += ["\n### Best legal per cell (mode in parens where not static)\n",
              "| dev GiB | " + " | ".join(labels.get(f, f) for f in families) + " |",
              "|" + "---|" * (len(families) + 1)]
    for dev in devs:
        row = []
        for fam in families:
            cands = [(m, cells[(fam, dev, m)]) for m in modes
                     if (fam, dev, m) in cells and cells[(fam, dev, m)]["ok"]]
            if not cands:
                row.append("—")
                continue
            m, c = max(cands, key=lambda mc: mc[1]["wall"])
            tag = "" if m == "static" else f" ({m})"
            row.append(f"{c['wall']:,.0f}{tag}")
        lines.append(f"| {dev} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", required=True,
                    help="comma list of config prefixes, e.g. dsv3-mini,glm52-mini")
    ap.add_argument("--seq-tag", default="s4k")
    ap.add_argument("--device-gib", required=True, help="comma list, e.g. 12,16,20")
    ap.add_argument("--placements", default="static,vmm")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--shapes", default="cached",
                    help='"cached" | "oracle" | explicit "12:bs4ga4,16:bs8ga2,..."')
    ap.add_argument("--seqs-per-step", type=int, default=16,
                    help="oracle mode only: tokens/step = this * seq_len")
    ap.add_argument("--run", action="store_true", help="execute missing cells")
    ap.add_argument("--rerun", action="store_true", help="re-execute all cells")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the bench_train commands without running")
    ap.add_argument("--pace-seconds", type=int, default=40,
                    help="sleep between invocations (oomd pressure decay)")
    ap.add_argument("--allow-illegal", action="store_true",
                    help="render envelope-busting rows (flagged) instead of dropping")
    ap.add_argument("--out", default="-", help="output markdown path, - = stdout")
    ap.add_argument("--cells-dir", default=None,
                    help="emit per-cell measured.json + plan.json under this dir")
    args = ap.parse_args()

    families = args.families.split(",")
    devs = [int(x) for x in args.device_gib.split(",")]
    modes = args.placements.split(",")
    labels = {"dsv3-mini": "dsv3 (dense MLA)", "dsv32-mini": "dsv32 (DSA)",
              "glm52-mini": "glm52 (DSA+IndexShare)"}

    cells = load_cells(families, args.allow_illegal)
    if args.shapes == "cached":
        shapes = shapes_cached(cells, families, devs)
    elif args.shapes == "oracle":
        shapes = shapes_oracle(families, devs, args.seq_tag, args.seqs_per_step)
    else:
        per_dev = dict(kv.split(":") for kv in args.shapes.split(","))
        shapes = {(f, int(d)): s for f in families for d, s in per_dev.items()}

    if (args.run or args.rerun or args.dry_run) and not args.render_only:
        run_cells(families, devs, modes, shapes, args.seq_tag, args.steps,
                  args.pace_seconds, args.rerun, cells, args.dry_run)
        if not args.dry_run:
            cells = load_cells(families, args.allow_illegal)

    md = render(cells, families, devs, modes, labels)
    if args.cells_dir:
        emit_cells(cells, families, devs, modes, Path(args.cells_dir))
        md += (f"\nPer-cell artifacts (measured.json + plan.json): "
               f"`{args.cells_dir}/{{family}}-{{dev}}gib-{{mode}}/`\n")
    if args.out == "-":
        print(md)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
