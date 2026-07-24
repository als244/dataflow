#!/usr/bin/env python
"""Turn one or more sweep runs into a written report.

    python report.py                       # this machine's run
    python report.py --runs "80GB card"=. "24GB card"=../other --out REPORT.md

Runs are labelled by their device unless a name is given, so a report describes
hardware rather than whoever's machines produced it.

Everything quantitative in the report comes from the JSONL the sweep wrote, so
the numbers cannot drift from the run that produced them. Prose that interprets
those numbers belongs in the report file afterwards; this generates the
skeleton, the tables, the figures and the headline statistics.

Sections, in order: what was run, the throughput landscape, how the planner
spends the budget, whether the simulator tells the truth, whether the host
allowance binds, and — when given more than one run — how the machines compare.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    """One box's results."""

    name: str
    root: Path

    @property
    def env(self) -> dict:
        path = self.root / "env.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def rows(self, kind: str, opt: str) -> list[dict]:
        path = self.root / "data" / f"{kind}_{opt}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.open()]

    def opts(self) -> list[str]:
        found = [p.name.split("_")[-1].removesuffix(".jsonl")
                 for p in (self.root / "data").glob("predict_measured_*.jsonl")]
        return sorted(found)


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def frontier(rows: list[dict]) -> dict:
    """Best tokens-per-round for each (seq, tokens/step, budget) — the choice an
    operator would actually make, since round size is theirs to pick."""
    best: dict = {}
    for r in rows:
        if "tok_s" not in r:
            continue
        slot = (r["seq"], r["t_step"], r["budget"])
        if slot not in best or r["tok_s"] > best[slot]["tok_s"]:
            best[slot] = r
    return best


# ------------------------------------------------------------- sections ---

def section_setup(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        e = run.env
        link = e.get("link", {})
        rows.append([
            run.name, e.get("device", "?"), f"{e.get('device_gib', '?')} GiB",
            f"{e.get('host_limit_gib', '?')} GiB ({e.get('host_limit_source', '?')})",
            f"{link.get('bidi_h2d_gbs', '?')}/{link.get('bidi_d2h_gbs', '?')}",
            e.get("preset", "?"), f"{e.get('backing_gib', '?')} GiB",
            ", ".join(f"{b:g}" for b in e.get("budgets", [])),
        ])
    body = table(["run", "device", "device memory", "host limit", "link H2D/D2H GB/s",
                  "model", "allowance", "GPU memory budgets (GiB)"], rows)
    return ("## What was run\n\n"
            "Each machine chose its own budgets and allowance from what it "
            "actually has; the model is the largest whose parameters, optimizer "
            "state and gradients fit the host.\n\n" + body + "\n\n"
            "The link rate is the engine's own measurement with both directions "
            "in flight, which is what plans price transfers at.\n")


def section_landscape(run: Run, opt: str) -> str:
    pred = run.rows("predict_measured", opt)
    feas = [r for r in pred if "tok_s" in r]
    if not feas:
        return ""
    front = frontier(feas)
    best = max(front.values(), key=lambda r: r["tok_s"])
    seqs = sorted({r["seq"] for r in front.values()})
    budgets = sorted({r["budget"] for r in front.values()})
    steps = sorted({r["t_step"] for r in front.values()})
    ts = steps[len(steps) // 2]      # the middle tokens-per-step, as the grid

    rows = []
    for sq in seqs:
        line = [sq]
        for b in budgets:
            cell = front.get((sq, ts, b))
            line.append(f"{cell['tok_s']:,.0f}<br><sub>{cell['t_round'] // 1024}K</sub>"
                        if cell else "—")
        rows.append(line)
    grid = table(["seq \\ budget"] + [f"{b:g} GiB" for b in budgets], rows)

    return (f"### Throughput vs GPU memory ({opt})\n\n"
            f"Tokens/s at each budget with the round size optimised (the winning "
            f"tokens-per-round is the small number underneath), at "
            f"{ts // 1024}K tokens/step.\n\n{grid}\n\n"
            f"Peak: **{best['tok_s']:,.0f} tok/s** "
            f"({best['eff_tfs']:.0f} effective TFLOP/s) at seq {best['seq']}, "
            f"{best['t_round'] // 1024}K tokens/round, "
            f"{best['budget']:g} GiB budget. "
            f"{len(feas)} of {len(pred)} grid cells were feasible; the rest could "
            f"not be planned at that budget and are recorded with the planner's "
            f"reason.\n\n"
            f"![throughput](figs/frontier_{opt}.png)\n")


def section_planner(run: Run, opt: str) -> str:
    feas = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
    if not feas:
        return ""
    budgets = sorted({r["budget"] for r in feas})
    rows = []
    for b in budgets:
        at = [r for r in feas if r["budget"] == b]
        rows.append([f"{b:g}",
                     f"{statistics.fmean(r['rc_pct'] for r in at):.0f}%",
                     f"{statistics.fmean(r['idle_pct'] for r in at):.0f}%",
                     f"{statistics.fmean(r['h2d_pct'] for r in at):.0f}%",
                     f"{statistics.fmean(r['d2h_pct'] for r in at):.0f}%",
                     f"{statistics.fmean(r['recompute'] / max(1, r['rewritable']) for r in at) * 100:.0f}%"])
    return ("### How the planner spends a tight budget\n\n"
            "Averaged over every feasible cell at each budget. Recompute and "
            "idle are shares of makespan (time), not counts of tasks.\n\n"
            + table(["budget GiB", "recompute time", "idle", "H2D duty",
                     "D2H duty", "layers recomputed"], rows) + "\n\n"
            f"![recompute](figs/recompute_pct_{opt}.png)\n")


def section_fidelity(runs: list[Run]) -> str:
    blocks = ["## Does the simulator tell the truth?\n",
              "Every measured cell was planned with the same profiled costs and "
              "the same host allowance the run actually had, so predicted and "
              "measured describe one plan rather than two.\n"]
    for run in runs:
        for opt in run.opts():
            meas = [m for m in run.rows("measure", opt) if "meas_s" in m]
            if not meas:
                continue
            ratios = sorted(m["ratio"] for m in meas)
            rows = [[f"{m['seq']}", f"{m['t_round'] // 1024}K",
                     f"{m['t_step'] // 1024}K", f"{m['budget']:g}",
                     f"{m['pred_s']:.2f}", f"{m['meas_s']:.2f}",
                     f"**{m['ratio']:.2f}**", f"{m['tok_s']:,.0f}",
                     f"{m['eff_tfs']:.0f}",
                     ",".join(m.get("spines", []))]
                    for m in sorted(meas, key=lambda m: (m["seq"], m["budget"]))]
            failed = [m for m in run.rows("measure", opt) if "failed" in m]
            blocks.append(
                f"### {run.name} · {opt}\n\n"
                + table(["seq", "t/round", "t/step", "budget", "pred s", "meas s",
                         "meas/pred", "tok/s", "effTF", "role"], rows)
                + f"\n\nMedian ratio **{statistics.median(ratios):.2f}**, "
                  f"range {ratios[0]:.2f}–{ratios[-1]:.2f} over {len(ratios)} cells"
                + (f"; {len(failed)} cells failed to run." if failed else ".") + "\n")
    return "\n".join(blocks)


def section_host(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        for opt in run.opts():
            feas = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
            if not feas:
                continue
            bind = [r for r in feas if r.get("binding")]
            gains = sorted(r["host_marginal_gain"] for r in feas
                           if r.get("host_marginal_gain") is not None)
            rows.append([
                f"{run.name} · {opt}", f"{run.env.get('backing_gib', '?')} GiB",
                f"{len(bind)}/{len(feas)}",
                f"{statistics.median(gains) * 100:+.1f}%" if gains else "—",
                f"{gains[-1] * 100:+.1f}%" if gains else "—",
                f"{sum(1 for g in gains if g > 0.02)}" if gains else "—"])
    if not rows:
        return ""
    return ("## Does the host allowance bind?\n\n"
            "How much host memory a plan *wants* is only defined when host "
            "memory is free, so the allowance is set from the machine and its "
            "effect measured: whether plans hit it, and what the same cell does "
            "when re-planned with 25% more room.\n\n"
            + table(["run", "allowance", "cells binding", "median value of +25%",
                     "best", "cells gaining >2%"], rows) + "\n")


def section_compare(runs: list[Run], opt: str) -> str:
    if len(runs) < 2:
        return ""
    rows = []
    for run in runs:
        feas = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
        if not feas:
            continue
        front = frontier(feas)
        best = max(front.values(), key=lambda r: r["tok_s"])
        e = run.env
        per_gib = best["tok_s"] / max(1e-9, best["budget"])
        rows.append([run.name, e.get("device", "?"),
                     f"{best['tok_s']:,.0f}", f"{best['eff_tfs']:.0f}",
                     f"{best['budget']:g}", f"{per_gib:,.0f}",
                     f"{statistics.fmean(r['rc_pct'] for r in feas):.0f}%"])
    return ("## How the machines compare\n\n"
            f"Best achievable throughput on each box ({opt}), with round size "
            "optimised. Tokens per second per GiB of GPU memory is the figure "
            "that matters when the point is training under a memory ceiling.\n\n"
            + table(["run", "device", "peak tok/s", "effective TFLOP/s",
                     "at budget GiB", "tok/s per GiB", "mean recompute time"],
                    rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", default=None,
                    help="name=path pairs; default: this directory")
    ap.add_argument("--opt", default="adamw", help="optimizer for the landscape")
    ap.add_argument("--out", default=None, help="default: REPORT.md here")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    if a.runs:
        runs = [Run(*spec.split("=", 1)) for spec in a.runs]
        runs = [Run(r.name, Path(r.root)) for r in runs]
    else:
        env_path = here / "env.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else {}
        runs = [Run(env.get("device", "this machine"), here)]

    parts = ["# Training throughput under a GPU memory ceiling\n",
             "*Generated from the sweep's own output; every number below comes "
             "from the JSONL files it wrote.*\n",
             section_setup(runs)]
    primary = runs[0]
    parts.append("## The throughput landscape\n")
    parts.append(section_landscape(primary, a.opt))
    parts.append(section_planner(primary, a.opt))
    parts.append(section_compare(runs, a.opt))
    parts.append(section_fidelity(runs))
    parts.append(section_host(runs))
    parts.append("## Reproducing this\n\n```bash\n"
                 "python reproducibility/throughput_fidelity/run_experiment.py\n"
                 "python reproducibility/throughput_fidelity/report.py\n```\n\n"
                 "See that directory's README for the stage-by-stage description "
                 "and every configuration flag.\n")

    out = Path(a.out) if a.out else here / "REPORT.md"
    out.write_text("\n".join(p for p in parts if p))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
