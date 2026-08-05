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
    """Time shares at each budget, over the FRONTIER cells only.

    Averaging every feasible cell mixes populations that are not comparable:
    idle runs ~80% at a 1K round (nothing to hide the transfers behind) and
    ~1% at 16K, so the mean is decided by which round sizes happened to be
    feasible -- which differs per optimizer. Read across optimizers it then
    says the opposite of the truth: it showed muon idling MORE than adamw when
    muon idles less on every matched cell, because it does more compute over
    the same bytes. The frontier is one cell per (sequence, tokens-step,
    budget) for both, so the columns mean the same thing."""
    feas = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
    if not feas:
        return ""
    best: dict = {}
    for r in feas:
        slot = (r["seq"], r["t_step"], r["budget"])
        if slot not in best or r["tok_s"] > best[slot]["tok_s"]:
            best[slot] = r
    feas = list(best.values())
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
    return (f"### How the planner spends a tight budget ({opt})\n\n"
            "Averaged over the FRONTIER cells at each budget — the best round "
            "size per sequence and tokens-per-step, which is what an operator "
            "would run. Recompute and idle are shares of makespan (time), not "
            "counts of tasks.\n\n"
            + table(["budget GiB", "recompute time", "idle", "H2D duty",
                     "D2H duty", "layers recomputed"], rows) + "\n\n"
            f"![recompute](figs/recompute_pct_{opt}.png)\n")


def section_fidelity(runs: list[Run]) -> str:
    blocks = ["## Does the simulator tell the truth?\n",
              "Every measured cell EXECUTES the plan the predict stage saved: "
              "the registered program's content hash is gated against the "
              "artifact's, so a prediction and its measurement can only ever "
              "describe the same program.\n"]
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


def section_frontier_truth(run: Run, opt: str) -> str:
    """Is the SIMULATED frontier the REAL frontier?

    Measuring every frontier cell gives the measured curve, but the round
    choice itself was the simulator's. The choice survives measurement when
    the predicted margin between the best round and the runner-up exceeds
    the spread of measured/predicted ratios — a near-tie inside that spread
    could go either way on the real engine."""
    pred = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
    meas = [m for m in run.rows("measure", opt) if "meas_s" in m]
    if not pred or not meas:
        return ""
    ratios = [m["ratio"] for m in meas]
    band = max(ratios) - min(ratios)
    slots: dict = {}
    for r in pred:
        slots.setdefault((r["seq"], r["t_step"], r["budget"]), []).append(r)
    contested = [s for s, rows in slots.items() if len(rows) > 1]
    if not contested:
        return ""
    robust = 0
    for s in contested:
        ordered = sorted(slots[s], key=lambda r: -r["tok_s"])
        margin = (ordered[0]["tok_s"] - ordered[1]["tok_s"]) / ordered[0]["tok_s"]
        if margin > band:
            robust += 1
    return (f"### Is the simulated frontier the real frontier? ({run.name} · {opt})\n\n"
            f"Measured/predicted ratios span {min(ratios):.2f}–{max(ratios):.2f} "
            f"(width {band:.2f}). Of the {len(contested)} (sequence, tokens/step, "
            f"budget) slots where more than one round size was feasible, the "
            f"predicted best round beats its runner-up by MORE than that width "
            f"in {robust} ({robust / len(contested) * 100:.0f}%) — in those "
            f"slots the measured frontier's round choice is the simulator's "
            f"choice, not an artifact of measurement spread. The remaining "
            f"slots are predicted near-ties where either round is equivalent "
            f"on the real engine to within the observed fidelity.\n")


def section_caveats(runs: list[Run]) -> str:
    """Known limits of the method, with the numbers this run produced."""
    blocks = ["## Known limits of the method\n"]
    peaks = []
    for run in runs:
        for opt in run.opts():
            for m in run.rows("measure", opt):
                if "device_peak_delta_gib" in m:
                    peaks.append(m["device_peak_delta_gib"] - m["budget"])
    if peaks:
        peaks.sort()
        blocks.append(
            f"- **A budget is not a device ceiling.** A cell planned at budget "
            f"B used B {peaks[0]:+.1f}..{peaks[-1]:+.1f} GiB of device memory "
            f"(median {peaks[len(peaks) // 2]:+.1f}): placement extent above "
            f"the plan's peak, kernel workspaces held by the framework "
            f"allocator, and the CUDA context — none of it inside the "
            f"engine's budget. The `frontier_by_peak` figures plot throughput "
            f"against MEASURED device peak for exactly this reason, and the "
            f"top budget rung can OOM on a card the budget nominally fits.")
    med = {}
    for run in runs:
        for opt in run.opts():
            rs = [m["ratio"] for m in run.rows("measure", opt)
                  if "meas_s" in m]
            if rs:
                med[f"{run.name} · {opt}"] = statistics.median(rs)
    if med:
        spread = ", ".join(f"{k} {v:.3f}" for k, v in med.items())
        blocks.append(
            f"- **Profiling is cache-warm.** Task profiles time a kernel in a "
            f"back-to-back repeat loop on the same buffers (warm L2/TLB); in "
            f"a real step each block runs once on freshly-streamed weights. "
            f"The residual shows up as median ratios slightly above 1 on "
            f"compute-bound cells (medians: {spread}); trace decomposition "
            f"attributes it to compute-side costs, forward-type ops hardest.")
    if len(med) > 1:
        vals = sorted(med.items(), key=lambda kv: kv[1])
        blocks.append(
            f"- **Optimizer-specific bias is visible and constant.** The gap "
            f"between per-optimizer medians ({vals[0][0]} {vals[0][1]:.3f} vs "
            f"{vals[-1][0]} {vals[-1][1]:.3f}) is a stable offset, not "
            f"scatter — consistent with operand-value-dependent clocks: "
            f"profiles seed operands N(0,1) while real gradient-scale values "
            f"draw different power. Untested; the probe is profiling the "
            f"optimizer task at gradient scale.")
    blocks.append(
        "- **The planner needs headroom to plan at all.** The budget ladder's "
        "floor is set by the largest single task's working set (roughly twice "
        "it in practice), so the sweep cannot see budgets below that floor — "
        "the regime boundary it maps starts there.")
    return "\n".join(blocks) + "\n"


def section_host(runs: list[Run]) -> str:
    """Whether the host allowance binds, over the FRONTIER cells.

    Same reason as section_planner: the feasible set differs per optimizer and
    per box, so a fraction taken over it compares different populations. The
    frontier is one cell per (sequence, tokens-step, budget) everywhere."""
    rows = []
    for run in runs:
        for opt in run.opts():
            feas = [r for r in run.rows("predict_measured", opt) if "tok_s" in r]
            if not feas:
                continue
            best: dict = {}
            for r in feas:
                slot = (r["seq"], r["t_step"], r["budget"])
                if slot not in best or r["tok_s"] > best[slot]["tok_s"]:
                    best[slot] = r
            feas = list(best.values())
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
            "when re-planned with 25% more room. Frontier cells only, so the "
            "fractions compare like with like.\n\n"
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
                    help="name=path pairs, each path a run's results root; "
                         "default: this experiment's results/")
    ap.add_argument("--opt", default="adamw", help="optimizer for the landscape")
    ap.add_argument("--results",
                    default=Path(__file__).resolve().parents[1] / "results",
                    help="run output root (reads env.json + data/ inside)")
    ap.add_argument("--out", default=None,
                    help="default: REPORT.md in the results root")
    a = ap.parse_args()

    results = Path(a.results)
    if a.runs:
        runs = [Run(*spec.split("=", 1)) for spec in a.runs]
        runs = [Run(r.name, Path(r.root)) for r in runs]
    else:
        env_path = results / "env.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else {}
        runs = [Run(env.get("device", "this machine"), results)]

    parts = ["# Training throughput under a GPU memory ceiling\n",
             "*Generated from the sweep's own output; every number below comes "
             "from the JSONL files it wrote.*\n",
             section_setup(runs)]
    primary = runs[0]
    # Every optimizer swept gets the same treatment. Rendering the landscape
    # for one of them and the fidelity tables for all of them left the report
    # asserting things about muon that the reader could only check for adamw.
    opts = [a.opt] + [o for o in primary.opts() if o != a.opt]
    parts.append("## The throughput landscape\n")
    for opt in opts:
        parts.append(section_landscape(primary, opt))
        parts.append(section_planner(primary, opt))
    for opt in opts:
        parts.append(section_compare(runs, opt))
    parts.append(section_fidelity(runs))
    for opt in opts:
        parts.append(section_frontier_truth(primary, opt))
    parts.append(section_host(runs))
    parts.append(section_caveats(runs))
    parts.append("## Reproducing this\n\n```bash\n"
                 "python reproducibility/throughput_fidelity/run_experiment.py\n"
                 "python reproducibility/throughput_fidelity/report.py\n```\n\n"
                 "See that directory's README for the stage-by-stage description "
                 "and every configuration flag.\n")

    out = Path(a.out) if a.out else results / "REPORT.md"
    out.write_text("\n".join(p for p in parts if p))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
