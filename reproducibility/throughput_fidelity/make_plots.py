#!/usr/bin/env python
"""Figures for the throughput / fidelity sweep.

Headline: throughput against GPU memory budget — the axis this runtime exists
to move — as small multiples over sequence length (rows) and tokens per step
(columns), with tokens-per-round as the curve family inside each panel and host
allowance as the line style. Companions explain the shape: recompute as a share
of MAKESPAN (time, the webapp's number — not a count of rewritten tasks) and
idle share. Measured cells overlay as stars wherever they exist.

    python make_plots.py [adamw|muon] [--layer auto|measured|unlimited]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
STYLES = ["-", "--", ":", "-."]


def load(name):
    path = os.path.join(DATA, name)
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def env_note():
    path = os.path.join(HERE, "env.json")
    if not os.path.exists(path):
        return ""
    e = json.load(open(path))
    return (f"{e.get('host', '?')}  ·  {e.get('device', '?')}  ·  "
            f"{e.get('preset', '?')}  ·  host {e.get('host_limit_gib', '?')} GiB")


def facet(pred, meas, metric, ylabel, title, fname, *, pct=False):
    rows = [r for r in pred if metric in r and "budget" in r]
    if not rows:
        print(f"skip {fname}: no rows carry {metric}")
        return
    seqs = sorted({r["seq"] for r in rows})
    tss = sorted({r["t_step"] for r in rows})
    trs = sorted({r["t_round"] for r in rows})
    backs = sorted({r.get("backing") for r in rows}, key=lambda b: (b is None, b))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(seqs), len(tss), squeeze=False, sharex=True,
                             figsize=(3.5 * len(tss) + 1.5, 2.7 * len(seqs) + 1.4))
    by_panel = {}
    for m in meas:
        if metric in m:
            by_panel.setdefault((m.get("seq"), m.get("t_step"), m.get("t_round")),
                                []).append(m)
    for i, sq in enumerate(seqs):
        for j, ts in enumerate(tss):
            ax = axes[i][j]
            for ti, tr in enumerate(trs):
                colour = cmap(ti / max(1, len(trs) - 1))
                for bi, bk in enumerate(backs):
                    pts = sorted([r for r in rows
                                  if r["seq"] == sq and r["t_step"] == ts
                                  and r["t_round"] == tr and r.get("backing") == bk],
                                 key=lambda r: r["budget"])
                    if not pts:
                        continue
                    ax.plot([p["budget"] for p in pts], [p[metric] for p in pts],
                            STYLES[bi % len(STYLES)], marker="o", ms=3.5, lw=1.4,
                            color=colour,
                            label=(f"{tr // 1024}K" if bi == 0 and i == 0 and j == 0
                                   else None))
            for (msq, mts, mtr), cells in by_panel.items():
                if msq != sq or mts != ts:
                    continue
                colour = cmap((trs.index(mtr) if mtr in trs else 0)
                              / max(1, len(trs) - 1))
                cells = sorted(cells, key=lambda m: m["budget"])
                ax.plot([m["budget"] for m in cells], [m[metric] for m in cells],
                        "*", ms=13, color=colour, markeredgecolor="k",
                        markeredgewidth=0.6, zorder=6)
            ax.set_xscale("log", base=2)
            ticks = sorted({r["budget"] for r in rows})
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
            if pct:
                ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{ts // 1024}K tokens/step", fontsize=10)
            if i == len(seqs) - 1:
                ax.set_xlabel("GPU memory budget (GiB)")
            if j == 0:
                ax.set_ylabel(f"seq {sq}\n{ylabel}", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(handles, labels, fontsize=7, title="tokens/round",
                           loc="best")
    style_note = ""
    real_backs = [b for b in backs if b is not None]
    if len(real_backs) > 1:
        style_note = "  ·  line style = host allowance " + ", ".join(
            f"{STYLES[i % len(STYLES)]} {b:g} GiB" for i, b in enumerate(real_backs))
    star = "  ·  ★ measured" if meas else ""
    fig.suptitle(f"{title}\n{env_note()}{style_note}{star}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, fname)
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"wrote {out}   ({len(seqs)}x{len(tss)} panels, {len(trs)} curves, "
          f"{len(real_backs)} allowances, {len(rows)} cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opt", nargs="?", default="adamw")
    ap.add_argument("--layer", default="auto",
                    choices=["auto", "measured", "unlimited"])
    a = ap.parse_args()

    pred, layer = [], ""
    if a.layer in ("auto", "measured"):
        pred = load(f"predict_measured_{a.opt}.jsonl")
        layer = "plans under a host allowance"
    if not pred and a.layer != "measured":
        pred = load(f"predict_unlimited_{a.opt}.jsonl")
        layer = "host allowance unconstrained"
    meas = load(f"measure_{a.opt}.jsonl")
    feasible = [r for r in pred if "tok_s" in r]
    print(f"{a.opt}: {len(pred)} predicted rows "
          f"({len(feasible)} feasible, {len(pred) - len(feasible)} infeasible), "
          f"{len([m for m in meas if 'meas_s' in m])} measured")
    if not feasible:
        raise SystemExit("no feasible predictions yet")

    facet(feasible, [m for m in meas if "tok_s" in m], "tok_s", "tok/s",
          f"Predicted throughput vs GPU memory — {a.opt} ({layer})",
          f"throughput_{a.opt}.png")
    facet(feasible, [m for m in meas if "eff_tfs" in m], "eff_tfs",
          "effective TFLOP/s",
          f"Predicted effective TFLOP/s vs GPU memory — {a.opt}",
          f"eff_tflops_{a.opt}.png")
    facet(feasible, [], "rc_pct", "recompute % of makespan",
          f"Time spent recomputing vs GPU memory — {a.opt}",
          f"recompute_pct_{a.opt}.png", pct=True)
    facet(feasible, [], "idle_pct", "idle % of makespan",
          f"Compute idle vs GPU memory — {a.opt}",
          f"idle_pct_{a.opt}.png", pct=True)


if __name__ == "__main__":
    main()
