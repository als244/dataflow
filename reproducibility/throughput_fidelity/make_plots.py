#!/usr/bin/env python
"""Visualizations for the llama3_8b throughput/fidelity sweep.

Headline: throughput vs MEMORY BUDGET, small-multiples faceted by
seqlen (rows) x tokens/step (cols), with t_round as the within-panel curve
family. Companion panels: recompute% (TIME = rc_pct = recompute_us/makespan,
the webapp's number) and idle%. Measured points overlay the predicted curves.

Reads whatever JSONL layers exist in data/ (predict_roofline / predict_measured
/ measure). Robust to partial data (prototypes on whatever is present).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def load(name):
    p = os.path.join(DATA, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def facet(pred, meas, metric, ylabel, title, fname, *, pct=False):
    """rows=seqlen, cols=tokens/step, x=budget, hue=t_round."""
    rows = [r for r in pred if metric in r and "budget" in r]
    if not rows:
        print(f"skip {fname}: no rows with {metric}")
        return
    seqs = sorted({r["seq"] for r in rows})
    tss = sorted({r["t_step"] for r in rows})
    trs = sorted({r["t_round"] for r in rows})
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(len(seqs), len(tss), squeeze=False, sharex=True,
                           figsize=(3.4 * len(tss) + 1, 2.6 * len(seqs) + 1))
    meas_by = {}
    for m in meas:
        if metric in m or metric in ("tok_s", "eff_tfs", "hw_tfs"):
            meas_by.setdefault((m.get("seq"), m.get("t_step"), m.get("t_round")), []).append(m)
    for i, sq in enumerate(seqs):
        for j, ts in enumerate(tss):
            a = ax[i][j]
            for k, tr in enumerate(trs):
                pts = sorted([r for r in rows if r["seq"] == sq and r["t_step"] == ts
                              and r["t_round"] == tr], key=lambda r: r["budget"])
                if pts:
                    a.plot([p["budget"] for p in pts], [p[metric] for p in pts],
                           "o-", color=cmap(k / max(1, len(trs) - 1)),
                           label=f"tr={tr//1024}K", ms=4, lw=1.4)
                mm = sorted([m for m in meas_by.get((sq, ts, tr), []) if metric in m],
                            key=lambda m: m["budget"])
                if mm:
                    a.plot([m["budget"] for m in mm], [m[metric] for m in mm],
                           "*", color=cmap(k / max(1, len(trs) - 1)), ms=12,
                           markeredgecolor="k", markeredgewidth=0.5, zorder=5)
            a.set_xscale("log", base=2)
            a.set_xticks([4, 8, 16, 32, 64])
            a.set_xticklabels([4, 8, 16, 32, 64])
            if pct:
                a.set_ylim(0, 100)
            a.grid(True, alpha=0.3)
            if i == 0:
                a.set_title(f"tok/step={ts//1024}K", fontsize=10)
            if i == len(seqs) - 1:
                a.set_xlabel("fast budget (GiB)")
            if j == 0:
                a.set_ylabel(f"seq={sq}\n{ylabel}", fontsize=9)
    ax[0][-1].legend(fontsize=7, title="t_round", loc="best")
    fig.suptitle(title + "   (★ = measured)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIGS, fname)
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print("wrote", out, f"({len(seqs)}x{len(tss)} facets, {len(trs)} t_round curves)")


def main():
    opt = sys.argv[1] if len(sys.argv) > 1 else "adamw"
    # prefer measured-cost predictions for the landscape; fall back to roofline
    pred = load(f"predict_measured_{opt}.jsonl") or load(f"predict_roofline_{opt}.jsonl")
    meas = load(f"measure_{opt}.jsonl")
    layer = "measured-cost" if load(f"predict_measured_{opt}.jsonl") else "ROOFLINE(proxy)"
    print(f"opt={opt}: {len(pred)} predicted ({layer}), {len(meas)} measured")
    facet(pred, meas, "tok_s", "tok/s", f"llama3_8b {opt}: throughput vs budget [{layer}]",
          f"throughput_{opt}.png")
    facet(pred, meas, "eff_tfs", "eff TFLOP/s", f"llama3_8b {opt}: effective TFLOP/s vs budget [{layer}]",
          f"eff_tflops_{opt}.png")
    facet(pred, [], "rc_pct", "recompute % of makespan",
          f"llama3_8b {opt}: recompute-time % vs budget [{layer}]", f"recompute_pct_{opt}.png", pct=True)
    facet(pred, [], "idle_pct", "idle % of makespan",
          f"llama3_8b {opt}: idle % vs budget [{layer}]", f"idle_pct_{opt}.png", pct=True)


if __name__ == "__main__":
    main()
