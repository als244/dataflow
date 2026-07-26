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


def frontier(pred, meas, fname, opt):
    """Throughput vs GPU memory with tokens-per-round OPTIMISED.

    Solid line + circles: simulated frontier. Dashed line + stars: the same
    cells measured on the hardware. The two curves are the report's claim.

    Tokens-per-round is a knob the operator sets, not a property of the
    hardware, so the useful question is not "how fast is a 64K round at 8 GiB"
    (nobody would choose that) but "how fast can this budget go, and which round
    size gets there". Each point is the best round size for that budget, and the
    label says which one won."""
    rows = [r for r in pred if "tok_s" in r]
    if not rows:
        return
    seqs = sorted({r["seq"] for r in rows})
    tss = sorted({r["t_step"] for r in rows})
    backs = sorted({r.get("backing") for r in rows}, key=lambda b: (b is None, b))
    fig, axes = plt.subplots(1, len(tss), squeeze=False, sharey=True,
                            figsize=(4.2 * len(tss) + 1, 4.4))
    cmap = plt.get_cmap("viridis")
    for j, ts in enumerate(tss):
        ax = axes[0][j]
        for si, sq in enumerate(seqs):
            colour = cmap(si / max(1, len(seqs) - 1))
            for bi, bk in enumerate(backs):
                best = {}
                for r in rows:
                    if r["seq"] != sq or r["t_step"] != ts or r.get("backing") != bk:
                        continue
                    cur = best.get(r["budget"])
                    if cur is None or r["tok_s"] > cur["tok_s"]:
                        best[r["budget"]] = r
                if not best:
                    continue
                pts = [best[b] for b in sorted(best)]
                ax.plot([p["budget"] for p in pts], [p["tok_s"] for p in pts],
                        STYLES[bi % len(STYLES)], marker="o", ms=4, lw=1.6,
                        color=colour,
                        label=(f"seq {sq}" if bi == 0 and j == 0 else None))
                if bi == len(backs) - 1:
                    for p in pts:
                        ax.annotate(f"{p['t_round'] // 1024}K",
                                    (p["budget"], p["tok_s"]), fontsize=6,
                                    textcoords="offset points", xytext=(0, 5),
                                    ha="center", color=colour)
        # Measured cells, joined into a MEASURED FRONTIER wherever there is
        # more than one budget for a sequence. Every measured cell is now the
        # frontier pick for its (seq, tokens/step, budget), so the points form
        # the real curve and can be read against the simulated one directly --
        # a scatter of stars leaves the eye to guess whether the shapes agree.
        by_seq = {}
        for m in meas:
            if m.get("t_step") == ts and "tok_s" in m:
                by_seq.setdefault(m["seq"], []).append(m)
        for sq, cells in by_seq.items():
            si = seqs.index(sq) if sq in seqs else 0
            colour = cmap(si / max(1, len(seqs) - 1))
            cells.sort(key=lambda m: m["budget"])
            if len(cells) > 1:
                ax.plot([m["budget"] for m in cells], [m["tok_s"] for m in cells],
                        "--", lw=1.6, color=colour, zorder=5,
                        label="measured" if (si == 0 and j == 0) else None)
            ax.plot([m["budget"] for m in cells], [m["tok_s"] for m in cells],
                    "*", ms=13, color=colour, markeredgecolor="k",
                    markeredgewidth=0.6, zorder=6, linestyle="none")
        ax.set_xscale("log", base=2)
        ticks = sorted({r["budget"] for r in rows})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
        ax.set_title(f"{ts // 1024}K tokens/step", fontsize=10)
        ax.set_xlabel("GPU memory budget (GiB)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("tok/s  (best tokens/round)")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Achievable throughput vs GPU memory — {opt}\n{env_note()}"
                 f"  ·  labels = winning tokens/round"
                 + ("  ·  ★ measured" if meas else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, fname)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}   (frontier over tokens/round)")



def frontier_by_peak(meas, fname, opt):
    """Measured throughput against the device memory the run ACTUALLY used.

    The budget axis is what the planner was given; this is what the card had
    to have. They differ by a consistent margin -- the engine's placement
    extent above the plan's peak, torch's kernel workspace, and the CUDA
    context, none of which the budget covers -- so a reader sizing hardware
    from the budget plot would under-provision by several GiB. Measured cells
    only: nothing predicts device peak, it is observed."""
    rows = [m for m in meas if "tok_s" in m and m.get("device_peak_gib")]
    if not rows:
        return
    seqs = sorted({r["seq"] for r in rows})
    tss = sorted({r["t_step"] for r in rows})
    fig, axes = plt.subplots(1, len(tss), squeeze=False, sharey=True,
                             figsize=(4.2 * len(tss) + 1, 4.4))
    cmap = plt.get_cmap("viridis")
    for j, ts in enumerate(tss):
        ax = axes[0][j]
        for si, sq in enumerate(seqs):
            cells = sorted((r for r in rows if r["seq"] == sq and r["t_step"] == ts),
                           key=lambda r: r["device_peak_gib"])
            if not cells:
                continue
            colour = cmap(si / max(1, len(seqs) - 1))
            ax.plot([c["device_peak_gib"] for c in cells],
                    [c["tok_s"] for c in cells], "-", marker="*", ms=13,
                    lw=1.6, color=colour, markeredgecolor="k",
                    markeredgewidth=0.6,
                    label=(f"seq {sq}" if j == 0 else None))
            for c in cells:      # what the planner was told, for reference
                ax.annotate(f"{c['budget']:g}", (c["device_peak_gib"], c["tok_s"]),
                            fontsize=6, textcoords="offset points",
                            xytext=(0, 6), ha="center", color=colour)
        ax.set_title(f"{ts // 1024}K tokens/step", fontsize=10)
        ax.set_xlabel("peak device memory actually used (GiB)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("tok/s  (measured)")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Measured throughput vs DEVICE MEMORY USED — {opt}\n{env_note()}"
                 f"  ·  small labels = the budget the planner was given",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, fname)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}   (throughput vs measured device peak)")



def time_budget(pred, fname, opt):
    """Where the step's time goes, on ONE pair of axes per geometry.

    Recompute share and idle share were separate figures, which is the wrong
    split: they are two slices of the same makespan and the question is how
    they TRADE. A tight budget buys back memory by recomputing, and the plan
    is only worth it if the recompute it adds costs less than the idle it
    removes -- which you cannot see with the two curves on different pages.
    Plotted at the frontier round for each budget, since that is the round an
    operator would actually run."""
    rows = [r for r in pred if "tok_s" in r and "rc_pct" in r and "idle_pct" in r]
    if not rows:
        print(f"skip {fname}: no rows carry rc_pct/idle_pct")
        return
    seqs = sorted({r["seq"] for r in rows})
    tss = sorted({r["t_step"] for r in rows})
    fig, axes = plt.subplots(len(seqs), len(tss), squeeze=False, sharex=True,
                             sharey=True,
                             figsize=(3.5 * len(tss) + 1.5, 2.5 * len(seqs) + 1.4))
    for i, sq in enumerate(seqs):
        for j, ts in enumerate(tss):
            ax = axes[i][j]
            best = {}
            for r in rows:
                if r["seq"] != sq or r["t_step"] != ts:
                    continue
                b = r["budget"]
                if b not in best or r["tok_s"] > best[b]["tok_s"]:
                    best[b] = r
            pts = [best[b] for b in sorted(best)]
            if not pts:
                continue
            xs = [p["budget"] for p in pts]
            ax.plot(xs, [p["rc_pct"] for p in pts], "-o", ms=3.5, lw=1.6,
                    color="#d1495b",
                    label="recompute" if (i == 0 and j == 0) else None)
            ax.plot(xs, [p["idle_pct"] for p in pts], "--s", ms=3.5, lw=1.6,
                    color="#00798c",
                    label="idle" if (i == 0 and j == 0) else None)
            ax.plot(xs, [p["rc_pct"] + p["idle_pct"] for p in pts], ":", lw=1.2,
                    color="0.35",
                    label="both" if (i == 0 and j == 0) else None)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{ts // 1024}K tokens/step", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"seq {sq}\n% of makespan", fontsize=9)
            if i == len(seqs) - 1:
                ax.set_xlabel("GPU memory budget (GiB)")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Where the step's time goes — {opt}\n{env_note()}"
                 f"  ·  frontier round at each budget", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, fname)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}   (recompute + idle on one axis)")


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
    # dominated controls are deliberately off the frontier -- they exist to
    # check the RANKING, not the curve. Drawn against a frontier envelope they
    # read as fidelity misses, which is the opposite of what they show.
    meas = [m for m in meas if "dominated_control" not in m.get("spines", [])]
    feasible = [r for r in pred if "tok_s" in r]
    print(f"{a.opt}: {len(pred)} predicted rows "
          f"({len(feasible)} feasible, {len(pred) - len(feasible)} infeasible), "
          f"{len([m for m in meas if 'meas_s' in m])} measured")
    if not feasible:
        raise SystemExit("no feasible predictions yet")

    facet(feasible, [m for m in meas if "tok_s" in m], "tok_s", "tok/s",
          f"Predicted throughput vs GPU memory — {a.opt} ({layer})",
          f"throughput_{a.opt}.png")
    frontier(feasible, [m for m in meas if "tok_s" in m],
             f"frontier_{a.opt}.png", a.opt)
    frontier_by_peak([m for m in meas if "tok_s" in m],
                     f"frontier_by_peak_{a.opt}.png", a.opt)
    time_budget(feasible, f"time_budget_{a.opt}.png", a.opt)
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
